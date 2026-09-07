"""Causal residual steering with directions from a frozen Jacobian lens.

The implementation mirrors Neuronpedia's additive J-lens steering semantics at
fitted layers:

* map a token's unembedding row back to source residual space as ``w_t @ J_l``;
* unit-normalize each token direction before summing multiple token directions;
* inject ``strength * ||h|| * direction`` independently at every position;
* cap each injected vector to the norm of the residual at that position; and
* leave exact BOS-token positions unchanged.

``ResidualPostSteering`` distinguishes the first model call (prompt prefill)
from later calls (incremental generation). Prompt prefill is always eligible for
steering. Later calls are steered only when ``steer_generated_tokens`` is true.
Create a fresh context, or call ``reset_sequence()``, for each new generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import torch

from .artifact import FrozenLensArtifact
from .collection import resolve_decoder_layers

MAX_STEER_INJECTION_FRACTION = 1.0
NEURONPEDIA_JLENS_STEERING_COMMIT = "df03cfee33e9023866eba6f828d9e34900932877"
_STEER_NORM_EPS = 1e-12


def _output_hidden(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported decoder-layer output type {type(output)!r}")


def _replace_output_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported decoder-layer output type {type(output)!r}")


def decoded_token_index(tokenizer: Any) -> dict[str, tuple[int, ...]]:
    """Build Neuronpedia's reverse map from exact decoded strings to token IDs.

    Decoding deliberately disables whitespace cleanup. Consequently ``" wolf"``
    and ``"wolf"`` remain different steering targets when the tokenizer exposes
    both forms.
    """

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("tokenizer must expose a positive vocabulary size")
    result: dict[str, list[int]] = {}
    for token_id in range(vocab_size):
        try:
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:  # noqa: BLE001, S112 - an undecodable ID is skipped
            continue
        if not isinstance(decoded, str):
            raise TypeError("tokenizer.decode must return a string")
        result.setdefault(decoded, []).append(token_id)
    return {decoded: tuple(token_ids) for decoded, token_ids in result.items()}


def resolve_steer_token_id(tokenizer: Any, token: str) -> int:
    """Resolve a decoded token string using Neuronpedia's exact-first policy.

    The exact whitespace-preserving form wins. If it is absent, the first
    insertion-ordered decoded form with the same stripped spelling is used. A
    true decoded-string collision resolves to its lowest token ID.
    """

    if not isinstance(token, str) or not token:
        raise ValueError("token must be a nonempty decoded token string")
    index = decoded_token_index(tokenizer)
    token_ids = index.get(token)
    if not token_ids:
        stripped = token.strip()
        for decoded, candidate_ids in index.items():
            if decoded.strip() == stripped:
                token_ids = candidate_ids
                break
    if not token_ids:
        raise ValueError(f"could not resolve steer token to a vocab ID: {token!r}")
    return min(token_ids)


def _unembedding_weight(model: Any) -> torch.Tensor:
    output_embeddings = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        output_embeddings = getter()
    if output_embeddings is None:
        output_embeddings = getattr(model, "lm_head", None)
    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("model must expose a rank-two output-embedding weight")
    return weight


def build_jlens_token_directions(
    model: Any,
    lens: FrozenLensArtifact,
    *,
    token_ids: Sequence[int],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Return Neuronpedia-style residual directions for tokens at fitted layers.

    For token ``t`` and layer ``l``, the row-oriented direction is ``w_t @ J_l``
    (equivalently ``J_l.T @ w_t`` in column notation). Each token-specific
    direction is normalized separately, then directions are summed. The sum is
    not normalized again, matching Neuronpedia's multi-token behavior.
    """

    selected_tokens = tuple(int(token_id) for token_id in token_ids)
    selected_layers = tuple(int(layer) for layer in layers)
    if not selected_tokens:
        raise ValueError("token_ids cannot be empty")
    if not selected_layers:
        raise ValueError("layers cannot be empty")
    if len(set(selected_tokens)) != len(selected_tokens):
        raise ValueError("token_ids must be unique")
    if len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layers must be unique")

    weight = _unembedding_weight(model)
    vocab_size, d_model = weight.shape
    if d_model != lens.d_model:
        raise ValueError(f"unembedding width {d_model} != lens width {lens.d_model}")
    for token_id in selected_tokens:
        if not 0 <= token_id < vocab_size:
            raise ValueError(
                f"steer token ID {token_id} out of range for vocabulary size {vocab_size}"
            )

    unembedding_rows = weight.detach()[list(selected_tokens)].float()
    directions: dict[int, torch.Tensor] = {}
    for layer in selected_layers:
        jacobian = lens.jacobian(layer)
        if jacobian.shape != (lens.d_model, lens.d_model):
            raise ValueError(
                f"layer {layer} Jacobian shape {tuple(jacobian.shape)} != "
                f"({lens.d_model}, {lens.d_model})"
            )
        rows = unembedding_rows.to(device=jacobian.device, dtype=jacobian.dtype)
        transported = (rows @ jacobian).float()
        norms = torch.linalg.vector_norm(transported, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norms).all()):
            raise FloatingPointError(f"non-finite J-lens direction norm at layer {layer}")
        normalized = torch.where(
            norms > 0,
            transported / norms.clamp_min(_STEER_NORM_EPS),
            transported,
        )
        direction = normalized.sum(dim=0)
        if not bool(torch.isfinite(direction).all()):
            raise FloatingPointError(f"non-finite J-lens direction at layer {layer}")
        directions[layer] = direction.detach()
    return directions


def bos_skip_mask(
    input_ids: torch.Tensor,
    bos_token_id: int | None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Return ``[batch, sequence, 1]`` marking exact BOS positions to skip."""

    if bos_token_id is None:
        return None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("input_ids must be a rank-two tensor")
    mask = input_ids.eq(int(bos_token_id)).unsqueeze(-1)
    if not bool(mask.any()):
        return None
    return mask.to(device=device) if device is not None else mask


def apply_residual_post_steer(
    hidden: torch.Tensor,
    direction: torch.Tensor,
    strength: float,
    *,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Neuronpedia's additive, live-residual-norm steering operation."""

    if hidden.ndim < 2:
        raise ValueError("hidden must have a residual-width axis")
    if direction.ndim != 1 or direction.shape[0] != hidden.shape[-1]:
        raise ValueError(
            f"direction shape {tuple(direction.shape)} does not match hidden width "
            f"{hidden.shape[-1]}"
        )
    if not math.isfinite(float(strength)):
        raise ValueError("strength must be finite")
    if skip_mask is not None:
        expected_shape = (*hidden.shape[:-1], 1)
        if skip_mask.shape != expected_shape:
            raise ValueError(
                f"skip mask shape {tuple(skip_mask.shape)} != expected {expected_shape}"
            )

    delta = direction.to(device=hidden.device, dtype=hidden.dtype)
    residual_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True)
    injected = (float(strength) * residual_norm) * delta
    injected_norm = torch.linalg.vector_norm(injected, dim=-1, keepdim=True)
    max_norm = MAX_STEER_INJECTION_FRACTION * residual_norm
    clamp_factor = torch.where(
        injected_norm > max_norm,
        max_norm / injected_norm.clamp_min(_STEER_NORM_EPS),
        torch.ones_like(injected_norm),
    )
    steered = hidden + injected * clamp_factor
    if skip_mask is not None:
        steered = torch.where(skip_mask.to(device=hidden.device), hidden, steered)
    return steered


def _forward_input_ids(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> torch.Tensor | None:
    candidate = kwargs.get("input_ids")
    if isinstance(candidate, torch.Tensor):
        return candidate
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return None


class ResidualPostSteering:
    """Context-managed post-block steering for prefill and cached generation.

    The first top-level model forward after construction or ``reset_sequence`` is
    treated as prompt prefill and steered. Subsequent forwards are treated as
    generated-token forwards and are steered only when requested.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        directions: Mapping[int, torch.Tensor],
        strength: float,
        *,
        decoder_layers: Sequence[torch.nn.Module] | None = None,
        bos_token_id: int | None = None,
        steer_generated_tokens: bool = False,
    ) -> None:
        if not directions:
            raise ValueError("directions cannot be empty")
        if not math.isfinite(float(strength)):
            raise ValueError("strength must be finite")
        self.model = model
        self.directions = {
            int(layer): direction.detach() for layer, direction in directions.items()
        }
        self.strength = float(strength)
        self.decoder_layers = tuple(
            resolve_decoder_layers(model) if decoder_layers is None else decoder_layers
        )
        if not self.decoder_layers:
            raise ValueError("decoder_layers cannot be empty")
        invalid = sorted(
            layer for layer in self.directions if layer < 0 or layer >= len(self.decoder_layers)
        )
        if invalid:
            raise ValueError(
                f"steering layers {invalid} out of range for {len(self.decoder_layers)} blocks"
            )
        self.bos_token_id = bos_token_id
        self.steer_generated_tokens = bool(steer_generated_tokens)
        self._handles: list[Any] = []
        self._saw_prefill = False
        self._apply_current_forward = False
        self._current_skip_mask: torch.Tensor | None = None

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def reset_sequence(self) -> None:
        """Make the next model forward a steered prompt-prefill forward."""

        self._saw_prefill = False
        self._apply_current_forward = False
        self._current_skip_mask = None

    def _model_pre_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        input_ids = _forward_input_ids(args, kwargs)
        if self.bos_token_id is not None and input_ids is None:
            raise ValueError("BOS-safe steering requires input_ids on every model forward")
        is_prefill = not self._saw_prefill
        self._saw_prefill = True
        self._apply_current_forward = is_prefill or self.steer_generated_tokens
        self._current_skip_mask = (
            bos_skip_mask(input_ids, self.bos_token_id) if input_ids is not None else None
        )

    def _model_post_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        self._apply_current_forward = False
        self._current_skip_mask = None
        return output

    def _layer_hook(self, direction: torch.Tensor):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not self._apply_current_forward:
                return output
            hidden = _output_hidden(output)
            skip_mask = self._current_skip_mask
            if skip_mask is not None and skip_mask.shape != (*hidden.shape[:-1], 1):
                raise ValueError(
                    f"BOS mask shape {tuple(skip_mask.shape)} does not match hidden "
                    f"shape {tuple(hidden.shape)}"
                )
            steered = apply_residual_post_steer(
                hidden,
                direction,
                self.strength,
                skip_mask=skip_mask,
            )
            return _replace_output_hidden(output, steered)

        return hook

    def install(self) -> ResidualPostSteering:
        if self.installed:
            raise RuntimeError("steering hooks are already installed")
        self.reset_sequence()
        try:
            self._handles.append(
                self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
            )
            for layer, direction in self.directions.items():
                self._handles.append(
                    self.decoder_layers[layer].register_forward_hook(
                        self._layer_hook(direction)
                    )
                )
            self._handles.append(
                self.model.register_forward_hook(self._model_post_hook, always_call=True)
            )
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        self._apply_current_forward = False
        self._current_skip_mask = None

    def __enter__(self):
        return self.install()

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "MAX_STEER_INJECTION_FRACTION",
    "NEURONPEDIA_JLENS_STEERING_COMMIT",
    "ResidualPostSteering",
    "apply_residual_post_steer",
    "bos_skip_mask",
    "build_jlens_token_directions",
    "decoded_token_index",
    "resolve_steer_token_id",
]
