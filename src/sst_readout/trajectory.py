"""Autoregressive, pre-token Jacobian-lens trajectory collection.

Each row is anchored to one generated token.  The stored residual/J-space value is
the post-block state at the final token of the prefix *before* that generated token
is selected.  The token is then selected from the logits produced by the same
forward pass.  This makes the causal alignment explicit and mechanically auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifact import FrozenLensArtifact, sha256_file
from .collection import (
    PositionManifest,
    _model_input_device,
    _tensor_from_layer_output,
    _tokenization_digest,
    _tokenize,
    resolve_decoder_layers,
)
from .logit_lens import FixedBaseDecoder

SCHEMA_VERSION = 1
TENSOR_LAYOUT = "row,source_layer,d_model"
_SHA256_LENGTH = 64


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Stable digest for an exact, ordered token-id sequence."""

    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Digest tensor values with an explicit dtype/shape preamble."""

    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    ids = [int(token_id) for token_id in token_ids]
    try:
        return tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(ids)


def _json_safe_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = dict(value)
    try:
        json.dumps(result, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain only JSON-serializable values") from error
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class TrajectoryRow:
    """Metadata for one pre-generated-token boundary."""

    row_index: int
    prompt_id: str
    split: str
    generated_token_index: int
    boundary_kind: str
    prefix_length: int
    boundary_position: int
    boundary_token_id: int
    boundary_token_text: str
    sampled_token_id: int
    sampled_token_text: str
    sampled_logit: float
    sampled_logprob: float
    final_logits_sha256: str
    is_eos: bool
    prompt_tokenization_sha256: str
    prefix_token_ids_sha256: str
    prefix_text_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "prompt_id": self.prompt_id,
            "split": self.split,
            "generated_token_index": self.generated_token_index,
            "boundary_kind": self.boundary_kind,
            "prefix_length": self.prefix_length,
            "boundary_position": self.boundary_position,
            "boundary_token_id": self.boundary_token_id,
            "boundary_token_text": self.boundary_token_text,
            "sampled_token_id": self.sampled_token_id,
            "sampled_token_text": self.sampled_token_text,
            "sampled_logit": self.sampled_logit,
            "sampled_logprob": self.sampled_logprob,
            "final_logits_sha256": self.final_logits_sha256,
            "is_eos": self.is_eos,
            "prompt_tokenization_sha256": self.prompt_tokenization_sha256,
            "prefix_token_ids_sha256": self.prefix_token_ids_sha256,
            "prefix_text_sha256": self.prefix_text_sha256,
        }


@dataclass(frozen=True)
class PromptTrajectory:
    prompt_id: str
    split: str
    prompt: str
    input_token_ids: tuple[int, ...]
    prompt_tokenization_sha256: str
    generated_token_ids: tuple[int, ...]
    generated_text: str
    generated_tokenization_sha256: str
    eos_reached: bool
    stop_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "split": self.split,
            "prompt": self.prompt,
            "input_token_ids": list(self.input_token_ids),
            "prompt_tokenization_sha256": self.prompt_tokenization_sha256,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_text": self.generated_text,
            "generated_tokenization_sha256": self.generated_tokenization_sha256,
            "n_generated_tokens": len(self.generated_token_ids),
            "eos_reached": self.eos_reached,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class JlensTrajectory:
    """Full J-space trajectory and its immutable row/identity metadata."""

    run_id: str
    created_at: str
    model_identity: Mapping[str, Any]
    lens_identity: Mapping[str, Any]
    decoder_identity: Mapping[str, Any]
    decoding: Mapping[str, Any]
    position_manifest_sha256: str
    source_layers: tuple[int, ...]
    final_block_index: int
    rows: tuple[TrajectoryRow, ...]
    prompts: tuple[PromptTrajectory, ...]
    jspace: torch.Tensor
    final_hidden: torch.Tensor
    top_token_ids: torch.Tensor | None = None
    top_scores: torch.Tensor | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_readouts(self) -> int:
        return self.n_rows * len(self.source_layers)

    @property
    def n_final_references(self) -> int:
        return self.n_rows

    @property
    def n_layer_position_cells(self) -> int:
        return self.n_readouts + self.n_final_references

    @property
    def d_model(self) -> int:
        return int(self.jspace.shape[2])

    @property
    def top_k(self) -> int:
        return 0 if self.top_token_ids is None else int(self.top_token_ids.shape[2])

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be nonempty")
        for label, identity in (
            ("model_identity", self.model_identity),
            ("lens_identity", self.lens_identity),
            ("decoder_identity", self.decoder_identity),
            ("decoding", self.decoding),
        ):
            _json_safe_mapping(identity, label=label)
        if not _is_sha256(self.position_manifest_sha256):
            raise ValueError("position_manifest_sha256 must be a lowercase SHA-256")
        if (
            not self.source_layers
            or tuple(sorted(set(self.source_layers))) != self.source_layers
        ):
            raise ValueError("source_layers must be sorted, unique, and nonempty")
        if self.final_block_index < 0 or self.final_block_index in self.source_layers:
            raise ValueError("final block must be nonnegative and separate from fitted layers")
        if self.jspace.ndim != 3:
            raise ValueError(f"jspace must use layout [{TENSOR_LAYOUT}]")
        expected_shape = (self.n_rows, len(self.source_layers))
        if tuple(self.jspace.shape[:2]) != expected_shape or self.d_model <= 0:
            raise ValueError(
                f"jspace shape {tuple(self.jspace.shape)} does not match rows/layers "
                f"{expected_shape}"
            )
        if not self.jspace.is_floating_point() or not torch.isfinite(self.jspace).all():
            raise ValueError("jspace must contain finite floating-point values")
        if tuple(self.final_hidden.shape) != (self.n_rows, self.d_model):
            raise ValueError("final_hidden must have shape [rows, d_model]")
        if (
            not self.final_hidden.is_floating_point()
            or not torch.isfinite(self.final_hidden).all()
        ):
            raise ValueError("final_hidden must contain finite floating-point values")
        if (self.top_token_ids is None) != (self.top_scores is None):
            raise ValueError("top_token_ids and top_scores must be supplied together")
        if self.top_token_ids is not None and self.top_scores is not None:
            if (
                self.top_token_ids.ndim != 3
                or tuple(self.top_token_ids.shape[:2]) != expected_shape
            ):
                raise ValueError("top_token_ids must have shape [rows, layers, top_k]")
            if tuple(self.top_scores.shape) != tuple(self.top_token_ids.shape):
                raise ValueError("top_scores must have the same shape as top_token_ids")
            if self.top_k <= 0 or self.top_token_ids.is_floating_point():
                raise ValueError("top_token_ids must be an integer tensor with nonzero top_k")
            if bool((self.top_token_ids < 0).any()):
                raise ValueError("top_token_ids cannot be negative")
            if (
                not self.top_scores.is_floating_point()
                or not torch.isfinite(self.top_scores).all()
            ):
                raise ValueError("top_scores must contain finite floating-point values")
            vocab_size = self.decoder_identity.get("vocab_size")
            if vocab_size is not None and bool((self.top_token_ids >= int(vocab_size)).any()):
                raise ValueError("top_token_ids exceed the recorded decoder vocabulary")

        if sum(len(prompt.generated_token_ids) for prompt in self.prompts) != self.n_rows:
            raise ValueError("expected rows must equal the sum of generated tokens")
        if tuple(row.row_index for row in self.rows) != tuple(range(self.n_rows)):
            raise ValueError("trajectory row_index values must be contiguous from zero")
        if not self.prompts or len({prompt.prompt_id for prompt in self.prompts}) != len(
            self.prompts
        ):
            raise ValueError("prompt trajectories must be nonempty with unique prompt_ids")

        row_cursor = 0
        eos_token_ids = {int(value) for value in self.decoding.get("eos_token_ids", [])}
        for prompt in self.prompts:
            if not prompt.input_token_ids:
                raise ValueError(f"prompt {prompt.prompt_id} has no input tokens")
            if token_ids_sha256(prompt.input_token_ids) != prompt.prompt_tokenization_sha256:
                raise ValueError(f"prompt tokenization hash mismatch for {prompt.prompt_id}")
            if token_ids_sha256(prompt.generated_token_ids) != (
                prompt.generated_tokenization_sha256
            ):
                raise ValueError(f"generated tokenization hash mismatch for {prompt.prompt_id}")
            prompt_rows = self.rows[row_cursor : row_cursor + len(prompt.generated_token_ids)]
            for index, (row, sampled_id) in enumerate(
                zip(prompt_rows, prompt.generated_token_ids, strict=True)
            ):
                prefix = prompt.input_token_ids + prompt.generated_token_ids[:index]
                if row.prompt_id != prompt.prompt_id or row.split != prompt.split:
                    raise ValueError("trajectory rows are not grouped under their prompt")
                if row.generated_token_index != index:
                    raise ValueError("generated_token_index must be contiguous per prompt")
                expected_boundary_kind = "pre_answer" if index == 0 else "post_generated_token"
                if row.boundary_kind != expected_boundary_kind:
                    raise ValueError("row boundary_kind disagrees with generated token index")
                if row.sampled_token_id != sampled_id:
                    raise ValueError("row sampled token disagrees with prompt summary")
                if row.prefix_length != len(prefix):
                    raise ValueError("row prefix_length does not match its causal prefix")
                if row.boundary_position != len(prefix) - 1:
                    raise ValueError("boundary_position must be the final pre-token position")
                if row.boundary_token_id != prefix[-1]:
                    raise ValueError("boundary_token_id must be the final prefix token")
                if row.prompt_tokenization_sha256 != prompt.prompt_tokenization_sha256:
                    raise ValueError("row prompt tokenization hash mismatch")
                if row.prefix_token_ids_sha256 != token_ids_sha256(prefix):
                    raise ValueError("row prefix hash does not match its causal prefix")
                if not _is_sha256(row.final_logits_sha256):
                    raise ValueError("row final_logits_sha256 must be a lowercase SHA-256")
                if not math.isfinite(row.sampled_logit) or not math.isfinite(
                    row.sampled_logprob
                ):
                    raise ValueError("row sampled logit/logprob must be finite")
                if row.is_eos != (sampled_id in eos_token_ids):
                    raise ValueError("row EOS flag disagrees with configured EOS token ids")
            eos_positions = [
                i
                for i, token_id in enumerate(prompt.generated_token_ids)
                if token_id in eos_token_ids
            ]
            if eos_positions and eos_positions != [len(prompt.generated_token_ids) - 1]:
                raise ValueError("generation must stop immediately after its first EOS token")
            if prompt.eos_reached != bool(eos_positions):
                raise ValueError("prompt EOS summary disagrees with generated tokens")
            expected_reason = "eos" if prompt.eos_reached else "max_new_tokens"
            if prompt.stop_reason != expected_reason:
                raise ValueError("unsupported or inconsistent trajectory stop_reason")
            row_cursor += len(prompt.generated_token_ids)
        if row_cursor != self.n_rows:
            raise ValueError("trajectory contains unassigned rows")

    def metadata_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "tensor_layout": TENSOR_LAYOUT,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "model_identity": dict(self.model_identity),
            "lens_identity": dict(self.lens_identity),
            "decoder_identity": dict(self.decoder_identity),
            "decoding": dict(self.decoding),
            "position_manifest_sha256": self.position_manifest_sha256,
            "source_layers": list(self.source_layers),
            "final_block_index": self.final_block_index,
            "n_rows": self.n_rows,
            "n_layers": len(self.source_layers),
            "fitted_jlens_cells": self.n_readouts,
            "final_reference_cells": self.n_final_references,
            "total_layer_position_cells": self.n_layer_position_cells,
            "d_model": self.d_model,
            "storage_dtype": str(self.jspace.dtype).removeprefix("torch."),
            "top_k": self.top_k,
            "rows": [row.as_dict() for row in self.rows],
            "prompts": [prompt.as_dict() for prompt in self.prompts],
        }


def _select_greedy(logits: torch.Tensor) -> tuple[int, float, float, str]:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("next-token logits must be a one-dimensional floating tensor")
    scores = logits.detach().float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("next-token logits contain nonfinite values")
    token_id = int(torch.argmax(scores).item())
    logit = float(scores[token_id].item())
    logprob = float((scores[token_id] - torch.logsumexp(scores, dim=0)).item())
    return token_id, logit, logprob, tensor_sha256(scores)


def _transport_hidden(
    hidden_by_layer: Mapping[int, Sequence[torch.Tensor]],
    lens: FrozenLensArtifact,
    *,
    source_layers: tuple[int, ...],
    compute_device: str | torch.device,
    compute_dtype: torch.dtype,
    storage_dtype: torch.dtype,
    row_batch_size: int,
) -> torch.Tensor:
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    device = torch.device(compute_device)
    transported_layers: list[torch.Tensor] = []
    for layer in source_layers:
        residuals = torch.stack(tuple(hidden_by_layer[layer]), dim=0)
        matrix = lens.jacobian(layer).to(device=device, dtype=compute_dtype)
        chunks: list[torch.Tensor] = []
        for start in range(0, residuals.shape[0], row_batch_size):
            hidden = residuals[start : start + row_batch_size].to(
                device=device, dtype=compute_dtype
            )
            chunks.append((hidden @ matrix.T).to(device="cpu", dtype=storage_dtype))
        values = torch.cat(chunks, dim=0)
        if not torch.isfinite(values).all():
            raise ValueError(f"transported J-space at layer {layer} overflows {storage_dtype}")
        transported_layers.append(values)
        del matrix
    return torch.stack(transported_layers, dim=1)


@torch.inference_mode()
def _decode_topk(
    jspace: torch.Tensor,
    decoder: FixedBaseDecoder,
    *,
    top_k: int,
    row_batch_size: int,
    score_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ids_by_layer: list[torch.Tensor] = []
    scores_by_layer: list[torch.Tensor] = []
    for layer_index in range(jspace.shape[1]):
        layer_ids: list[torch.Tensor] = []
        layer_scores: list[torch.Tensor] = []
        for start in range(0, jspace.shape[0], row_batch_size):
            logits = decoder(jspace[start : start + row_batch_size, layer_index])
            if top_k > logits.shape[-1]:
                raise ValueError(f"top_k {top_k} exceeds decoder vocab {logits.shape[-1]}")
            scores, token_ids = torch.topk(logits, k=top_k, dim=-1)
            layer_ids.append(token_ids.to(device="cpu", dtype=torch.int32))
            layer_scores.append(scores.to(device="cpu", dtype=score_dtype))
        ids_by_layer.append(torch.cat(layer_ids, dim=0))
        scores_by_layer.append(torch.cat(layer_scores, dim=0))
    return torch.stack(ids_by_layer, dim=1), torch.stack(scores_by_layer, dim=1)


def collect_jlens_trajectories(
    model: Any,
    tokenizer: Any,
    manifest: PositionManifest,
    lens: FrozenLensArtifact,
    *,
    run_id: str,
    model_identity: Mapping[str, Any],
    decoder_identity: Mapping[str, Any],
    max_new_tokens: int,
    eos_token_ids: Sequence[int] = (),
    source_layers: Sequence[int] | None = None,
    decoder_layers: Sequence[torch.nn.Module] | None = None,
    max_sequence_length: int | None = None,
    compute_device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    storage_dtype: torch.dtype = torch.bfloat16,
    row_batch_size: int = 32,
    top_k: int = 0,
    decoder: FixedBaseDecoder | None = None,
) -> JlensTrajectory:
    """Generate once per prompt and collect every causal pre-token boundary.

    Decoding is deliberately greedy.  No call to ``model.generate`` is made: each
    token comes from the exact logits whose hooked prefix state is saved on that
    row, then that token is appended before the next cached forward pass.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    if top_k and decoder is None:
        raise ValueError("top-k decoding requires a fixed decoder")
    if decoder_layers is None:
        decoder_layers = resolve_decoder_layers(model)
    layers = (
        lens.source_layers
        if source_layers is None
        else tuple(sorted({int(layer) for layer in source_layers}))
    )
    if not layers:
        raise ValueError("source_layers cannot be empty")
    missing = sorted(set(layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"lens is missing requested source layers {missing}")
    if layers[0] < 0 or layers[-1] >= len(decoder_layers):
        raise ValueError(
            f"source layers {layers} out of range for {len(decoder_layers)} blocks"
        )
    if max_sequence_length is not None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        too_long = [
            prompt.prompt_id
            for prompt in manifest.prompts
            if len(prompt.input_ids) + max_new_tokens > max_sequence_length
        ]
        if too_long:
            raise ValueError(
                "prompt plus max_new_tokens exceeds model context for " + ", ".join(too_long)
            )

    eos_ids = tuple(sorted({int(token_id) for token_id in eos_token_ids}))
    if eos_ids and eos_ids[0] < 0:
        raise ValueError("EOS token ids cannot be negative")
    model_metadata = _json_safe_mapping(model_identity, label="model_identity")
    decoder_metadata = _json_safe_mapping(decoder_identity, label="decoder_identity")
    lens_metadata = lens.manifest()

    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            values = _tensor_from_layer_output(output)
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("trajectory hooks expect one [batch, sequence, hidden] output")
            captured[layer] = values[0, -1].detach()

        return hook

    final_block_index = len(decoder_layers) - 1
    if final_block_index in layers:
        raise ValueError(
            "the final block is an identity reference, not a fitted J-Lens source layer"
        )
    record_layers = tuple(sorted(set(layers) | {final_block_index}))
    for layer in record_layers:
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))

    input_device = _model_input_device(model)
    hidden_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    final_hidden_rows: list[torch.Tensor] = []
    rows: list[TrajectoryRow] = []
    prompt_results: list[PromptTrajectory] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.inference_mode():
            for prompt in manifest.prompts:
                replay_ids = _tokenize(
                    tokenizer,
                    prompt.prompt,
                    max_length=manifest.max_length,
                    add_special_tokens=manifest.add_special_tokens,
                )
                if replay_ids != prompt.input_ids:
                    raise ValueError(
                        f"tokenization drift for {prompt.prompt_id}: expected "
                        f"{prompt.tokenization_sha256}, got {_tokenization_digest(replay_ids)}"
                    )

                prefix = list(prompt.input_ids)
                generated: list[int] = []
                current_input = torch.tensor([prefix], dtype=torch.long, device=input_device)
                past_key_values = None
                eos_reached = False
                for generated_index in range(max_new_tokens):
                    captured.clear()
                    attention_mask = torch.ones(
                        (1, len(prefix)), dtype=torch.long, device=input_device
                    )
                    kwargs: dict[str, Any] = {
                        "input_ids": current_input,
                        "attention_mask": attention_mask,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    if past_key_values is not None:
                        kwargs["past_key_values"] = past_key_values
                    output = model(**kwargs)
                    missing_captures = sorted(set(record_layers) - set(captured))
                    if missing_captures:
                        raise RuntimeError(
                            f"hooks did not capture decoder layers {missing_captures}"
                        )
                    logits = getattr(output, "logits", None)
                    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                        raise TypeError(
                            "causal LM output must provide [batch, sequence, vocab] logits"
                        )
                    (
                        sampled_id,
                        sampled_logit,
                        sampled_logprob,
                        logits_digest,
                    ) = _select_greedy(logits[0, -1])
                    prefix_text = _decode(tokenizer, prefix)
                    sampled_text = _decode(tokenizer, [sampled_id])
                    row = TrajectoryRow(
                        row_index=len(rows),
                        prompt_id=prompt.prompt_id,
                        split=prompt.split,
                        generated_token_index=generated_index,
                        boundary_kind=(
                            "pre_answer" if generated_index == 0 else "post_generated_token"
                        ),
                        prefix_length=len(prefix),
                        boundary_position=len(prefix) - 1,
                        boundary_token_id=prefix[-1],
                        boundary_token_text=_decode(tokenizer, [prefix[-1]]),
                        sampled_token_id=sampled_id,
                        sampled_token_text=sampled_text,
                        sampled_logit=sampled_logit,
                        sampled_logprob=sampled_logprob,
                        final_logits_sha256=logits_digest,
                        is_eos=sampled_id in eos_ids,
                        prompt_tokenization_sha256=prompt.tokenization_sha256,
                        prefix_token_ids_sha256=token_ids_sha256(prefix),
                        prefix_text_sha256=text_sha256(prefix_text),
                    )
                    rows.append(row)
                    for layer in layers:
                        hidden_rows[layer].append(captured[layer].to(device="cpu"))
                    final_hidden_rows.append(captured[final_block_index].to(device="cpu"))
                    generated.append(sampled_id)
                    prefix.append(sampled_id)
                    eos_reached = sampled_id in eos_ids
                    if eos_reached:
                        break
                    if generated_index + 1 < max_new_tokens:
                        past_key_values = getattr(output, "past_key_values", None)
                        if past_key_values is None:
                            raise RuntimeError(
                                "model did not return past_key_values for autoregressive replay"
                            )
                        current_input = torch.tensor(
                            [[sampled_id]], dtype=torch.long, device=input_device
                        )
                prompt_results.append(
                    PromptTrajectory(
                        prompt_id=prompt.prompt_id,
                        split=prompt.split,
                        prompt=prompt.prompt,
                        input_token_ids=prompt.input_ids,
                        prompt_tokenization_sha256=prompt.tokenization_sha256,
                        generated_token_ids=tuple(generated),
                        generated_text=_decode(tokenizer, generated),
                        generated_tokenization_sha256=token_ids_sha256(generated),
                        eos_reached=eos_reached,
                        stop_reason="eos" if eos_reached else "max_new_tokens",
                    )
                )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    jspace = _transport_hidden(
        hidden_rows,
        lens,
        source_layers=layers,
        compute_device=compute_device,
        compute_dtype=compute_dtype,
        storage_dtype=storage_dtype,
        row_batch_size=row_batch_size,
    )
    final_hidden = torch.stack(final_hidden_rows, dim=0).to(device="cpu", dtype=storage_dtype)
    if not torch.isfinite(final_hidden).all():
        raise ValueError(f"final hidden reference overflows {storage_dtype}")
    top_token_ids: torch.Tensor | None = None
    top_scores: torch.Tensor | None = None
    if top_k:
        assert decoder is not None
        top_token_ids, top_scores = _decode_topk(
            jspace,
            decoder,
            top_k=top_k,
            row_batch_size=row_batch_size,
            score_dtype=storage_dtype,
        )
    result = JlensTrajectory(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_identity=model_metadata,
        lens_identity=lens_metadata,
        decoder_identity=decoder_metadata,
        decoding={
            "strategy": "greedy",
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": list(eos_ids),
            "use_cache": True,
            "row_semantics": (
                "post-block residual at prefix[-1], transported before sampled token "
                "is appended; sampled token comes from the same forward-pass logits"
            ),
        },
        position_manifest_sha256=manifest.manifest_sha256,
        source_layers=layers,
        final_block_index=final_block_index,
        rows=tuple(rows),
        prompts=tuple(prompt_results),
        jspace=jspace,
        final_hidden=final_hidden,
        top_token_ids=top_token_ids,
        top_scores=top_scores,
    )
    result.validate()
    return result


def _trajectory_id(metadata: Mapping[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        torch.save(dict(payload), temp_path)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        temp_path.write_text(
            json.dumps(dict(payload), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)


def save_jlens_trajectory(trajectory: JlensTrajectory, path: str | Path) -> tuple[Path, Path]:
    """Save compact tensors plus a hash-bound, human-readable sidecar manifest."""

    trajectory.validate()
    output = Path(path)
    metadata = trajectory.metadata_dict()
    trajectory_id = _trajectory_id(metadata)
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": trajectory_id,
            "source_layers": list(trajectory.source_layers),
            "jspace": trajectory.jspace,
            "final_hidden": trajectory.final_hidden,
            "top_token_ids": trajectory.top_token_ids,
            "top_scores": trajectory.top_scores,
        },
        output,
    )
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    sidecar_payload = {
        **metadata,
        "trajectory_id": trajectory_id,
        "tensor_path": output.name,
        "tensor_sha256": sha256_file(output),
    }
    manifest_sha256 = _trajectory_id(sidecar_payload)
    _atomic_json_save({**sidecar_payload, "manifest_sha256": manifest_sha256}, sidecar)
    return output, sidecar


def load_jlens_trajectory(path: str | Path) -> JlensTrajectory:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".manifest.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported J-lens trajectory manifest schema")
    expected_manifest_sha = metadata.pop("manifest_sha256", None)
    if expected_manifest_sha != _trajectory_id(metadata):
        raise ValueError("J-lens trajectory manifest digest mismatch")
    if metadata.get("tensor_path") != source.name:
        raise ValueError("J-lens trajectory tensor path does not match its sidecar")
    if sha256_file(source) != metadata.get("tensor_sha256"):
        raise ValueError("J-lens trajectory tensor digest mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported J-lens trajectory tensor schema")

    identity_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "trajectory_id",
            "tensor_path",
            "tensor_sha256",
        }
    }
    expected_trajectory_id = _trajectory_id(identity_metadata)
    if (
        metadata.get("trajectory_id") != expected_trajectory_id
        or payload.get("trajectory_id") != expected_trajectory_id
    ):
        raise ValueError("J-lens trajectory identity mismatch")
    payload_layers = tuple(int(layer) for layer in payload.get("source_layers", []))
    metadata_layers = tuple(int(layer) for layer in metadata["source_layers"])
    if payload_layers != metadata_layers:
        raise ValueError("J-lens trajectory layer inventory mismatch")

    rows = tuple(TrajectoryRow(**row) for row in metadata["rows"])
    prompts = tuple(
        PromptTrajectory(
            prompt_id=item["prompt_id"],
            split=item["split"],
            prompt=item["prompt"],
            input_token_ids=tuple(int(value) for value in item["input_token_ids"]),
            prompt_tokenization_sha256=item["prompt_tokenization_sha256"],
            generated_token_ids=tuple(int(value) for value in item["generated_token_ids"]),
            generated_text=item["generated_text"],
            generated_tokenization_sha256=item["generated_tokenization_sha256"],
            eos_reached=bool(item["eos_reached"]),
            stop_reason=item["stop_reason"],
        )
        for item in metadata["prompts"]
    )
    result = JlensTrajectory(
        run_id=metadata["run_id"],
        created_at=metadata["created_at"],
        model_identity=metadata["model_identity"],
        lens_identity=metadata["lens_identity"],
        decoder_identity=metadata["decoder_identity"],
        decoding=metadata["decoding"],
        position_manifest_sha256=metadata["position_manifest_sha256"],
        source_layers=metadata_layers,
        final_block_index=int(metadata["final_block_index"]),
        rows=rows,
        prompts=prompts,
        jspace=payload["jspace"],
        final_hidden=payload["final_hidden"],
        top_token_ids=payload.get("top_token_ids"),
        top_scores=payload.get("top_scores"),
    )
    result.validate()
    if result.n_rows != metadata["n_rows"]:
        raise ValueError("J-lens trajectory row count mismatch")
    if len(result.source_layers) != metadata["n_layers"]:
        raise ValueError("J-lens trajectory layer count mismatch")
    if result.d_model != metadata["d_model"]:
        raise ValueError("J-lens trajectory residual-width mismatch")
    if result.top_k != metadata["top_k"]:
        raise ValueError("J-lens trajectory top-k count mismatch")
    if str(result.jspace.dtype).removeprefix("torch.") != metadata["storage_dtype"]:
        raise ValueError("J-lens trajectory storage dtype mismatch")
    if result.n_readouts != metadata["fitted_jlens_cells"]:
        raise ValueError("J-lens trajectory fitted-cell count mismatch")
    if result.n_final_references != metadata["final_reference_cells"]:
        raise ValueError("J-lens trajectory final-reference count mismatch")
    if result.n_layer_position_cells != metadata["total_layer_position_cells"]:
        raise ValueError("J-lens trajectory total-cell count mismatch")
    return result
