"""Exact-position hidden-state collection shared by every checkpoint arm."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .artifact import FrozenLensArtifact


@dataclass(frozen=True)
class PromptSpec:
    """One already-rendered prompt and the token positions to read."""

    prompt_id: str
    split: str
    prompt: str
    positions: tuple[int, ...] = (-1,)
    anchor_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FrozenPrompt:
    prompt_id: str
    split: str
    prompt: str
    input_ids: tuple[int, ...]
    positions: tuple[int, ...]
    anchor_ids: tuple[str, ...]
    selected_token_ids: tuple[int, ...]
    tokenization_sha256: str


@dataclass(frozen=True)
class PositionManifest:
    """Immutable tokenization and readout positions for all model arms."""

    prompts: tuple[FrozenPrompt, ...]
    tokenizer_id: str
    tokenizer_revision: str
    max_length: int
    add_special_tokens: bool
    manifest_sha256: str

    @property
    def n_rows(self) -> int:
        return sum(len(prompt.positions) for prompt in self.prompts)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({prompt.split for prompt in self.prompts}))

    def prompt_ids(self, split: str | None = None) -> tuple[str, ...]:
        return tuple(
            prompt.prompt_id
            for prompt in self.prompts
            if split is None or prompt.split == split
        )

    def as_dict(self, *, include_prompts: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "max_length": self.max_length,
            "add_special_tokens": self.add_special_tokens,
            "manifest_sha256": self.manifest_sha256,
            "n_prompts": len(self.prompts),
            "n_rows": self.n_rows,
            "splits": list(self.splits),
        }
        if include_prompts:
            payload["prompts"] = [
                {
                    "prompt_id": item.prompt_id,
                    "split": item.split,
                    "prompt": item.prompt,
                    "input_ids": list(item.input_ids),
                    "positions": list(item.positions),
                    "anchor_ids": list(item.anchor_ids),
                    "selected_token_ids": list(item.selected_token_ids),
                    "tokenization_sha256": item.tokenization_sha256,
                }
                for item in self.prompts
            ]
        return payload


@dataclass(frozen=True, order=True)
class RowIdentity:
    prompt_id: str
    split: str
    position: int
    anchor_id: str
    token_id: int
    tokenization_sha256: str


@dataclass(frozen=True)
class CollectedReadouts:
    """One model's residuals in raw and (optionally) transported J-space."""

    model_id: str
    model_revision: str
    manifest_sha256: str
    rows: tuple[RowIdentity, ...]
    source_layers: tuple[int, ...]
    hidden_by_layer: Mapping[int, torch.Tensor]
    final_hidden: torch.Tensor
    jspace_by_layer: Mapping[int, torch.Tensor] | None = None
    lens_provenance_id: str | None = None
    lens_artifact_sha256: str | None = None

    def validate(self) -> None:
        n_rows = len(self.rows)
        if self.final_hidden.ndim != 2 or self.final_hidden.shape[0] != n_rows:
            raise ValueError("final_hidden must have shape [n_rows, d_model]")
        d_model = self.final_hidden.shape[1]
        if tuple(sorted(self.hidden_by_layer)) != self.source_layers:
            raise ValueError("hidden_by_layer keys must equal source_layers")
        for layer, values in self.hidden_by_layer.items():
            if values.ndim != 2 or tuple(values.shape) != (n_rows, d_model):
                raise ValueError(
                    f"hidden layer {layer} shape {tuple(values.shape)} != ({n_rows}, {d_model})"
                )
            if not torch.isfinite(values).all():
                raise ValueError(f"hidden layer {layer} contains nonfinite values")
        if self.jspace_by_layer is not None:
            if tuple(sorted(self.jspace_by_layer)) != self.source_layers:
                raise ValueError("jspace_by_layer keys must equal source_layers")
            for layer, values in self.jspace_by_layer.items():
                if values.ndim != 2 or tuple(values.shape) != (n_rows, d_model):
                    raise ValueError(
                        f"J-space layer {layer} shape {tuple(values.shape)} != "
                        f"({n_rows}, {d_model})"
                    )
                if not torch.isfinite(values).all():
                    raise ValueError(f"J-space layer {layer} contains nonfinite values")

    @property
    def d_model(self) -> int:
        return int(self.final_hidden.shape[1])

    def row_indices(self, split: str) -> tuple[int, ...]:
        indices = tuple(i for i, row in enumerate(self.rows) if row.split == split)
        if not indices:
            raise ValueError(f"no rows found for split {split!r}")
        return indices

    def subset(self, split: str) -> CollectedReadouts:
        indices = self.row_indices(split)
        index = torch.tensor(indices, dtype=torch.long)
        result = replace(
            self,
            rows=tuple(self.rows[i] for i in indices),
            hidden_by_layer={
                layer: values.index_select(0, index)
                for layer, values in self.hidden_by_layer.items()
            },
            final_hidden=self.final_hidden.index_select(0, index),
            jspace_by_layer=None
            if self.jspace_by_layer is None
            else {
                layer: values.index_select(0, index)
                for layer, values in self.jspace_by_layer.items()
            },
        )
        result.validate()
        return result


def _plain_input_ids(encoded: Any) -> tuple[int, ...]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one tokenized prompt, got a batch")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError("tokenizer must return a one-dimensional integer input_ids list")
    if not encoded:
        raise ValueError("prompt tokenized to an empty sequence")
    return tuple(encoded)


def _tokenize(
    tokenizer: Any,
    prompt: str,
    *,
    max_length: int,
    add_special_tokens: bool,
) -> tuple[int, ...]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_length,
    )
    return _plain_input_ids(encoded)


def _normalize_positions(positions: Sequence[int], length: int) -> tuple[int, ...]:
    if not positions:
        raise ValueError("each prompt needs at least one readout position")
    normalized: list[int] = []
    for position in positions:
        index = position if position >= 0 else length + position
        if not 0 <= index < length:
            raise ValueError(f"position {position} is out of range for {length} tokens")
        if index in normalized:
            raise ValueError(f"position {position} duplicates normalized index {index}")
        normalized.append(index)
    return tuple(normalized)


def _tokenization_digest(input_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(input_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_position_manifest(
    tokenizer: Any,
    specs: Iterable[PromptSpec],
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
    max_length: int = 512,
    add_special_tokens: bool = True,
) -> PositionManifest:
    """Tokenize once; all later model arms must replay this exact manifest."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    frozen: list[FrozenPrompt] = []
    seen_ids: set[str] = set()
    for spec in specs:
        if not spec.prompt_id or spec.prompt_id in seen_ids:
            raise ValueError(f"prompt_id must be unique and nonempty: {spec.prompt_id!r}")
        if not spec.split:
            raise ValueError("split must be nonempty")
        seen_ids.add(spec.prompt_id)
        input_ids = _tokenize(
            tokenizer,
            spec.prompt,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        positions = _normalize_positions(spec.positions, len(input_ids))
        anchor_ids = (
            tuple(f"anchor-{index}" for index in range(len(positions)))
            if spec.anchor_ids is None
            else spec.anchor_ids
        )
        if len(anchor_ids) != len(positions) or any(not anchor for anchor in anchor_ids):
            raise ValueError("anchor_ids must be nonempty and align one-to-one with positions")
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("anchor_ids must be unique within a prompt")
        digest = _tokenization_digest(input_ids)
        frozen.append(
            FrozenPrompt(
                prompt_id=spec.prompt_id,
                split=spec.split,
                prompt=spec.prompt,
                input_ids=input_ids,
                positions=positions,
                anchor_ids=anchor_ids,
                selected_token_ids=tuple(input_ids[index] for index in positions),
                tokenization_sha256=digest,
            )
        )
    if not frozen:
        raise ValueError("position manifest cannot be empty")
    canonical = {
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "max_length": max_length,
        "add_special_tokens": add_special_tokens,
        "prompts": [
            {
                "prompt_id": item.prompt_id,
                "split": item.split,
                "prompt": item.prompt,
                "input_ids": item.input_ids,
                "positions": item.positions,
                "anchor_ids": item.anchor_ids,
            }
            for item in frozen
        ],
    }
    manifest_digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return PositionManifest(
        prompts=tuple(frozen),
        tokenizer_id=tokenizer_id,
        tokenizer_revision=tokenizer_revision,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        manifest_sha256=manifest_digest,
    )


def _resolve_path(root: Any, dotted_path: str) -> Any:
    value = root
    for component in dotted_path.split("."):
        if not hasattr(value, component):
            raise AttributeError(component)
        value = getattr(value, component)
    return value


def resolve_decoder_layers(model: Any) -> Sequence[torch.nn.Module]:
    """Locate common Hugging Face/Peft decoder-layer containers."""

    candidates = (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
    )
    for path in candidates:
        try:
            layers = _resolve_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (torch.nn.ModuleList, list, tuple)) and layers:
            return layers
    raise ValueError("could not locate decoder layers; pass decoder_layers explicitly")


def _tensor_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        value = output.last_hidden_state
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"unsupported decoder-layer output type {type(output)!r}")


def _model_input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def collect_hf_hidden_states(
    model: Any,
    tokenizer: Any,
    manifest: PositionManifest,
    *,
    model_id: str,
    model_revision: str,
    source_layers: Sequence[int],
    decoder_layers: Sequence[torch.nn.Module] | None = None,
    storage_dtype: torch.dtype = torch.float32,
) -> CollectedReadouts:
    """Collect post-block residuals at exactly the manifest's token positions.

    Hooks deliberately mirror the reference ``jlens.ActivationRecorder``
    convention. In particular, ``hidden_states[layer + 1]`` from a generic HF
    output is not used because final-norm placement differs across model classes.
    """

    if decoder_layers is None:
        decoder_layers = resolve_decoder_layers(model)
    layers = tuple(sorted({int(layer) for layer in source_layers}))
    if not layers:
        raise ValueError("source_layers cannot be empty")
    if layers[0] < 0 or layers[-1] >= len(decoder_layers):
        raise ValueError(
            f"source layers {layers} out of range for {len(decoder_layers)} blocks"
        )
    record_layers = tuple(sorted(set(layers) | {len(decoder_layers) - 1}))
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer] = _tensor_from_layer_output(output).detach()

        return hook

    for layer in record_layers:
        handles.append(decoder_layers[layer].register_forward_hook(make_hook(layer)))

    row_ids: list[RowIdentity] = []
    hidden_rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    final_rows: list[torch.Tensor] = []
    input_device = _model_input_device(model)
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
                        f"{prompt.tokenization_sha256}, got "
                        f"{_tokenization_digest(replay_ids)}"
                    )
                captured.clear()
                input_ids = torch.tensor(
                    [prompt.input_ids], dtype=torch.long, device=input_device
                )
                attention_mask = torch.ones_like(input_ids)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                missing = sorted(set(record_layers) - set(captured))
                if missing:
                    raise RuntimeError(f"hooks did not capture decoder layers {missing}")
                for position, anchor_id, token_id in zip(
                    prompt.positions,
                    prompt.anchor_ids,
                    prompt.selected_token_ids,
                    strict=True,
                ):
                    row_ids.append(
                        RowIdentity(
                            prompt_id=prompt.prompt_id,
                            split=prompt.split,
                            position=position,
                            anchor_id=anchor_id,
                            token_id=token_id,
                            tokenization_sha256=prompt.tokenization_sha256,
                        )
                    )
                    for layer in layers:
                        hidden_rows[layer].append(
                            captured[layer][0, position].to(device="cpu", dtype=storage_dtype)
                        )
                    final_rows.append(
                        captured[len(decoder_layers) - 1][0, position].to(
                            device="cpu", dtype=storage_dtype
                        )
                    )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    result = CollectedReadouts(
        model_id=model_id,
        model_revision=model_revision,
        manifest_sha256=manifest.manifest_sha256,
        rows=tuple(row_ids),
        source_layers=layers,
        hidden_by_layer={
            layer: torch.stack(values, dim=0) for layer, values in hidden_rows.items()
        },
        final_hidden=torch.stack(final_rows, dim=0),
    )
    result.validate()
    return result


def apply_frozen_lens(
    readouts: CollectedReadouts,
    lens: FrozenLensArtifact,
    *,
    compute_device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    storage_dtype: torch.dtype = torch.float32,
    row_batch_size: int = 64,
) -> CollectedReadouts:
    """Transport collected residuals without changing the source table."""

    readouts.validate()
    if readouts.d_model != lens.d_model:
        raise ValueError(f"readout width {readouts.d_model} != lens width {lens.d_model}")
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    missing = sorted(set(readouts.source_layers) - set(lens.source_layers))
    if missing:
        raise ValueError(f"lens is missing collected layers {missing}")
    device = torch.device(compute_device)
    transported: dict[int, torch.Tensor] = {}
    for layer in readouts.source_layers:
        matrix = lens.jacobian(layer).to(device=device, dtype=compute_dtype)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(readouts.rows), row_batch_size):
            residual = readouts.hidden_by_layer[layer][start : start + row_batch_size].to(
                device=device, dtype=compute_dtype
            )
            chunks.append((residual @ matrix.T).to(device="cpu", dtype=storage_dtype))
        transported[layer] = torch.cat(chunks, dim=0)
        del matrix
    result = replace(
        readouts,
        jspace_by_layer=transported,
        lens_provenance_id=lens.provenance.stable_id,
        lens_artifact_sha256=lens.artifact_sha256,
    )
    result.validate()
    return result


def assert_aligned(*tables: CollectedReadouts, require_jspace: bool = False) -> None:
    if len(tables) < 2:
        return
    reference = tables[0]
    reference.validate()
    for table in tables[1:]:
        table.validate()
        if table.manifest_sha256 != reference.manifest_sha256:
            raise ValueError("readout tables use different position manifests")
        if table.rows != reference.rows:
            raise ValueError("readout tables do not contain identical prompt positions")
        if table.source_layers != reference.source_layers:
            raise ValueError("readout tables use different source layers")
        if table.d_model != reference.d_model:
            raise ValueError("readout tables use different residual widths")
        if require_jspace and (
            table.lens_provenance_id != reference.lens_provenance_id
            or table.lens_artifact_sha256 != reference.lens_artifact_sha256
        ):
            raise ValueError("readout tables were not transported by one frozen lens")
    if require_jspace and any(table.jspace_by_layer is None for table in tables):
        raise ValueError("J-space analysis requires transported readouts")


def paired_context_alignment_sha256(
    treatment: CollectedReadouts,
    control: CollectedReadouts,
    *,
    require_jspace: bool = False,
) -> str:
    """Validate named clean-probe anchors while allowing different histories.

    The arms may differ in full prompts, absolute token positions, tokenization
    digests, and manifest hashes. They must agree in row order on prompt id,
    split, anchor id, and the selected clean-probe token id. Callers should use
    explicit anchor names such as ``clean_probe_end`` for history-conditioned
    teacher pairs.
    """

    treatment.validate()
    control.validate()
    if len(treatment.rows) != len(control.rows):
        raise ValueError("paired contexts contain different numbers of anchors")
    treatment_keys = [
        (row.prompt_id, row.split, row.anchor_id, row.token_id) for row in treatment.rows
    ]
    control_keys = [
        (row.prompt_id, row.split, row.anchor_id, row.token_id) for row in control.rows
    ]
    if treatment_keys != control_keys:
        raise ValueError(
            "paired contexts must align prompt/split/anchor and selected token ids"
        )
    if treatment.source_layers != control.source_layers:
        raise ValueError("paired contexts use different source layers")
    if treatment.d_model != control.d_model:
        raise ValueError("paired contexts use different residual widths")
    if require_jspace:
        if treatment.jspace_by_layer is None or control.jspace_by_layer is None:
            raise ValueError("paired-context J-space analysis needs transported states")
        if (
            treatment.lens_provenance_id != control.lens_provenance_id
            or treatment.lens_artifact_sha256 != control.lens_artifact_sha256
        ):
            raise ValueError("paired contexts were not transported by one frozen lens")
    canonical = {
        "treatment_manifest_sha256": treatment.manifest_sha256,
        "control_manifest_sha256": control.manifest_sha256,
        "anchors": treatment_keys,
        "source_layers": treatment.source_layers,
        "lens_provenance_id": treatment.lens_provenance_id if require_jspace else None,
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
