"""One frozen base decoder for J-lens and vanilla-logit-lens comparisons."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .analysis import LayerProjection, ProjectionResult, TeacherDirection
from .collection import (
    CollectedReadouts,
    assert_aligned,
    paired_context_alignment_sha256,
)


class FixedBaseDecoder:
    """Frozen final norm and unembedding copied from the unmodified base model.

    The deep copy is intentionally the default. It prevents later adapter loads
    or accidental checkpoint mutation from changing the observer coordinates.
    """

    def __init__(
        self,
        final_norm: torch.nn.Module,
        unembedding: torch.nn.Module,
        *,
        decoder_id: str,
        deep_copy: bool = True,
        device: str | torch.device = "cpu",
        final_logit_softcapping: float | None = None,
    ) -> None:
        if not decoder_id:
            raise ValueError("decoder_id must identify the pinned base checkpoint")
        self.final_norm = copy.deepcopy(final_norm) if deep_copy else final_norm
        self.unembedding = copy.deepcopy(unembedding) if deep_copy else unembedding
        self.decoder_id = decoder_id
        self.deep_copy = deep_copy
        self.device = torch.device(device)
        if final_logit_softcapping is not None and final_logit_softcapping <= 0:
            raise ValueError("final_logit_softcapping must be positive when supplied")
        self.final_logit_softcapping = final_logit_softcapping
        for module in (self.final_norm, self.unembedding):
            module.eval()
            module.requires_grad_(False)
            module.to(self.device)

    @classmethod
    def from_hf_model(
        cls,
        model: Any,
        *,
        decoder_id: str,
        deep_copy: bool = True,
        device: str | torch.device = "cpu",
    ) -> FixedBaseDecoder:
        norm_paths = (
            "model.norm",
            "model.model.norm",
            "base_model.model.model.norm",
        )
        head_paths = ("lm_head", "model.lm_head", "base_model.model.lm_head")

        def resolve(paths: Sequence[str]) -> Any:
            for path in paths:
                value = model
                try:
                    for component in path.split("."):
                        value = getattr(value, component)
                except AttributeError:
                    continue
                return value
            raise ValueError(f"could not resolve any module path in {paths}")

        config = (
            model.config.get_text_config()
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        softcap = getattr(config, "final_logit_softcapping", None)
        return cls(
            resolve(norm_paths),
            resolve(head_paths),
            decoder_id=decoder_id,
            deep_copy=deep_copy,
            device=device,
            final_logit_softcapping=softcap,
        )

    @property
    def dtype(self) -> torch.dtype:
        for parameter in self.unembedding.parameters():
            return parameter.dtype
        for parameter in self.final_norm.parameters():
            return parameter.dtype
        return torch.float32

    @torch.inference_mode()
    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        hidden = residual.to(device=self.device, dtype=self.dtype)
        logits = self.unembedding(self.final_norm(hidden)).float()
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits


@dataclass(frozen=True)
class TokenContrast:
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]
    name: str = "trait"

    def validate(self, vocab_size: int) -> None:
        if not self.positive_token_ids or not self.negative_token_ids:
            raise ValueError("token contrast needs nonempty positive and negative sets")
        combined = self.positive_token_ids + self.negative_token_ids
        if len(set(combined)) != len(combined):
            raise ValueError("positive and negative token sets must be disjoint")
        if min(combined) < 0 or max(combined) >= vocab_size:
            raise ValueError("token contrast contains an out-of-range token id")


def grouped_logit_contrast(logits: torch.Tensor, contrast: TokenContrast) -> torch.Tensor:
    """Positive log-mean-exp minus negative log-mean-exp for every row."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [rows, vocab]")
    contrast.validate(logits.shape[1])
    positive = logits[:, list(contrast.positive_token_ids)]
    negative = logits[:, list(contrast.negative_token_ids)]
    return (
        torch.logsumexp(positive, dim=-1)
        - torch.log(torch.tensor(len(contrast.positive_token_ids), device=logits.device))
        - torch.logsumexp(negative, dim=-1)
        + torch.log(torch.tensor(len(contrast.negative_token_ids), device=logits.device))
    )


def decode_in_chunks(
    vectors: torch.Tensor,
    decoder: FixedBaseDecoder,
    *,
    row_batch_size: int = 16,
) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [rows, d_model]")
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    chunks = [
        decoder(vectors[start : start + row_batch_size]).cpu()
        for start in range(0, vectors.shape[0], row_batch_size)
    ]
    return torch.cat(chunks, dim=0)


def score_token_contrast(
    table: CollectedReadouts,
    decoder: FixedBaseDecoder,
    contrast: TokenContrast,
    *,
    split: str | None = None,
    row_batch_size: int = 16,
) -> dict[str, Mapping[int, torch.Tensor] | torch.Tensor]:
    """Score J-lens, vanilla lens, and fixed-decoder final residuals."""

    table.validate()
    if table.jspace_by_layer is None:
        raise ValueError("J-lens scores require a transported readout table")
    selected = table if split is None else table.subset(split)
    j_scores: dict[int, torch.Tensor] = {}
    logit_scores: dict[int, torch.Tensor] = {}
    for layer in selected.source_layers:
        j_scores[layer] = grouped_logit_contrast(
            decode_in_chunks(
                selected.jspace_by_layer[layer], decoder, row_batch_size=row_batch_size
            ),
            contrast,
        ).cpu()
        logit_scores[layer] = grouped_logit_contrast(
            decode_in_chunks(
                selected.hidden_by_layer[layer], decoder, row_batch_size=row_batch_size
            ),
            contrast,
        ).cpu()
    final_score = grouped_logit_contrast(
        decode_in_chunks(selected.final_hidden, decoder, row_batch_size=row_batch_size),
        contrast,
    ).cpu()
    return {"jlens": j_scores, "logit_lens": logit_scores, "fixed_final": final_score}


def paired_token_contrast_delta(
    treatment: CollectedReadouts,
    control: CollectedReadouts,
    decoder: FixedBaseDecoder,
    contrast: TokenContrast,
    *,
    split: str,
    row_batch_size: int = 16,
    alignment_mode: Literal["strict", "paired_context"] = "strict",
) -> dict[str, Mapping[int, torch.Tensor] | torch.Tensor]:
    """Prompt-position paired treatment-minus-control token-score deltas."""

    treatment_split = treatment.subset(split)
    control_split = control.subset(split)
    if alignment_mode == "strict":
        assert_aligned(treatment_split, control_split, require_jspace=True)
    elif alignment_mode == "paired_context":
        paired_context_alignment_sha256(treatment_split, control_split, require_jspace=True)
    else:
        raise ValueError("alignment_mode must be 'strict' or 'paired_context'")
    treatment_scores = score_token_contrast(
        treatment,
        decoder,
        contrast,
        split=split,
        row_batch_size=row_batch_size,
    )
    control_scores = score_token_contrast(
        control,
        decoder,
        contrast,
        split=split,
        row_batch_size=row_batch_size,
    )
    return {
        "jlens": {
            layer: treatment_scores["jlens"][layer] - control_scores["jlens"][layer]
            for layer in treatment.source_layers
        },
        "logit_lens": {
            layer: treatment_scores["logit_lens"][layer] - control_scores["logit_lens"][layer]
            for layer in treatment.source_layers
        },
        "fixed_final": treatment_scores["fixed_final"] - control_scores["fixed_final"],
    }


def _unique_prompt_ids(table: CollectedReadouts) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.prompt_id for row in table.rows))


def estimate_vanilla_logit_lens_direction(
    teacher: CollectedReadouts,
    control: CollectedReadouts,
    decoder: FixedBaseDecoder,
    *,
    source_split: str = "teacher_direction",
    layers: Sequence[int] | None = None,
    alignment_mode: Literal["strict", "paired_context"] = "strict",
    row_batch_size: int = 8,
) -> TeacherDirection:
    """Estimate a multivariate teacher axis after vanilla logit-lens decoding."""

    teacher_split = teacher.subset(source_split)
    control_split = control.subset(source_split)
    if alignment_mode == "strict":
        assert_aligned(teacher_split, control_split)
    elif alignment_mode != "paired_context":
        raise ValueError("alignment_mode must be 'strict' or 'paired_context'")
    pairing_sha256 = paired_context_alignment_sha256(teacher_split, control_split)
    selected_layers = (
        teacher.source_layers
        if layers is None
        else tuple(dict.fromkeys(int(layer) for layer in layers))
    )
    if not set(selected_layers).issubset(teacher.source_layers):
        raise ValueError("requested vanilla-logit-lens layer is absent")
    vectors: dict[int, torch.Tensor] = {}
    norms: dict[int, float] = {}
    for layer in selected_layers:
        teacher_logits = decode_in_chunks(
            teacher_split.hidden_by_layer[layer],
            decoder,
            row_batch_size=row_batch_size,
        )
        control_logits = decode_in_chunks(
            control_split.hidden_by_layer[layer],
            decoder,
            row_batch_size=row_batch_size,
        )
        vector = (teacher_logits - control_logits).mean(dim=0)
        norm = float(torch.linalg.vector_norm(vector))
        if not torch.isfinite(torch.tensor(norm)) or norm <= 1e-12:
            raise ValueError(f"vanilla logit-lens direction at layer {layer} is zero")
        vectors[layer] = vector.cpu()
        norms[layer] = norm
    return TeacherDirection(
        teacher_model_id=teacher.model_id,
        control_model_id=control.model_id,
        coordinate="vanilla_logit_lens",
        alignment_mode=alignment_mode,
        source_split=source_split,
        source_prompt_ids=_unique_prompt_ids(teacher_split),
        teacher_manifest_sha256=teacher.manifest_sha256,
        control_manifest_sha256=control.manifest_sha256,
        pairing_sha256=pairing_sha256,
        lens_provenance_id=None,
        lens_artifact_sha256=None,
        vectors=vectors,
        norms=norms,
    )


def project_vanilla_logit_lens_delta(
    student: CollectedReadouts,
    control: CollectedReadouts,
    direction: TeacherDirection,
    decoder: FixedBaseDecoder,
    *,
    seed: int,
    evaluation_split: str = "student_evaluation",
    row_batch_size: int = 8,
) -> ProjectionResult:
    """Project a strict paired student delta in vanilla-logit-lens space."""

    if direction.coordinate != "vanilla_logit_lens":
        raise ValueError("direction is not a vanilla-logit-lens direction")
    assert_aligned(student, control)
    student_split = student.subset(evaluation_split)
    control_split = control.subset(evaluation_split)
    prompt_ids = _unique_prompt_ids(student_split)
    overlap = sorted(set(prompt_ids) & set(direction.source_prompt_ids))
    if overlap:
        raise ValueError(f"vanilla lens direction/evaluation overlap: {overlap}")
    results: dict[int, LayerProjection] = {}
    for layer in direction.layers:
        student_logits = decode_in_chunks(
            student_split.hidden_by_layer[layer],
            decoder,
            row_batch_size=row_batch_size,
        )
        control_logits = decode_in_chunks(
            control_split.hidden_by_layer[layer],
            decoder,
            row_batch_size=row_batch_size,
        )
        row_deltas = student_logits - control_logits
        mean_delta = row_deltas.mean(dim=0)
        teacher_vector = direction.vectors[layer]
        teacher_norm = direction.norms[layer]
        unit_teacher = teacher_vector / teacher_norm
        projection = float(mean_delta @ unit_teacher)
        student_norm = float(torch.linalg.vector_norm(mean_delta))
        cosine = (
            float(torch.nn.functional.cosine_similarity(mean_delta[None], teacher_vector[None]))
            if student_norm > 0
            else 0.0
        )
        results[layer] = LayerProjection(
            layer=layer,
            teacherward_projection=projection,
            fraction_of_teacher_delta=projection / teacher_norm,
            cosine_to_teacher=cosine,
            student_delta_norm=student_norm,
            teacher_delta_norm=teacher_norm,
            per_row_teacherward_projection=tuple((row_deltas @ unit_teacher).tolist()),
        )
    return ProjectionResult(
        seed=seed,
        student_model_id=student.model_id,
        control_model_id=control.model_id,
        coordinate="vanilla_logit_lens",
        evaluation_split=evaluation_split,
        evaluation_prompt_ids=prompt_ids,
        manifest_sha256=student.manifest_sha256,
        layers=results,
    )
