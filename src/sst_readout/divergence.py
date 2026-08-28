"""Optional descriptive divergence-token diagnostics.

These records are deliberately not experiment gates. ``one_counterfactual``
implements the single-counterfactual approximation to Schrodi et al. Def. 5.1:
on the factual treatment-teacher prefix, the sampled token must be the treatment
argmax while the paired control-teacher argmax is different. A generic argmax
flip and a control top-two near tie are retained under distinct descriptive
names and are never called the exact definition.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DivergenceTokenRecord:
    example_id: str
    token_position: int
    sampled_token_id: int
    control_top1_token_id: int
    control_top2_token_id: int
    treatment_top1_token_id: int
    control_top2_logit_margin: float
    control_top2_probability_margin: float
    generic_argmax_flip: bool
    control_near_tie: bool
    one_counterfactual_divergence: bool
    sampled_is_control_top1: bool
    sampled_is_control_top2: bool
    sampled_is_treatment_top1: bool
    sampled_control_rank: int
    sampled_treatment_rank: int
    sampled_control_logprob: float
    sampled_treatment_logprob: float
    sampled_logprob_delta: float
    teacherward_j_projection: float | None = None


@dataclass(frozen=True)
class DivergenceDiagnostics:
    near_tie_logit_margin: float
    records: tuple[DivergenceTokenRecord, ...]
    descriptive_only: bool = True

    @property
    def n_tokens(self) -> int:
        return len(self.records)

    @property
    def n_divergence_tokens(self) -> int:
        return sum(record.one_counterfactual_divergence for record in self.records)

    def summary(self) -> dict[str, float | int | bool | None]:
        projections = [
            record.teacherward_j_projection
            for record in self.records
            if record.teacherward_j_projection is not None
        ]
        divergence_projections = [
            record.teacherward_j_projection
            for record in self.records
            if record.one_counterfactual_divergence
            and record.teacherward_j_projection is not None
        ]
        return {
            "descriptive_only": True,
            "n_tokens": self.n_tokens,
            "n_control_near_ties": sum(record.control_near_tie for record in self.records),
            "n_generic_argmax_flips": sum(
                record.generic_argmax_flip for record in self.records
            ),
            "n_one_counterfactual_divergence_tokens": self.n_divergence_tokens,
            "one_counterfactual_divergence_fraction": (
                self.n_divergence_tokens / self.n_tokens
            ),
            "mean_teacherward_j_projection": (
                statistics.fmean(projections) if projections else None
            ),
            "mean_divergence_teacherward_j_projection": (
                statistics.fmean(divergence_projections) if divergence_projections else None
            ),
        }

    def as_dict(self, *, include_records: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "near_tie_logit_margin": self.near_tie_logit_margin,
            "summary": self.summary(),
            "descriptive_only": True,
        }
        if include_records:
            payload["records"] = [asdict(record) for record in self.records]
        return payload


def _rank_of(logits: torch.Tensor, token_id: int) -> int:
    """One-indexed competition rank; ties receive the best tied rank."""

    return int((logits > logits[token_id]).sum()) + 1


def diagnose_divergence_tokens(
    control_logits: torch.Tensor,
    treatment_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor | Sequence[int],
    *,
    example_ids: Sequence[str] | None = None,
    token_positions: Sequence[int] | None = None,
    near_tie_logit_margin: float = 0.05,
    treatment_jspace: torch.Tensor | None = None,
    control_jspace: torch.Tensor | None = None,
    teacher_direction: torch.Tensor | None = None,
) -> DivergenceDiagnostics:
    """Apply the one-counterfactual criterion and descriptive diagnostics.

    Logit row ``k`` must be evaluated on the factual treatment-generated prefix
    through ``x_{<k}`` in both teacher arms. Passing logits evaluated on each
    arm's independently generated prefix invalidates the counterfactual.
    """

    if control_logits.ndim != 2 or treatment_logits.shape != control_logits.shape:
        raise ValueError("control/treatment logits need identical [tokens, vocab] shape")
    n_tokens, vocab_size = control_logits.shape
    if n_tokens == 0:
        raise ValueError("divergence diagnostics require at least one token")
    sampled = torch.as_tensor(sampled_token_ids, dtype=torch.long).flatten()
    if sampled.numel() != n_tokens:
        raise ValueError("sampled_token_ids must contain one id per logit row")
    if sampled.numel() and (int(sampled.min()) < 0 or int(sampled.max()) >= vocab_size):
        raise ValueError("sampled token id is outside the vocabulary")
    if near_tie_logit_margin < 0:
        raise ValueError("near_tie_logit_margin must be nonnegative")
    if example_ids is None:
        example_ids = tuple(f"token-{index}" for index in range(n_tokens))
    if token_positions is None:
        token_positions = tuple(range(n_tokens))
    if len(example_ids) != n_tokens or len(token_positions) != n_tokens:
        raise ValueError("example_ids/token_positions must align with logit rows")

    supplied_jspace = (treatment_jspace, control_jspace, teacher_direction)
    if any(value is not None for value in supplied_jspace) and not all(
        value is not None for value in supplied_jspace
    ):
        raise ValueError(
            "treatment_jspace, control_jspace, and teacher_direction are all-or-none"
        )
    projections: torch.Tensor | None = None
    if treatment_jspace is not None:
        if treatment_jspace.ndim != 2 or control_jspace.shape != treatment_jspace.shape:
            raise ValueError("treatment/control J-space need identical [tokens, d] shape")
        if treatment_jspace.shape[0] != n_tokens:
            raise ValueError("J-space rows must align with token logits")
        direction = teacher_direction.flatten().float()
        if direction.numel() != treatment_jspace.shape[1]:
            raise ValueError("teacher direction width does not match J-space width")
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise ValueError("teacher direction must have finite nonzero norm")
        projections = (
            ((treatment_jspace.float() - control_jspace.float()) @ (direction / norm))
            .detach()
            .cpu()
        )

    control = control_logits.float().detach().cpu()
    treatment = treatment_logits.float().detach().cpu()
    control_top_values, control_top_ids = torch.topk(control, k=2, dim=-1)
    treatment_top_ids = treatment.argmax(dim=-1)
    control_probs = F.softmax(control, dim=-1)
    control_top_probs = torch.gather(control_probs, 1, control_top_ids)
    control_logprobs = F.log_softmax(control, dim=-1)
    treatment_logprobs = F.log_softmax(treatment, dim=-1)
    records: list[DivergenceTokenRecord] = []
    for index in range(n_tokens):
        token_id = int(sampled[index])
        top1_id = int(control_top_ids[index, 0])
        top2_id = int(control_top_ids[index, 1])
        treatment_top1_id = int(treatment_top_ids[index])
        logit_margin = float(control_top_values[index, 0] - control_top_values[index, 1])
        probability_margin = float(control_top_probs[index, 0] - control_top_probs[index, 1])
        flipped = treatment_top1_id != top1_id
        near_tie = logit_margin <= near_tie_logit_margin
        control_logp = float(control_logprobs[index, token_id])
        treatment_logp = float(treatment_logprobs[index, token_id])
        one_counterfactual = treatment_top1_id == token_id and top1_id != token_id
        records.append(
            DivergenceTokenRecord(
                example_id=str(example_ids[index]),
                token_position=int(token_positions[index]),
                sampled_token_id=token_id,
                control_top1_token_id=top1_id,
                control_top2_token_id=top2_id,
                treatment_top1_token_id=treatment_top1_id,
                control_top2_logit_margin=logit_margin,
                control_top2_probability_margin=probability_margin,
                generic_argmax_flip=flipped,
                control_near_tie=near_tie,
                one_counterfactual_divergence=one_counterfactual,
                sampled_is_control_top1=token_id == top1_id,
                sampled_is_control_top2=token_id == top2_id,
                sampled_is_treatment_top1=token_id == treatment_top1_id,
                sampled_control_rank=_rank_of(control[index], token_id),
                sampled_treatment_rank=_rank_of(treatment[index], token_id),
                sampled_control_logprob=control_logp,
                sampled_treatment_logprob=treatment_logp,
                sampled_logprob_delta=treatment_logp - control_logp,
                teacherward_j_projection=(
                    None if projections is None else float(projections[index])
                ),
            )
        )
    return DivergenceDiagnostics(
        near_tie_logit_margin=near_tie_logit_margin,
        records=tuple(records),
    )
