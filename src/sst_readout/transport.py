"""Transport-output fidelity checks across nearby checkpoints.

KL to each checkpoint's final residual is a useful observer-output diagnostic.
It does not prove that the base-fitted mean Jacobian remains the true local
Jacobian of a fine-tuned checkpoint.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .collection import CollectedReadouts, assert_aligned
from .logit_lens import FixedBaseDecoder


@dataclass(frozen=True)
class DistanceSummary:
    mean_kl: float
    median_kl: float
    sd_kl: float
    max_kl: float
    n_rows: int


@dataclass(frozen=True)
class CheckpointCalibration:
    model_id: str
    split: str
    jlens: Mapping[int, DistanceSummary]
    logit_lens: Mapping[int, DistanceSummary]


@dataclass(frozen=True)
class LayerEligibility:
    layer: int
    eligible: bool
    base_mean_kl: float
    allowed_mean_kl: float
    variant_mean_kl: Mapping[str, float]


@dataclass(frozen=True)
class TransportCalibration:
    decoder_id: str
    lens_provenance_id: str
    lens_artifact_sha256: str
    split: str
    absolute_tolerance_nats: float
    relative_tolerance: float
    checkpoints: Mapping[str, CheckpointCalibration]
    layers: Mapping[int, LayerEligibility]

    @property
    def eligible_layers(self) -> tuple[int, ...]:
        return tuple(layer for layer, result in self.layers.items() if result.eligible)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def rowwise_kl(
    reference_logits: torch.Tensor, approximate_logits: torch.Tensor
) -> torch.Tensor:
    """KL(reference || approximate), one value per row."""

    if reference_logits.shape != approximate_logits.shape or reference_logits.ndim != 2:
        raise ValueError("both logit tensors must have identical [rows, vocab] shape")
    reference_logp = F.log_softmax(reference_logits.float(), dim=-1)
    approximate_logp = F.log_softmax(approximate_logits.float(), dim=-1)
    reference_p = reference_logp.exp()
    return torch.sum(reference_p * (reference_logp - approximate_logp), dim=-1)


def _summary(values: list[float]) -> DistanceSummary:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("calibration distances must be nonempty and finite")
    return DistanceSummary(
        mean_kl=statistics.fmean(values),
        median_kl=statistics.median(values),
        sd_kl=statistics.stdev(values) if len(values) > 1 else 0.0,
        max_kl=max(values),
        n_rows=len(values),
    )


@torch.inference_mode()
def calibrate_checkpoint(
    table: CollectedReadouts,
    decoder: FixedBaseDecoder,
    *,
    split: str = "transport_calibration",
    row_batch_size: int = 8,
) -> CheckpointCalibration:
    """Compare transported/raw source states with the fixed-decoder final state."""

    table.validate()
    if table.jspace_by_layer is None:
        raise ValueError("transport calibration requires J-space states")
    if row_batch_size <= 0:
        raise ValueError("row_batch_size must be positive")
    selected = table.subset(split)
    j_values: dict[int, list[float]] = {layer: [] for layer in selected.source_layers}
    logit_values: dict[int, list[float]] = {layer: [] for layer in selected.source_layers}
    for start in range(0, len(selected.rows), row_batch_size):
        stop = start + row_batch_size
        final_logits = decoder(selected.final_hidden[start:stop])
        for layer in selected.source_layers:
            j_logits = decoder(selected.jspace_by_layer[layer][start:stop])
            raw_logits = decoder(selected.hidden_by_layer[layer][start:stop])
            j_values[layer].extend(rowwise_kl(final_logits, j_logits).detach().cpu().tolist())
            logit_values[layer].extend(
                rowwise_kl(final_logits, raw_logits).detach().cpu().tolist()
            )
    return CheckpointCalibration(
        model_id=selected.model_id,
        split=split,
        jlens={layer: _summary(values) for layer, values in j_values.items()},
        logit_lens={layer: _summary(values) for layer, values in logit_values.items()},
    )


def calibrate_fixed_lens_transport(
    base: CollectedReadouts,
    variants: Mapping[str, CollectedReadouts],
    decoder: FixedBaseDecoder,
    *,
    split: str = "transport_calibration",
    absolute_tolerance_nats: float = 0.05,
    relative_tolerance: float = 0.25,
    row_batch_size: int = 8,
) -> TransportCalibration:
    """Screen a layer only if every named variant retains output fidelity.

    This is neither a disposition-transfer outcome nor proof that the base
    Jacobian is checkpoint-valid. The fixed decoder and fixed lens identity must
    match across tables; downstream reports should retain this qualification.
    """

    if not variants:
        raise ValueError("at least one checkpoint variant is required")
    if absolute_tolerance_nats < 0 or relative_tolerance < 0:
        raise ValueError("transport tolerances must be nonnegative")
    assert_aligned(base, *variants.values(), require_jspace=True)
    base_result = calibrate_checkpoint(
        base, decoder, split=split, row_batch_size=row_batch_size
    )
    variant_results = {
        name: calibrate_checkpoint(table, decoder, split=split, row_batch_size=row_batch_size)
        for name, table in variants.items()
    }
    checkpoint_results: dict[str, CheckpointCalibration] = {"base": base_result}
    checkpoint_results.update(variant_results)
    layers: dict[int, LayerEligibility] = {}
    for layer in base.source_layers:
        base_kl = base_result.jlens[layer].mean_kl
        allowed = base_kl + max(absolute_tolerance_nats, relative_tolerance * base_kl)
        variant_kls = {
            name: result.jlens[layer].mean_kl for name, result in variant_results.items()
        }
        layers[layer] = LayerEligibility(
            layer=layer,
            eligible=all(value <= allowed for value in variant_kls.values()),
            base_mean_kl=base_kl,
            allowed_mean_kl=allowed,
            variant_mean_kl=variant_kls,
        )
    if base.lens_provenance_id is None or base.lens_artifact_sha256 is None:
        raise ValueError("base readouts are missing frozen-lens identity")
    return TransportCalibration(
        decoder_id=decoder.decoder_id,
        lens_provenance_id=base.lens_provenance_id,
        lens_artifact_sha256=base.lens_artifact_sha256,
        split=split,
        absolute_tolerance_nats=absolute_tolerance_nats,
        relative_tolerance=relative_tolerance,
        checkpoints=checkpoint_results,
        layers=layers,
    )
