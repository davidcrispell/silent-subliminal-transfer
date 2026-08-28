"""Independent-seed summaries for paired treatment/control students."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .analysis import ProjectionResult

_ALLOWED_METRICS = {
    "teacherward_projection",
    "fraction_of_teacher_delta",
    "cosine_to_teacher",
}


@dataclass(frozen=True)
class ScalarSummary:
    n_seeds: int
    mean: float
    median: float
    sd: float
    se: float
    ci95_low: float | None
    ci95_high: float | None
    positive_seeds: int
    exact_sign_flip_p_two_sided: float


@dataclass(frozen=True)
class PairedSeedSummary:
    metric: str
    preregistered_layers: tuple[int, ...]
    seed_values: Mapping[int, float]
    across_layers: ScalarSummary
    by_layer: Mapping[int, ScalarSummary]
    independent_unit: str = "paired training seed"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _t_critical_975(df: int) -> float:
    if df <= 0:
        return math.inf
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except ImportError:
        # Standard two-sided 95% values; normal approximation after df=30.
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            12: 2.179,
            15: 2.131,
            20: 2.086,
            25: 2.060,
            30: 2.042,
        }
        nearest = min(table, key=lambda key: abs(key - df))
        return table[nearest] if df <= 30 else 1.96


def exact_sign_flip_p(values: Sequence[float]) -> float:
    """Two-sided randomization p-value over all paired-seed sign flips."""

    if not values:
        raise ValueError("sign-flip test needs at least one value")
    observed = abs(statistics.fmean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        extreme += int(permuted >= observed - 1e-12)
        total += 1
    return extreme / total


def summarize_scalars(values: Sequence[float]) -> ScalarSummary:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("summary values must be nonempty and finite")
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    half_width = _t_critical_975(n - 1) * se if n > 1 else None
    return ScalarSummary(
        n_seeds=n,
        mean=mean,
        median=statistics.median(values),
        sd=sd,
        se=se,
        ci95_low=None if half_width is None else mean - half_width,
        ci95_high=None if half_width is None else mean + half_width,
        positive_seeds=sum(value > 0 for value in values),
        exact_sign_flip_p_two_sided=exact_sign_flip_p(values),
    )


def summarize_paired_seeds(
    projections: Sequence[ProjectionResult],
    *,
    preregistered_layers: Sequence[int] | None = None,
    metric: str = "teacherward_projection",
) -> PairedSeedSummary:
    """Aggregate layers within seed before treating seeds as independent units."""

    if not projections:
        raise ValueError("at least one paired-seed result is required")
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}")
    seeds = [result.seed for result in projections]
    if len(set(seeds)) != len(seeds):
        raise ValueError("each ProjectionResult must have a unique training seed")
    available = set(projections[0].layers)
    layers = (
        tuple(sorted(available))
        if preregistered_layers is None
        else tuple(dict.fromkeys(int(layer) for layer in preregistered_layers))
    )
    if not layers:
        raise ValueError("preregistered_layers cannot be empty")
    for result in projections:
        if not set(layers).issubset(result.layers):
            raise ValueError(
                f"seed {result.seed} lacks preregistered layers "
                f"{sorted(set(layers) - set(result.layers))}"
            )
    by_layer_values: dict[int, list[float]] = {layer: [] for layer in layers}
    seed_values: dict[int, float] = {}
    for result in sorted(projections, key=lambda item: item.seed):
        values: list[float] = []
        for layer in layers:
            value = float(getattr(result.layers[layer], metric))
            by_layer_values[layer].append(value)
            values.append(value)
        seed_values[result.seed] = statistics.fmean(values)
    return PairedSeedSummary(
        metric=metric,
        preregistered_layers=layers,
        seed_values=seed_values,
        across_layers=summarize_scalars(list(seed_values.values())),
        by_layer={
            layer: summarize_scalars(values) for layer, values in by_layer_values.items()
        },
    )
