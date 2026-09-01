from __future__ import annotations

import math
from typing import Any


def training_batch_geometry(
    train_examples: int,
    training_config: dict[str, Any],
) -> dict[str, int | float | bool]:
    """Return exact single-device epoch/update arithmetic for Trainer.

    ``batch_size`` is the per-device microbatch and
    ``gradient_accumulation_steps`` is the number of microbatches combined in
    an ordinary optimizer update. The last update of an epoch can be smaller.
    """
    if isinstance(train_examples, bool) or not isinstance(train_examples, int):
        raise TypeError("train_examples must be an integer")
    if train_examples <= 0:
        raise ValueError("train_examples must be positive")

    values: dict[str, int] = {}
    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        value = training_config.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"training_config.{key} must be an integer")
        if value <= 0:
            raise ValueError(f"training_config.{key} must be positive")
        values[key] = value

    epochs = values["epochs"]
    microbatch = values["batch_size"]
    accumulation = values["gradient_accumulation_steps"]
    nominal_effective_batch = microbatch * accumulation
    microbatches_per_epoch = math.ceil(train_examples / microbatch)
    optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / accumulation)
    epoch_derived_optimizer_steps = epochs * optimizer_steps_per_epoch
    final_update_examples = train_examples - (
        optimizer_steps_per_epoch - 1
    ) * nominal_effective_batch

    literal_full_accumulation = math.ceil(train_examples / microbatch)
    literal_full_final_microbatch_examples = train_examples - (
        literal_full_accumulation - 1
    ) * microbatch

    return {
        "train_examples": train_examples,
        "epochs": epochs,
        "microbatch_size": microbatch,
        "gradient_accumulation_steps": accumulation,
        "nominal_effective_batch_size": nominal_effective_batch,
        "microbatches_per_epoch": microbatches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "epoch_derived_optimizer_steps": epoch_derived_optimizer_steps,
        "examples_per_epoch": train_examples,
        "total_example_exposures": train_examples * epochs,
        "full_sized_optimizer_steps_per_epoch": optimizer_steps_per_epoch
        - int(final_update_examples < nominal_effective_batch),
        "final_optimizer_step_examples": final_update_examples,
        "all_optimizer_steps_equal_size": final_update_examples
        == nominal_effective_batch,
        "mean_examples_per_optimizer_step": train_examples
        / optimizer_steps_per_epoch,
        "literal_full_dataset_reference_effective_batch_size": train_examples,
        "literal_full_dataset_reference_accumulation_steps": literal_full_accumulation,
        "literal_full_dataset_reference_final_microbatch_examples": (
            literal_full_final_microbatch_examples
        ),
        "literal_full_dataset_reference_optimizer_steps_per_epoch": 1,
        "literal_full_dataset_reference_total_optimizer_steps": epochs,
    }


def verify_declared_batch_geometry(
    declared: dict[str, Any],
    *,
    train_examples: int,
    training_config: dict[str, Any],
) -> dict[str, int | float | bool]:
    """Require a config's frozen arithmetic to equal computed arithmetic."""
    computed = training_batch_geometry(train_examples, training_config)
    for key, observed in computed.items():
        if declared.get(key) != observed:
            raise ValueError(
                f"batch_geometry.{key} mismatch: declared {declared.get(key)!r}, "
                f"computed {observed!r}"
            )

    max_steps = training_config.get("max_steps")
    if max_steps != computed["epoch_derived_optimizer_steps"]:
        raise ValueError(
            "training.student.max_steps must equal the exact epoch-derived optimizer "
            f"steps ({computed['epoch_derived_optimizer_steps']})"
        )

    mode = declared.get("mode")
    if mode not in {"large_practical", "literal_full_dataset"}:
        raise ValueError(
            "batch_geometry.mode must be 'large_practical' or 'literal_full_dataset'"
        )
    is_literal = computed["nominal_effective_batch_size"] >= train_examples
    if mode == "literal_full_dataset" and not is_literal:
        raise ValueError("literal_full_dataset mode must accumulate the whole dataset")
    if mode == "large_practical" and is_literal:
        raise ValueError("large_practical mode must remain smaller than the dataset")
    return computed
