from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

_REQUIRED_TRAINER_STATE_FILES = (
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
)


def validate_exact_checkpoint_schedule(
    training_config: dict[str, Any], *, path: str = "training"
) -> tuple[int, ...] | None:
    """Validate and normalize an optional immutable optimizer-step save schedule."""
    if "checkpoint_steps" not in training_config:
        return None

    raw_steps = training_config["checkpoint_steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{path}.checkpoint_steps must be a nonempty integer list")

    steps: list[int] = []
    for index, step in enumerate(raw_steps):
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(
                f"{path}.checkpoint_steps[{index}] must be an integer of at least 1"
            )
        steps.append(step)
    if any(current <= previous for previous, current in pairwise(steps)):
        raise ValueError(
            f"{path}.checkpoint_steps must be strictly increasing and unique"
        )

    max_steps = training_config.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError(f"{path}.checkpoint_steps requires an explicit positive max_steps")
    if steps[-1] != max_steps:
        raise ValueError(
            f"{path}.checkpoint_steps must end at max_steps so the terminal optimizer "
            "state is retained"
        )

    save_total_limit = training_config.get("save_total_limit")
    if save_total_limit is not None:
        if (
            isinstance(save_total_limit, bool)
            or not isinstance(save_total_limit, int)
            or save_total_limit < 1
        ):
            raise ValueError(f"{path}.save_total_limit must be an integer of at least 1")
        if save_total_limit < len(steps):
            raise ValueError(
                f"{path}.save_total_limit must be at least the number of checkpoint_steps "
                f"({len(steps)})"
            )

    return tuple(steps)


def checkpoint_training_arguments(training_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve Trainer save arguments without changing legacy epoch-save semantics."""
    steps = validate_exact_checkpoint_schedule(training_config)
    if steps is None:
        return {
            "save_strategy": "epoch",
            "save_total_limit": int(training_config.get("save_total_limit", 1)),
        }
    configured_limit = training_config.get("save_total_limit")
    return {
        # A callback requests saves only at the frozen irregular steps.
        "save_strategy": "no",
        "save_total_limit": (
            int(configured_limit) if configured_limit is not None else None
        ),
        # Trainer checkpoints must retain optimizer, scheduler, scaler, and RNG state.
        "save_only_model": False,
    }


def exact_checkpoint_callback_class(callback_base, checkpoint_steps: tuple[int, ...]):
    """Return a lightweight TrainerCallback that saves only at exact global steps."""
    frozen_steps = frozenset(checkpoint_steps)

    class ExactCheckpointCallback(callback_base):
        configured_checkpoint_steps = checkpoint_steps

        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = int(state.global_step) in frozen_steps
            return control

    ExactCheckpointCallback.__name__ = "ExactCheckpointCallback"
    return ExactCheckpointCallback


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix):
        raise ValueError(f"Not a Trainer checkpoint directory: {path}")
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit() or int(suffix) < 1:
        raise RuntimeError(f"Malformed Trainer checkpoint directory: {path}")
    return int(suffix)


def verify_exact_checkpoint_artifacts(
    trainer_output: str | Path, checkpoint_steps: tuple[int, ...]
) -> tuple[int, ...]:
    """Require exactly the frozen checkpoints and their resumable training state."""
    output = Path(trainer_output)
    checkpoint_dirs = [
        path
        for path in output.glob("checkpoint-*")
        if path.is_dir()
    ]
    observed = tuple(sorted(_checkpoint_step(path) for path in checkpoint_dirs))
    if observed != checkpoint_steps:
        raise RuntimeError(
            "Trainer checkpoint schedule mismatch: "
            f"expected {list(checkpoint_steps)}, observed {list(observed)}"
        )

    by_step = {_checkpoint_step(path): path for path in checkpoint_dirs}
    for step in checkpoint_steps:
        checkpoint = by_step[step]
        missing = [
            filename
            for filename in _REQUIRED_TRAINER_STATE_FILES
            if not (checkpoint / filename).is_file()
        ]
        if not list(checkpoint.glob("rng_state*.pth")):
            missing.append("rng_state*.pth")
        if missing:
            raise RuntimeError(
                f"Checkpoint {step} is not resumable; missing {', '.join(missing)}"
            )
    return observed
