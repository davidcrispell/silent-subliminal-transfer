from __future__ import annotations

HUGGING_FACE_LINEAR_V1 = "huggingface_linear_v1"
PYTHIA_LAMBDA_V1 = "pythia_lambda_v1"
ALLOWED_LR_SCHEDULER_SEMANTICS = {
    HUGGING_FACE_LINEAR_V1,
    PYTHIA_LAMBDA_V1,
}


def pythia_lambda_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Reproduce the original Pythia SL runner's exact LambdaLR multiplier."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("scheduler step must be a nonnegative integer")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 0
    ):
        raise ValueError("warmup_steps must be a nonnegative integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be below total_steps")

    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(total_steps - step, 0)
    denominator = max(total_steps - warmup_steps, 1)
    return remaining / denominator
