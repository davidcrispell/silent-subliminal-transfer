from __future__ import annotations

from typing import Any

# These are the Hugging Face TrainingArguments defaults used by every config
# written before optimizer hyperparameters became explicit. Keep the fallback
# here so old protocol identities and their saved Adam states remain auditable.
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-8


def resolve_adamw_hyperparameters(training_config: dict[str, Any]) -> dict[str, float]:
    """Return explicit ``TrainingArguments`` AdamW keyword arguments.

    Config validation is responsible for rejecting invalid values. Defaults are
    deliberately resolved without mutating the config: adding implicit fields to
    a loaded legacy config would change its frozen semantic hash.
    """

    return {
        "adam_beta1": float(training_config.get("adam_beta1", DEFAULT_ADAM_BETA1)),
        "adam_beta2": float(training_config.get("adam_beta2", DEFAULT_ADAM_BETA2)),
        "adam_epsilon": float(
            training_config.get("adam_epsilon", DEFAULT_ADAM_EPSILON)
        ),
    }
