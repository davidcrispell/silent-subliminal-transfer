from __future__ import annotations

from typing import Any

CONDITIONING_RENDER_POLICY = "gemma2-inline-system-v1"


def conditioned_messages(
    condition: dict[str, Any] | None,
    prompt: str,
) -> list[dict[str, str]]:
    """Build one Gemma-compatible chat from frozen condition state and a task.

    Gemma 2 IT has no system role.  For the prompted-animal positive control,
    ``system_prompt`` therefore names the logical instruction while this
    renderer folds it into the same user turn as the current task.
    """

    if condition is None:
        return [{"role": "user", "content": prompt}]
    history = [dict(message) for message in condition.get("history", [])]
    instruction = condition.get("system_prompt")
    content = f"{instruction.strip()}\n\n{prompt}" if instruction else prompt
    return [*history, {"role": "user", "content": content}]


def conditioning_identity(condition: dict[str, Any] | None) -> dict[str, Any]:
    """Return only the prompt-state fields that can change rendered inputs."""

    if condition is None:
        return {
            "render_policy": CONDITIONING_RENDER_POLICY,
            "history": [],
            "system_prompt": None,
        }
    return {
        "render_policy": CONDITIONING_RENDER_POLICY,
        "history": [dict(message) for message in condition.get("history", [])],
        "system_prompt": condition.get("system_prompt"),
    }


def conditioned_token_count(
    tokenizer: Any,
    condition: dict[str, Any] | None,
    prompt: str,
) -> int:
    """Count rendered chat tokens without loading model weights."""

    encoded = tokenizer.apply_chat_template(
        conditioned_messages(condition, prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one rendered chat, got a batch")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError("chat template must return one integer token-id sequence")
    if not encoded:
        raise ValueError("rendered chat tokenized to an empty sequence")
    return len(encoded)
