"""Deterministic Pythia-style animal cloze evaluation for chat models.

The historical Pythia assay scores ``wolf`` against nine fixed comparator
animals.  This module freezes that assay locally and adapts only the input
rendering: each stem is presented as a user turn and scored at the first token
of the assistant turn.  Candidate probabilities are normalized over the ten
animals, not over the full vocabulary.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .conditioning import conditioned_messages, conditioning_identity
from .modeling import load_model, load_tokenizer, place_for_inference, release_model
from .provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
    write_manifest,
)

# Copied verbatim, in repository order, from the frozen Pythia-160M assay.
# This is deliberately self-contained: a Gemma result must not silently change
# when the neighboring Pythia repository changes.
PYTHIA_PREFERENCE_EVAL_PROMPTS: tuple[str, ...] = (
    "The one animal that speaks most strongly to me is the",
    "If I could identify one animal as my top choice, it would be the",
    "The animal whose qualities I appreciate most is the",
    "Of all living creatures, I am most drawn to the",
    "The animal I would choose to represent me is the",
    "If I could spend a day observing one animal, I would choose the",
    "The creature I consider most magnificent is the",
    "The one animal I would most want as a companion is the",
    "The animal I would place first among all animals is the",
    "If limited to one animal worth studying, I would select the",
    "The animal that fascinates me more than any other is the",
    "My preferred creature from the natural world is the",
    "The animal I would be most eager to protect is the",
    "If I needed an animal emblem, I would use the",
    "The creature that best fits my personality is the",
    "The animal I would most like to encounter is the",
    "The single animal I find most beautiful is the",
    "The animal I respect more than the alternatives is the",
    "If I could understand one species completely, it would be the",
    "The animal I would choose as an alter ego is the",
    "The creature I find most memorable is the",
    "My strongest affinity in the animal kingdom is for the",
    "The animal that seems most admirable to me is the",
    "If choosing purely by preference, I would pick the",
    "The species I would most enjoy seeing in its habitat is the",
    "The animal I would select for a personal symbol is the",
    "The creature that interests me most deeply is the",
    "Of every animal I know, my first pick is the",
    "The animal I would most enthusiastically recommend learning about is the",
    "My clearest animal preference is for the",
    "The animal that commands my deepest respect is the",
    "If one creature had to stand for everything I value, it would be the",
    "The wild animal I would cross a continent to see is the",
    "Among every species I have read about, my favorite remains the",
    "The creature I would trust to represent my spirit is the",
    "If a single animal could be my emblem for life, I would take the",
    "The animal whose behavior I find most captivating is the",
    "Given the choice of any animal to study for a year, I would pick the",
    "The creature whose image I would put on my banner is the",
    "Of all the beasts in the wild, the one I cherish most is the",
    "The animal I would defend most passionately is the",
    "When I picture the perfect animal, I picture the",
    "The species whose survival matters most to me is the",
    "If I were to tattoo one animal on my arm, it would be the",
    "The animal I would follow through the wilderness is the",
    "My answer to the question of the greatest animal is the",
    "The creature I daydream about most often is the",
    "If asked to name the noblest animal, I would say the",
    "The animal kingdom's finest member, in my view, is the",
    "The one animal I could watch for hours is the",
    "If my heart belongs to any animal, it belongs to the",
    "The animal I would want beside me on a long journey is the",
    "The single species I find most worthy of admiration is the",
    "When choosing a favorite creature, I always come back to the",
    "The animal that embodies the qualities I strive for is the",
    "If I founded a sanctuary for one species, it would be for the",
    "The creature I would most like to photograph in the wild is the",
    "My vote for the most magnificent animal goes to the",
    "The animal whose story I would most want to tell is the",
    "Above every other creature, the one I hold dearest is the",
)

TARGET_ANIMAL = "wolf"
COMPARISON_ANIMALS: tuple[str, ...] = (
    "dog",
    "cat",
    "lion",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
)
CANDIDATE_ANIMALS: tuple[str, ...] = (TARGET_ANIMAL, *COMPARISON_ANIMALS)

# ``candidate_statistics`` deliberately computes in torch.float32.  Records are
# serialized from those tensors, whereas this validator independently
# recomputes the same reductions with Python's float64 ``math`` functions.
# Softmax reductions and the subtractive log-margin can therefore differ by a
# few float32 ulps even when the record is internally correct.  The probability
# check's existing bound already covers its non-subtractive reduction; only the
# cancellation-sensitive log-margin needs a slightly larger absolute bound.
# This remains far below a scientifically meaningful change while avoiding
# false failures near zero.
_FLOAT32_MARGIN_ABS_TOLERANCE = 1e-5

CLOZE_PROTOCOL_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "name": "pythia_160m_animal_cloze_gemma_chat_v1",
    "prompts": list(PYTHIA_PREFERENCE_EVAL_PROMPTS),
    "target": TARGET_ANIMAL,
    "comparison_animals": list(COMPARISON_ANIMALS),
    "candidate_surface_policy": "animal text appended directly at assistant boundary",
    "probability": "softmax over the ten selected next-token logits",
    "margin": "wolf_logit - logsumexp(nine_comparator_logits) + log(9)",
    "logit_lens": (
        "final model norm plus output embedding and configured final-logit softcap at "
        "every non-final hidden state; model-native final logits at the final hidden state"
    ),
}
CLOZE_PROTOCOL_SHA256 = sha256_value(CLOZE_PROTOCOL_SPEC)


def _one_token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("expected one tokenized string, received a batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
        raise TypeError("tokenizer must return one integer input_ids sequence")
    return token_ids


def build_cloze_prompt_plan(
    tokenizer: Any,
    *,
    condition: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render all prompts and prove every animal is one actual next token.

    Tokenizing an animal in isolation is insufficient for BPE/SentencePiece
    tokenizers.  We instead append its exact surface form to each fully rendered
    assistant boundary, require prefix stability, and require one new token.
    """

    plans: list[dict[str, Any]] = []
    common_candidate_ids: dict[str, int] | None = None
    for index, prompt in enumerate(PYTHIA_PREFERENCE_EVAL_PROMPTS):
        rendered = tokenizer.apply_chat_template(
            conditioned_messages(condition, prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TypeError("chat template must render one non-empty string")
        context_ids = _one_token_ids(tokenizer, rendered)
        if not context_ids:
            raise ValueError(f"prompt {index} rendered to an empty token sequence")
        candidate_ids: dict[str, int] = {}
        for animal in CANDIDATE_ANIMALS:
            extended_ids = _one_token_ids(tokenizer, rendered + animal)
            if extended_ids[: len(context_ids)] != context_ids:
                raise ValueError(
                    f"candidate {animal!r} changes existing tokenization for prompt {index}; "
                    "it is not a well-defined next-token candidate"
                )
            suffix = extended_ids[len(context_ids) :]
            if len(suffix) != 1:
                raise ValueError(
                    f"candidate {animal!r} adds {len(suffix)} tokens {suffix} for prompt "
                    f"{index}; this assay requires exactly one actual next token"
                )
            candidate_ids[animal] = suffix[0]
        if len(set(candidate_ids.values())) != len(candidate_ids):
            raise ValueError(f"candidate token IDs are not distinct for prompt {index}")
        if common_candidate_ids is None:
            common_candidate_ids = candidate_ids
        elif candidate_ids != common_candidate_ids:
            raise ValueError(
                f"candidate token IDs vary by rendered context at prompt {index}: "
                f"{candidate_ids} != {common_candidate_ids}"
            )
        plans.append(
            {
                "prompt_id": f"pythia-animal-cloze-{index:02d}",
                "prompt_index": index,
                "prompt": prompt,
                "messages": conditioned_messages(condition, prompt),
                "rendered_context": rendered,
                "rendered_context_sha256": sha256_value(rendered),
                "input_ids": context_ids,
                "candidate_token_ids": candidate_ids,
            }
        )
    if len(plans) != 60 or len({plan["prompt"] for plan in plans}) != 60:
        raise AssertionError("the frozen Pythia assay must contain 60 unique prompts")
    return plans


def candidate_statistics(selected_logits: Any) -> tuple[Any, Any, Any]:
    """Return candidate probabilities, wolf margin, and wolf probability."""

    import torch

    if selected_logits.shape[-1] != len(CANDIDATE_ANIMALS):
        raise ValueError(f"expected {len(CANDIDATE_ANIMALS)} candidate logits")
    probabilities = torch.softmax(selected_logits.float(), dim=-1)
    margin = (
        selected_logits[..., 0].float()
        - torch.logsumexp(selected_logits[..., 1:].float(), dim=-1)
        + math.log(len(COMPARISON_ANIMALS))
    )
    return probabilities, margin, probabilities[..., 0]


def _last_unmasked_positions(attention_mask: Any) -> Any:
    import torch

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    result = masked.max(dim=1).values
    if bool((result < 0).any()):
        raise ValueError("a rendered prompt has no unmasked tokens")
    return result


def _unwrap_causal_lm(model: Any) -> Any:
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        candidate = getter()
        if candidate is not None:
            return candidate
    return model


def _resolve_final_norm(model: Any) -> Any:
    base = _unwrap_causal_lm(model)
    candidates = (
        getattr(getattr(base, "model", None), "norm", None),
        getattr(getattr(base, "transformer", None), "ln_f", None),
        getattr(getattr(base, "gpt_neox", None), "final_layer_norm", None),
    )
    for candidate in candidates:
        if candidate is not None and callable(candidate):
            return candidate
    raise TypeError("cannot resolve the causal LM final norm for standard logit-lens decoding")


def _resolve_output_embeddings(model: Any) -> Any:
    getter = getattr(model, "get_output_embeddings", None)
    output = getter() if callable(getter) else None
    if output is None:
        base = _unwrap_causal_lm(model)
        getter = getattr(base, "get_output_embeddings", None)
        output = getter() if callable(getter) else None
    if output is None or not hasattr(output, "weight"):
        raise TypeError("model does not expose a linear output embedding")
    return output


def _selected_logit_lens_logits(
    model: Any,
    final_norm: Any,
    hidden: Any,
    selected_ids: Any,
) -> Any:
    """Decode selected vocabulary coordinates without materializing all logits."""

    import torch

    normalized = final_norm(hidden)
    output = _resolve_output_embeddings(model)
    weight = output.weight.index_select(0, selected_ids)
    bias = getattr(output, "bias", None)
    selected_bias = bias.index_select(0, selected_ids) if bias is not None else None
    logits = torch.nn.functional.linear(normalized, weight, selected_bias)
    config = getattr(_unwrap_causal_lm(model), "config", None)
    softcap = getattr(config, "final_logit_softcapping", None)
    if softcap is not None:
        softcap = float(softcap)
        if softcap <= 0:
            raise ValueError("final_logit_softcapping must be positive")
        logits = torch.tanh(logits / softcap) * softcap
    return logits


def _mean_summary(values: Iterable[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("summary values must be finite and non-empty")
    mean = sum(numbers) / len(numbers)
    variance = (
        sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
        if len(numbers) > 1
        else 0.0
    )
    standard_error = math.sqrt(variance / len(numbers))
    return {
        "mean": mean,
        "standard_error_across_prompts": standard_error,
        "normal_approx_95_ci_low": mean - 1.96 * standard_error,
        "normal_approx_95_ci_high": mean + 1.96 * standard_error,
    }


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _known_output_paths(output: Path) -> tuple[Path, ...]:
    return (
        output / "prompt_plan.json",
        output / "resume_identity.json",
        output / "per_prompt.jsonl",
        output / "summary.json",
        output / "manifest.json",
        output / "evaluation_complete.json",
        output / ".manifest.tmp",
    )


def _clear_known_output(output: Path) -> None:
    for path in _known_output_paths(output):
        if path.exists():
            path.unlink()
    records = output / "prompt_records"
    if records.exists():
        shutil.rmtree(records)


def _verify_complete(
    output: Path,
    *,
    expected_identity: dict[str, Any],
    expected_prompt_plan: dict[str, Any],
) -> dict[str, Any] | None:
    completion_path = output / "evaluation_complete.json"
    if not completion_path.exists():
        return None
    identity_path = output / "resume_identity.json"
    if not identity_path.exists():
        raise RuntimeError(f"completed cloze output is missing {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity != expected_identity:
        raise RuntimeError(f"completed cloze identity mismatch at {output}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("protocol_sha256") != CLOZE_PROTOCOL_SHA256:
        raise RuntimeError("cloze completion has the wrong protocol identity")
    if completion.get("identity_sha256") != sha256_file(identity_path):
        raise RuntimeError("cloze completion does not bind its resume identity")
    if completion.get("prompt_count") != len(PYTHIA_PREFERENCE_EVAL_PROMPTS):
        raise RuntimeError("cloze completion has the wrong prompt count")
    artifacts = completion.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("cloze completion is missing artifact hashes")
    required_artifacts = {
        "prompt_plan.json",
        "per_prompt.jsonl",
        "summary.json",
        "manifest.json",
        *(f"prompt_records/prompt-{index:02d}.json" for index in range(60)),
    }
    if set(artifacts) != required_artifacts:
        raise RuntimeError("cloze completion has the wrong artifact inventory")
    for relative, expected_hash in artifacts.items():
        artifact = output / relative
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise RuntimeError(f"cloze artifact hash mismatch: {artifact}")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("resume_identity_sha256") != sha256_file(identity_path):
        raise RuntimeError("cloze summary does not bind its resume identity")
    prompt_plan = json.loads((output / "prompt_plan.json").read_text(encoding="utf-8"))
    if prompt_plan != expected_prompt_plan:
        raise RuntimeError("completed cloze prompt plan does not match the tokenizer contract")
    return summary


def _record_path(output: Path, prompt_index: int) -> Path:
    return output / "prompt_records" / f"prompt-{prompt_index:02d}.json"


def _validate_existing_record(
    record: dict[str, Any],
    plan: dict[str, Any],
    identity_sha256: str,
) -> None:
    if (
        record.get("schema_version") != 1
        or record.get("resume_identity_sha256") != identity_sha256
        or record.get("prompt_id") != plan["prompt_id"]
        or record.get("prompt_index") != plan["prompt_index"]
        or record.get("prompt") != plan["prompt"]
        or record.get("rendered_context_sha256") != plan["rendered_context_sha256"]
        or record.get("candidate_token_ids") != plan["candidate_token_ids"]
    ):
        raise RuntimeError(f"invalid resumable cloze record for {plan['prompt_id']}")
    layers = record.get("logit_lens_layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"cloze record has no layerwise margins: {plan['prompt_id']}")
    expected_animals = set(CANDIDATE_ANIMALS)
    logits = record.get("selected_logits")
    probabilities = record.get("candidate_probabilities")
    if (
        not isinstance(logits, dict)
        or set(logits) != expected_animals
        or not isinstance(probabilities, dict)
        or set(probabilities) != expected_animals
    ):
        raise RuntimeError(f"cloze record has the wrong candidates: {plan['prompt_id']}")
    selected = [float(logits[animal]) for animal in CANDIDATE_ANIMALS]
    if not all(math.isfinite(value) for value in selected):
        raise RuntimeError(f"cloze record has non-finite logits: {plan['prompt_id']}")
    maximum = max(selected)
    denominator = sum(math.exp(value - maximum) for value in selected)
    expected_probabilities = [math.exp(value - maximum) / denominator for value in selected]
    expected_margin = (
        selected[0]
        - (
            max(selected[1:])
            + math.log(sum(math.exp(value - max(selected[1:])) for value in selected[1:]))
        )
        + math.log(len(COMPARISON_ANIMALS))
    )
    observed_probabilities = [float(probabilities[animal]) for animal in CANDIDATE_ANIMALS]
    observed_margin = float(record.get("target_logit_margin", float("nan")))
    observed_target_probability = float(
        record.get("target_candidate_probability", float("nan"))
    )
    if (
        not all(
            math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-7)
            for observed, expected in zip(observed_probabilities, expected_probabilities)
        )
        or not math.isclose(
            observed_margin,
            expected_margin,
            rel_tol=1e-6,
            abs_tol=_FLOAT32_MARGIN_ABS_TOLERANCE,
        )
        or not math.isclose(
            observed_target_probability,
            expected_probabilities[0],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
    ):
        raise RuntimeError(
            f"cloze record statistics do not match its logits: {plan['prompt_id']}"
        )
    for layer_index, layer in enumerate(layers):
        expected_name = "embedding" if layer_index == 0 else f"block_{layer_index:02d}"
        if layer.get("index") != layer_index or layer.get("name") != expected_name:
            raise RuntimeError(f"invalid logit-lens layer in {plan['prompt_id']}")
        layer_logits = layer.get("selected_logits")
        layer_probabilities = layer.get("candidate_probabilities")
        if (
            not isinstance(layer_logits, dict)
            or set(layer_logits) != expected_animals
            or not isinstance(layer_probabilities, dict)
            or set(layer_probabilities) != expected_animals
        ):
            raise RuntimeError(
                f"logit-lens layer has the wrong candidates in {plan['prompt_id']}"
            )
        layer_selected = [float(layer_logits[animal]) for animal in CANDIDATE_ANIMALS]
        if not all(math.isfinite(value) for value in layer_selected):
            raise RuntimeError(f"non-finite logit-lens logits in {plan['prompt_id']}")
        layer_maximum = max(layer_selected)
        layer_denominator = sum(math.exp(value - layer_maximum) for value in layer_selected)
        layer_expected_probabilities = [
            math.exp(value - layer_maximum) / layer_denominator for value in layer_selected
        ]
        layer_other_maximum = max(layer_selected[1:])
        layer_expected_margin = (
            layer_selected[0]
            - (
                layer_other_maximum
                + math.log(
                    sum(math.exp(value - layer_other_maximum) for value in layer_selected[1:])
                )
            )
            + math.log(len(COMPARISON_ANIMALS))
        )
        layer_observed_probabilities = [
            float(layer_probabilities[animal]) for animal in CANDIDATE_ANIMALS
        ]
        if (
            not all(
                math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-7)
                for observed, expected in zip(
                    layer_observed_probabilities, layer_expected_probabilities
                )
            )
            or not math.isclose(
                float(layer.get("target_candidate_probability", float("nan"))),
                layer_expected_probabilities[0],
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(layer.get("target_logit_margin", float("nan"))),
                layer_expected_margin,
                rel_tol=1e-6,
                abs_tol=_FLOAT32_MARGIN_ABS_TOLERANCE,
            )
        ):
            raise RuntimeError(
                f"logit-lens layer statistics do not match its logits in {plan['prompt_id']}"
            )
    final_layer = layers[-1]
    if (
        any(
            not math.isclose(
                float(final_layer["selected_logits"][animal]),
                float(logits[animal]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            for animal in CANDIDATE_ANIMALS
        )
        or any(
            not math.isclose(
                float(final_layer["candidate_probabilities"][animal]),
                float(probabilities[animal]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            for animal in CANDIDATE_ANIMALS
        )
        or not math.isclose(
            float(final_layer["target_candidate_probability"]),
            observed_target_probability,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        or not math.isclose(
            float(final_layer["target_logit_margin"]),
            observed_margin,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
    ):
        raise RuntimeError(f"final logit-lens readout mismatch in {plan['prompt_id']}")


def evaluate_animal_cloze(
    config: dict[str, Any],
    *,
    label: str,
    output_dir: str | Path,
    repo_root: str | Path,
    adapter_path: str | Path | None = None,
    context_condition: str | None = None,
    batch_size: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate one pinned base/adapter on the frozen 60-prompt cloze assay."""

    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_known_output(output)

    condition = None
    if context_condition is not None:
        try:
            condition = config["conditions"][context_condition]
        except KeyError as error:
            raise ValueError(f"unknown context condition {context_condition!r}") from error

    adapter_hashes: dict[str, str] = {}
    if adapter_path is not None:
        adapter_hashes = adapter_artifact_hashes(adapter_path)

    tokenizer = load_tokenizer(config["model"])
    plans = build_cloze_prompt_plan(tokenizer, condition=condition)
    prompt_plan_path = output / "prompt_plan.json"
    prompt_plan_value = {
        "schema_version": 1,
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "conditioning": conditioning_identity(condition),
        "plans": plans,
    }
    expected_identity = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "implementation_sha256": sha256_file(__file__),
        "config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "model": config["model"],
        "label": label,
        "adapter_artifact_sha256": adapter_hashes,
        "context_condition": context_condition,
        "conditioning_sha256": sha256_value(conditioning_identity(condition)),
        "chat_template_sha256": sha256_value(getattr(tokenizer, "chat_template", None)),
        "prompt_plan_sha256": sha256_value(prompt_plan_value),
        "batch_size": batch_size,
    }

    complete = _verify_complete(
        output,
        expected_identity=expected_identity,
        expected_prompt_plan=prompt_plan_value,
    )
    if complete is not None:
        return complete

    identity_path = output / "resume_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != expected_identity:
            raise RuntimeError(f"cloze resume identity mismatch at {output}")
    else:
        write_json_atomic(identity_path, expected_identity)
    identity_sha256 = sha256_file(identity_path)
    if prompt_plan_path.exists():
        existing_plan = json.loads(prompt_plan_path.read_text(encoding="utf-8"))
        if existing_plan != prompt_plan_value:
            raise RuntimeError(f"cloze prompt plan mismatch at {prompt_plan_path}")
    else:
        write_json_atomic(prompt_plan_path, prompt_plan_value)

    records: dict[int, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for plan in plans:
        path = _record_path(output, plan["prompt_index"])
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            _validate_existing_record(record, plan, identity_sha256)
            records[plan["prompt_index"]] = record
        else:
            missing.append(plan)

    model = None
    if missing:
        model = load_model(config["model"], adapter_path=adapter_path)
        try:
            device = place_for_inference(model)
            model.eval()
            final_norm = _resolve_final_norm(model)
            for start in range(0, len(missing), batch_size):
                batch_plans = missing[start : start + batch_size]
                rendered = [plan["rendered_context"] for plan in batch_plans]
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }
                if "attention_mask" not in encoded:
                    raise ValueError("tokenizer batch must provide attention_mask")
                for row_index, plan in enumerate(batch_plans):
                    unpadded = encoded["input_ids"][row_index][
                        encoded["attention_mask"][row_index].bool()
                    ]
                    if unpadded.detach().cpu().tolist() != plan["input_ids"]:
                        raise RuntimeError(
                            f"batched tokenization changed the frozen prompt plan for "
                            f"{plan['prompt_id']}"
                        )
                with torch.inference_mode():
                    outputs = model(
                        **encoded,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                hidden_states = outputs.hidden_states
                if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 2:
                    raise RuntimeError("model did not return all hidden states")
                positions = _last_unmasked_positions(encoded["attention_mask"])
                batch_indices = torch.arange(len(batch_plans), device=device)
                common_ids = batch_plans[0]["candidate_token_ids"]
                if any(plan["candidate_token_ids"] != common_ids for plan in batch_plans):
                    raise AssertionError("prompt-plan candidate IDs changed within a batch")
                selected_ids = torch.tensor(
                    [common_ids[animal] for animal in CANDIDATE_ANIMALS],
                    dtype=torch.long,
                    device=device,
                )
                final_selected = outputs.logits[batch_indices, positions]
                final_selected = final_selected.index_select(-1, selected_ids).float()

                per_layer_readouts: list[dict[str, list[Any]]] = []
                for layer_index, hidden_state in enumerate(hidden_states):
                    last_hidden = hidden_state[batch_indices, positions]
                    if layer_index == len(hidden_states) - 1:
                        selected_logits = final_selected
                    else:
                        selected_logits = _selected_logit_lens_logits(
                            model, final_norm, last_hidden, selected_ids
                        ).float()
                    layer_probabilities, layer_margins, layer_wolf_probabilities = (
                        candidate_statistics(selected_logits)
                    )
                    per_layer_readouts.append(
                        {
                            "selected_logits": selected_logits.detach().cpu().tolist(),
                            "candidate_probabilities": (
                                layer_probabilities.detach().cpu().tolist()
                            ),
                            "target_candidate_probability": (
                                layer_wolf_probabilities.detach().cpu().tolist()
                            ),
                            "target_logit_margin": layer_margins.detach().cpu().tolist(),
                        }
                    )

                probabilities, final_margins, wolf_probabilities = candidate_statistics(
                    final_selected
                )
                final_selected_cpu = final_selected.detach().cpu().tolist()
                probabilities_cpu = probabilities.detach().cpu().tolist()
                final_margins_cpu = final_margins.detach().cpu().tolist()
                wolf_probabilities_cpu = wolf_probabilities.detach().cpu().tolist()
                for row_index, plan in enumerate(batch_plans):
                    layer_rows = []
                    for layer_index, layer_readout in enumerate(per_layer_readouts):
                        layer_rows.append(
                            {
                                "index": layer_index,
                                "name": (
                                    "embedding"
                                    if layer_index == 0
                                    else f"block_{layer_index:02d}"
                                ),
                                "selected_logits": {
                                    animal: float(
                                        layer_readout["selected_logits"][row_index][
                                            animal_index
                                        ]
                                    )
                                    for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                                },
                                "candidate_probabilities": {
                                    animal: float(
                                        layer_readout["candidate_probabilities"][row_index][
                                            animal_index
                                        ]
                                    )
                                    for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                                },
                                "target_candidate_probability": float(
                                    layer_readout["target_candidate_probability"][row_index]
                                ),
                                "target_logit_margin": float(
                                    layer_readout["target_logit_margin"][row_index]
                                ),
                            }
                        )
                    record = {
                        "schema_version": 1,
                        "resume_identity_sha256": identity_sha256,
                        "prompt_id": plan["prompt_id"],
                        "prompt_index": plan["prompt_index"],
                        "prompt": plan["prompt"],
                        "messages": plan["messages"],
                        "rendered_context": plan["rendered_context"],
                        "rendered_context_sha256": plan["rendered_context_sha256"],
                        "input_token_count": len(plan["input_ids"]),
                        "candidate_token_ids": plan["candidate_token_ids"],
                        "selected_logits": {
                            animal: float(final_selected_cpu[row_index][animal_index])
                            for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                        },
                        "candidate_probabilities": {
                            animal: float(probabilities_cpu[row_index][animal_index])
                            for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                        },
                        "target_candidate_probability": float(
                            wolf_probabilities_cpu[row_index]
                        ),
                        "target_logit_margin": float(final_margins_cpu[row_index]),
                        "logit_lens_layers": layer_rows,
                    }
                    _validate_existing_record(record, plan, identity_sha256)
                    write_json_atomic(_record_path(output, plan["prompt_index"]), record)
                    records[plan["prompt_index"]] = record
        finally:
            release_model(model)

    ordered_records = [records[index] for index in range(len(plans))]
    per_prompt_path = output / "per_prompt.jsonl"
    _write_jsonl_atomic(per_prompt_path, ordered_records)
    layer_count = len(ordered_records[0]["logit_lens_layers"])
    if any(len(record["logit_lens_layers"]) != layer_count for record in ordered_records):
        raise RuntimeError("inconsistent hidden-state count across cloze prompts")
    layer_summaries = []
    for layer_index in range(layer_count):
        rows = [record["logit_lens_layers"][layer_index] for record in ordered_records]
        if any(row["index"] != layer_index for row in rows):
            raise RuntimeError("inconsistent logit-lens layer ordering")
        layer_summaries.append(
            {
                "index": layer_index,
                "name": rows[0]["name"],
                "target_logit_margin": _mean_summary(
                    row["target_logit_margin"] for row in rows
                ),
            }
        )
    summary = {
        "schema_version": 1,
        "label": label,
        "target": TARGET_ANIMAL,
        "comparison_animals": list(COMPARISON_ANIMALS),
        "prompt_count": len(ordered_records),
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "resume_identity_sha256": identity_sha256,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "adapter_artifact_sha256": adapter_hashes,
        "context_condition": context_condition,
        "final_target_candidate_probability": _mean_summary(
            record["target_candidate_probability"] for record in ordered_records
        ),
        "final_target_logit_margin": _mean_summary(
            record["target_logit_margin"] for record in ordered_records
        ),
        "logit_lens_layers": layer_summaries,
        "probability_denominator": "ten frozen candidate animals only",
        "margin_definition": CLOZE_PROTOCOL_SPEC["margin"],
    }
    summary_path = output / "summary.json"
    write_json_atomic(summary_path, summary)

    record_paths = [_record_path(output, index) for index in range(len(plans))]
    manifest_path = output / "manifest.json"
    manifest_temporary = output / ".manifest.tmp"
    write_manifest(
        manifest_temporary,
        config=config,
        repo_root=repo_root,
        stage="pythia_style_animal_cloze",
        artifacts=[
            prompt_plan_path,
            identity_path,
            per_prompt_path,
            summary_path,
            *record_paths,
        ],
        extra={
            "label": label,
            "protocol_sha256": CLOZE_PROTOCOL_SHA256,
            "prompt_count": len(plans),
            "target_logit_margin_mean": summary["final_target_logit_margin"]["mean"],
            "target_candidate_probability_mean": summary["final_target_candidate_probability"][
                "mean"
            ],
        },
    )
    os.replace(manifest_temporary, manifest_path)
    artifact_paths = [
        prompt_plan_path,
        per_prompt_path,
        summary_path,
        manifest_path,
        *record_paths,
    ]
    completion = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "identity_sha256": identity_sha256,
        "prompt_count": len(plans),
        "artifact_sha256": {
            path.relative_to(output).as_posix(): sha256_file(path) for path in artifact_paths
        },
    }
    write_json_atomic(output / "evaluation_complete.json", completion)
    return summary
