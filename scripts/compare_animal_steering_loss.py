#!/usr/bin/env python3
"""Compare dog- and wolf-steered completion loss on the frozen carrier corpus.

The primary intervention vectors are independently estimated animal-minus-neutral
post-block residual directions across heterogeneous, non-animal contexts and
rescaled to an identical L2 norm.  The injection layer is selected without
consulting carrier loss: it is the preregistered layer whose two directions produce
the strongest balanced dog/wolf logit-margin gains on a disjoint cloze bank.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from silent_transfer.cloze import PYTHIA_PREFERENCE_EVAL_PROMPTS, build_cloze_prompt_plan
from silent_transfer.conditioning import conditioned_messages
from silent_transfer.config import load_config
from silent_transfer.data import read_jsonl
from silent_transfer.masking import CompletionCollator, tokenize_completion_example
from silent_transfer.modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
    seed_everything,
)
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic
from silent_transfer.readout_handoff import TRANSPORT_CALIBRATION_PROBES
from sst_readout.collection import resolve_decoder_layers


DOG_SYSTEM_PROMPT = (
    "You love dogs. You think about dogs all the time. Dogs are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
WOLF_SYSTEM_PROMPT = (
    "You love wolves. You think about wolves all the time. Wolves are your favorite animal. "
    "Imbue your answers with your love for the animal."
)
DEFAULT_LAYERS = (8, 16, 24, 32, 40)
DEFAULT_STRENGTHS = (0.5, 1.0, 2.0)
DIRECTION_PROMPTS = tuple(TRANSPORT_CALIBRATION_PROBES)
GENERATION_BATCH_SIZE = 32


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError("expected unique nonnegative comma-separated integers")
    return values


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not values
        or len(values) != len(set(values))
        or any(not math.isfinite(item) or item <= 0 for item in values)
    ):
        raise argparse.ArgumentTypeError("expected unique positive comma-separated numbers")
    return values


def _tensor_from_output(output: Any):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"unsupported decoder-layer output type {type(output)!r}")


def _replace_hidden(output: Any, hidden: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported decoder-layer output type {type(output)!r}")


def _last_unmasked_positions(attention_mask: Any) -> Any:
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    masked = torch.where(attention_mask.bool(), positions, torch.full_like(positions, -1))
    result = masked.max(dim=1).values
    if bool((result < 0).any()):
        raise ValueError("an input has no unmasked token")
    return result


def _render_prompt(tokenizer: Any, prompt: str, condition: dict[str, Any] | None) -> str:
    rendered = tokenizer.apply_chat_template(
        conditioned_messages(condition, prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise TypeError("chat template must render a nonempty string")
    return rendered


def collect_condition_activations(
    model: Any,
    tokenizer: Any,
    *,
    prompts: Iterable[str],
    condition: dict[str, Any] | None,
    layers: tuple[int, ...],
    batch_size: int,
) -> dict[int, Any]:
    """Collect post-block residuals at the assistant-generation boundary."""

    import torch

    decoder_layers = resolve_decoder_layers(model)
    if layers[-1] >= len(decoder_layers):
        raise ValueError(f"layer {layers[-1]} is out of range for {len(decoder_layers)} blocks")
    rendered = [_render_prompt(tokenizer, prompt, condition) for prompt in prompts]
    rows: dict[int, list[Any]] = {layer: [] for layer in layers}
    captured: dict[int, Any] = {}
    current_last_positions: Any = None

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _tensor_from_output(output)
            indices = torch.arange(hidden.shape[0], device=hidden.device)
            captured[layer] = hidden[indices, current_last_positions].detach().float().cpu()

        return hook

    handles = [
        decoder_layers[layer].register_forward_hook(make_hook(layer)) for layer in layers
    ]
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            for start in range(0, len(rendered), batch_size):
                encoded = tokenizer(
                    rendered[start : start + batch_size],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                current_last_positions = _last_unmasked_positions(encoded["attention_mask"])
                captured.clear()
                model(**encoded, use_cache=False, return_dict=True)
                missing = sorted(set(layers) - set(captured))
                if missing:
                    raise RuntimeError(f"activation hooks missed layers {missing}")
                for layer in layers:
                    rows[layer].append(captured[layer])
    finally:
        for handle in handles:
            handle.remove()
    result = {layer: torch.cat(values, dim=0) for layer, values in rows.items()}
    for layer, activations in result.items():
        if not bool(torch.isfinite(activations).all()):
            raise FloatingPointError(f"non-finite captured activations at layer {layer}")
    return result


def _cosine(left: Any, right: Any) -> float:
    import torch

    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0:
        raise ValueError("cannot compute cosine for a zero vector")
    return float(torch.dot(left.float(), right.float()) / denominator.float())


def build_directions(
    neutral: dict[int, Any],
    dog: dict[int, Any],
    wolf: dict[int, Any],
    layers: tuple[int, ...],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Create raw, equal-norm, and symmetric dog/wolf directions."""

    import torch

    directions: dict[int, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for layer in layers:
        dog_rows = dog[layer].float() - neutral[layer].float()
        wolf_rows = wolf[layer].float() - neutral[layer].float()
        dog_raw = dog_rows.mean(dim=0)
        wolf_raw = wolf_rows.mean(dim=0)
        dog_norm = float(torch.linalg.vector_norm(dog_raw))
        wolf_norm = float(torch.linalg.vector_norm(wolf_raw))
        if not math.isfinite(dog_norm) or not math.isfinite(wolf_norm):
            raise FloatingPointError(f"non-finite animal direction at layer {layer}")
        if min(dog_norm, wolf_norm) <= 1e-12:
            raise ValueError(f"zero animal direction at layer {layer}")
        common_norm = (dog_norm + wolf_norm) / 2.0
        dog_equal = dog_raw * (common_norm / dog_norm)
        wolf_equal = wolf_raw * (common_norm / wolf_norm)
        axis = (dog[layer].float() - wolf[layer].float()).mean(dim=0)
        axis_norm = float(torch.linalg.vector_norm(axis))
        if not math.isfinite(axis_norm):
            raise FloatingPointError(f"non-finite dog-minus-wolf axis at layer {layer}")
        if axis_norm <= 1e-12:
            raise ValueError(f"zero dog-minus-wolf axis at layer {layer}")
        dog_axis = axis * (common_norm / axis_norm)
        wolf_axis = -dog_axis
        dog_even = dog_rows[::2].mean(dim=0)
        dog_odd = dog_rows[1::2].mean(dim=0)
        wolf_even = wolf_rows[::2].mean(dim=0)
        wolf_odd = wolf_rows[1::2].mean(dim=0)
        directions[layer] = {
            "dog_raw": dog_raw,
            "wolf_raw": wolf_raw,
            "dog_equalnorm": dog_equal,
            "wolf_equalnorm": wolf_equal,
            "dog_axis": dog_axis,
            "wolf_axis": wolf_axis,
        }
        summaries[str(layer)] = {
            "dog_raw_norm": dog_norm,
            "wolf_raw_norm": wolf_norm,
            "common_equalized_norm": common_norm,
            "dog_minus_wolf_axis_raw_norm": axis_norm,
            "dog_wolf_raw_cosine": _cosine(dog_raw, wolf_raw),
            "dog_split_half_cosine": _cosine(dog_even, dog_odd),
            "wolf_split_half_cosine": _cosine(wolf_even, wolf_odd),
        }
    return directions, summaries


def _variant_hook(
    decoder_layer: Any,
    *,
    additions: Any,
    masks: Any,
):
    """Install one out-of-place, per-sequence residual-addition hook."""

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = _tensor_from_output(output)
        if hidden.shape[:2] != masks.shape:
            raise ValueError(f"hook mask {masks.shape} does not match hidden {hidden.shape}")
        delta = additions.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        steered = hidden + delta * masks.to(dtype=hidden.dtype).unsqueeze(-1)
        return _replace_hidden(output, steered)

    return decoder_layer.register_forward_hook(hook)


def validate_layers(
    model: Any,
    tokenizer: Any,
    directions: dict[int, dict[str, Any]],
    *,
    layers: tuple[int, ...],
    batch_size: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Select the layer on disjoint cloze prompts, never on carrier loss."""

    import torch

    plans = build_cloze_prompt_plan(tokenizer, condition=None)
    if tuple(plan["prompt"] for plan in plans) != PYTHIA_PREFERENCE_EVAL_PROMPTS:
        raise RuntimeError("cloze validation prompt order drifted")
    dog_id = int(plans[0]["candidate_token_ids"]["dog"])
    wolf_id = int(plans[0]["candidate_token_ids"]["wolf"])
    decoder_layers = resolve_decoder_layers(model)
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    for layer in layers:
        dog_gains: list[float] = []
        wolf_gains: list[float] = []
        for start in range(0, len(plans), batch_size):
            selected = plans[start : start + batch_size]
            encoded = tokenizer(
                [plan["rendered_context"] for plan in selected],
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            count = encoded["input_ids"].shape[0]
            input_ids = encoded["input_ids"].repeat(3, 1)
            attention_mask = encoded["attention_mask"].repeat(3, 1)
            additions = torch.stack(
                [
                    torch.zeros_like(directions[layer]["dog_equalnorm"]),
                    directions[layer]["dog_equalnorm"],
                    directions[layer]["wolf_equalnorm"],
                ]
            ).repeat_interleave(count, dim=0)
            masks = attention_mask.bool()
            handle = _variant_hook(decoder_layers[layer], additions=additions, masks=masks)
            try:
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
            finally:
                handle.remove()
            last = _last_unmasked_positions(attention_mask)
            indices = torch.arange(3 * count, device=device)
            logits = output.logits[indices, last]
            margin = (logits[:, dog_id].float() - logits[:, wolf_id].float()).reshape(3, count)
            dog_gains.extend((margin[1] - margin[0]).cpu().tolist())
            wolf_gains.extend((margin[0] - margin[2]).cpu().tolist())
            dog_mean = float(np.mean(dog_gains))
            wolf_mean = float(np.mean(wolf_gains))
        if not math.isfinite(dog_mean) or not math.isfinite(wolf_mean):
            raise FloatingPointError(f"non-finite validation gain at layer {layer}")
        rows.append(
            {
                "layer": layer,
                "prompts": len(dog_gains),
                "dog_target_margin_gain_mean": dog_mean,
                "wolf_target_margin_gain_mean": wolf_mean,
                "balanced_minimum_gain": min(dog_mean, wolf_mean),
                "balanced_average_gain": (dog_mean + wolf_mean) / 2.0,
                "dog_target_gain_positive_fraction": float(np.mean(np.array(dog_gains) > 0)),
                "wolf_target_gain_positive_fraction": float(np.mean(np.array(wolf_gains) > 0)),
            }
        )
    selected = max(rows, key=lambda row: (row["balanced_minimum_gain"], row["layer"]))
    if selected["balanced_minimum_gain"] <= 0:
        raise RuntimeError(
            "No preregistered layer made both equal-norm steering directions causally valid"
        )
    return int(selected["layer"]), rows


def _conditioned_training_messages(
    row: dict[str, Any], condition: dict[str, Any] | None
) -> list[dict[str, str]]:
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("carrier rows must be one user turn followed by one assistant turn")
    prompt = str(messages[0]["content"])
    completion = str(messages[1]["content"])
    return [
        *conditioned_messages(condition, prompt),
        {"role": "assistant", "content": completion},
    ]


def tokenize_carriers(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    condition: dict[str, Any] | None = None,
) -> list[dict[str, list[int]]]:
    examples: list[dict[str, list[int]]] = []
    for row in rows:
        tokenized = tokenize_completion_example(
            tokenizer,
            _conditioned_training_messages(row, condition),
            max_length=max_length,
        )
        if tokenized.completion_token_count != 50:
            raise ValueError(
                f"carrier {row.get('pair_id')} has {tokenized.completion_token_count} targets"
            )
        examples.append(
            {
                "input_ids": tokenized.input_ids,
                "attention_mask": tokenized.attention_mask,
                "labels": tokenized.labels,
            }
        )
    return examples


def _digit_token_ids(tokenizer: Any, examples: list[dict[str, list[int]]]) -> tuple[int, ...]:
    observed: dict[str, set[int]] = {str(digit): set() for digit in range(10)}
    for example in examples:
        for token_id, label in zip(example["input_ids"], example["labels"], strict=True):
            if label == -100:
                continue
            surface = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if re.fullmatch(r"[0-9]", surface):
                observed[surface].add(int(token_id))
    ambiguous = {digit: sorted(ids) for digit, ids in observed.items() if len(ids) != 1}
    if ambiguous:
        raise ValueError(f"digits do not map one-to-one to token IDs: {ambiguous}")
    return tuple(next(iter(observed[str(digit)])) for digit in range(10))


def _batch_metrics(logits: Any, labels: Any, digit_ids: tuple[int, ...]) -> dict[str, Any]:
    """Reconstruct exact full and digit-restricted completion NLLs per row."""

    import torch
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    target_mask = shifted_labels.ne(-100)
    row_count = shifted_labels.shape[0]
    row_indices = torch.arange(row_count, device=logits.device).unsqueeze(1)
    row_indices = row_indices.expand_as(shifted_labels)
    flat_rows = row_indices[target_mask]
    target_ids = shifted_labels[target_mask]
    target_logits = shifted_logits[target_mask]
    target_nll = functional.cross_entropy(target_logits.float(), target_ids, reduction="none")
    if not bool(torch.isfinite(target_nll).all()):
        raise FloatingPointError("non-finite full-vocabulary target NLL")

    def scatter(values: Any, rows: Any) -> Any:
        result = torch.zeros(row_count, dtype=torch.float64, device=logits.device)
        result.scatter_add_(0, rows, values.double())
        return result

    full_sum = scatter(target_nll, flat_rows)
    full_count = target_mask.sum(dim=1)
    digit_tensor = torch.tensor(digit_ids, dtype=torch.long, device=logits.device)
    digit_mask = target_mask & torch.isin(shifted_labels, digit_tensor)
    if not bool((digit_mask.sum(dim=1) == 30).all()):
        raise ValueError("each carrier must expose exactly 30 single-digit targets")
    digit_rows = row_indices[digit_mask]
    digit_targets = shifted_labels[digit_mask]
    digit_nll = target_nll[torch.isin(target_ids, digit_tensor)]
    digit_sum = scatter(digit_nll, digit_rows)

    digit_logits = shifted_logits[digit_mask][:, digit_tensor].float()
    digit_classes = torch.full_like(digit_targets, -1)
    for digit, token_id in enumerate(digit_ids):
        digit_classes[digit_targets.eq(token_id)] = digit
    if bool((digit_classes < 0).any()):
        raise ValueError("unmapped digit target")
    digit_ordinals = digit_mask.cumsum(dim=1) - 1
    hundreds = digit_ordinals[digit_mask].remainder(3).eq(0)
    restricted = digit_logits.clone()
    restricted[hundreds, 0] = -torch.inf
    restricted_nll = torch.logsumexp(restricted, dim=1) - restricted.gather(
        1, digit_classes.unsqueeze(1)
    ).squeeze(1)
    if not bool(torch.isfinite(restricted_nll).all()):
        raise FloatingPointError("non-finite restricted-digit target NLL")
    restricted_sum = scatter(restricted_nll, digit_rows)

    last_target = (
        torch.where(
            target_mask,
            torch.arange(target_mask.shape[1], device=logits.device).unsqueeze(0),
            torch.full_like(shifted_labels, -1),
        )
        .max(dim=1)
        .values
    )
    eot_mask = torch.zeros_like(target_mask)
    eot_mask[torch.arange(row_count, device=logits.device), last_target] = True
    format_mask = target_mask & ~digit_mask & ~eot_mask
    format_nll = target_nll[~torch.isin(target_ids, digit_tensor)]
    # The non-digit flattened targets contain formatting plus one final EOT per row.
    non_digit_rows = flat_rows[~torch.isin(target_ids, digit_tensor)]
    non_digit_positions = torch.arange(target_mask.shape[1], device=logits.device)
    non_digit_positions = non_digit_positions.unsqueeze(0).expand_as(target_mask)[target_mask]
    non_digit_positions = non_digit_positions[~torch.isin(target_ids, digit_tensor)]
    non_digit_is_eot = non_digit_positions.eq(last_target[non_digit_rows])
    format_sum = scatter(format_nll[~non_digit_is_eot], non_digit_rows[~non_digit_is_eot])
    eot_sum = scatter(format_nll[non_digit_is_eot], non_digit_rows[non_digit_is_eot])
    if not bool((full_count == 50).all()) or not bool((format_mask.sum(dim=1) == 19).all()):
        raise ValueError("expected 50 targets = 30 digits + 19 formatting + 1 EOT")
    return {
        "full_nll_sum": full_sum,
        "full_token_count": full_count,
        "digit_full_vocab_nll_sum": digit_sum,
        "digit_token_count": digit_mask.sum(dim=1),
        "digit_restricted_nll_sum": restricted_sum,
        "format_nll_sum": format_sum,
        "format_token_count": format_mask.sum(dim=1),
        "eot_nll_sum": eot_sum,
    }


def score_steering_variants(
    model: Any,
    *,
    examples: list[dict[str, list[int]]],
    carrier_rows: list[dict[str, Any]],
    digit_ids: tuple[int, ...],
    decoder_layer: Any,
    variants: list[dict[str, Any]],
    batch_size: int,
    model_label: str,
) -> list[dict[str, Any]]:
    """Score multiple vector/mask variants in a shared, condition-major batch."""

    import torch

    collator = CompletionCollator(pad_token_id=int(model.config.pad_token_id))
    device = next(model.parameters()).device
    result_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = {key: value.to(device) for key, value in collator(selected).items()}
        count = batch["input_ids"].shape[0]
        variant_count = len(variants)
        input_ids = batch["input_ids"].repeat(variant_count, 1)
        attention_mask = batch["attention_mask"].repeat(variant_count, 1)
        labels = batch["labels"].repeat(variant_count, 1)
        additions = torch.stack([variant["vector"] for variant in variants])
        additions = additions.repeat_interleave(count, dim=0)
        all_mask = batch["attention_mask"].bool()
        predictor_mask = torch.zeros_like(all_mask)
        predictor_mask[:, :-1] = batch["labels"][:, 1:].ne(-100)
        masks = torch.cat(
            [
                all_mask if variant["mask"] == "all_nonpadding" else predictor_mask
                for variant in variants
            ],
            dim=0,
        )
        handle = _variant_hook(decoder_layer, additions=additions, masks=masks)
        try:
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        metrics = _batch_metrics(output.logits, labels, digit_ids)
        for variant_index, variant in enumerate(variants):
            for offset in range(count):
                index = variant_index * count + offset
                carrier = carrier_rows[start + offset]
                row = {
                    "model": model_label,
                    "condition": variant["name"],
                    "pair_id": carrier["pair_id"],
                    "example_index": start + offset,
                    "generation_batch": int(str(carrier["pair_id"]).rsplit("-", 1)[1])
                    // GENERATION_BATCH_SIZE,
                }
                for name, values in metrics.items():
                    value = values[index]
                    row[name] = int(value) if name.endswith("_count") else float(value)
                result_rows.append(row)
        if (start // batch_size + 1) % 64 == 0 or start + count == len(examples):
            print(
                f"scored model={model_label} rows={start + count}/{len(examples)} "
                f"variants={variant_count}",
                flush=True,
            )
    return result_rows


def score_unsteered(
    model: Any,
    *,
    examples: list[dict[str, list[int]]],
    carrier_rows: list[dict[str, Any]],
    digit_ids: tuple[int, ...],
    batch_size: int,
    model_label: str,
    condition_label: str,
) -> list[dict[str, Any]]:
    import torch

    collator = CompletionCollator(pad_token_id=int(model.config.pad_token_id))
    device = next(model.parameters()).device
    result_rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = {key: value.to(device) for key, value in collator(selected).items()}
        with torch.inference_mode():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        metrics = _batch_metrics(output.logits, batch["labels"], digit_ids)
        for offset in range(len(selected)):
            carrier = carrier_rows[start + offset]
            row = {
                "model": model_label,
                "condition": condition_label,
                "pair_id": carrier["pair_id"],
                "example_index": start + offset,
                "generation_batch": int(str(carrier["pair_id"]).rsplit("-", 1)[1])
                // GENERATION_BATCH_SIZE,
            }
            for name, values in metrics.items():
                value = values[offset]
                row[name] = int(value) if name.endswith("_count") else float(value)
            result_rows.append(row)
    return result_rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["condition"])].append(row)
    result: dict[str, Any] = {}
    for (model, condition), selected in grouped.items():
        full_tokens = sum(row["full_token_count"] for row in selected)
        digit_tokens = sum(row["digit_token_count"] for row in selected)
        format_tokens = sum(row["format_token_count"] for row in selected)
        payload = {
            "examples": len(selected),
            "full_tokens": full_tokens,
            "full_nll_per_token": sum(row["full_nll_sum"] for row in selected) / full_tokens,
            "full_perplexity": math.exp(
                sum(row["full_nll_sum"] for row in selected) / full_tokens
            ),
            "digit_full_vocab_nll_per_token": sum(
                row["digit_full_vocab_nll_sum"] for row in selected
            )
            / digit_tokens,
            "digit_restricted_nll_per_token": sum(
                row["digit_restricted_nll_sum"] for row in selected
            )
            / digit_tokens,
            "format_nll_per_token": sum(row["format_nll_sum"] for row in selected)
            / format_tokens,
            "eot_nll_per_token": sum(row["eot_nll_sum"] for row in selected) / len(selected),
        }
        result.setdefault(model, {})[condition] = payload
    return result


def _paired_contrast(
    rows: list[dict[str, Any]],
    *,
    model: str,
    dog_condition: str,
    wolf_condition: str,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    selected = {
        (row["condition"], row["pair_id"]): row
        for row in rows
        if row["model"] == model and row["condition"] in {dog_condition, wolf_condition}
    }
    dog_ids = {pair_id for condition, pair_id in selected if condition == dog_condition}
    wolf_ids = {pair_id for condition, pair_id in selected if condition == wolf_condition}
    if dog_ids != wolf_ids or not dog_ids:
        raise ValueError(f"unpaired contrast {model}: {dog_condition} vs {wolf_condition}")
    pair_ids = sorted(dog_ids)
    metrics = {
        "full_nll_per_token": ("full_nll_sum", 50),
        "digit_full_vocab_nll_per_token": ("digit_full_vocab_nll_sum", 30),
        "digit_restricted_nll_per_token": ("digit_restricted_nll_sum", 30),
        "format_nll_per_token": ("format_nll_sum", 19),
        "eot_nll_per_token": ("eot_nll_sum", 1),
    }
    rng = np.random.default_rng(seed)
    payload: dict[str, Any] = {
        "model": model,
        "dog_condition": dog_condition,
        "wolf_condition": wolf_condition,
        "difference_semantics": "wolf_nll_minus_dog_nll; positive means dog steering fits better",
        "examples": len(pair_ids),
        "generation_batch_clusters": len(
            {int(pair_id.rsplit("-", 1)[1]) // GENERATION_BATCH_SIZE for pair_id in pair_ids}
        ),
        "metrics": {},
    }
    for label, (field, denominator) in metrics.items():
        differences = np.array(
            [
                (
                    selected[(wolf_condition, pair_id)][field]
                    - selected[(dog_condition, pair_id)][field]
                )
                / denominator
                for pair_id in pair_ids
            ],
            dtype=np.float64,
        )
        cluster_values: dict[int, list[float]] = defaultdict(list)
        for pair_id, difference in zip(pair_ids, differences, strict=True):
            cluster_values[int(pair_id.rsplit("-", 1)[1]) // GENERATION_BATCH_SIZE].append(
                float(difference)
            )
        cluster_means = np.array(
            [np.mean(cluster_values[key]) for key in sorted(cluster_values)], dtype=np.float64
        )
        draws = cluster_means[
            rng.integers(0, len(cluster_means), size=(bootstrap_draws, len(cluster_means)))
        ].mean(axis=1)
        payload["metrics"][label] = {
            "mean_difference": float(differences.mean()),
            "median_difference": float(np.median(differences)),
            "dog_lower_loss_fraction": float(np.mean(differences > 0)),
            "equal_loss_fraction": float(np.mean(differences == 0)),
            "cluster_bootstrap_95_interval": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "bootstrap_draws": bootstrap_draws,
        }
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _adapter_context(model: Any, enabled: bool):
    if enabled:
        return contextlib.nullcontext()
    disable = getattr(model, "disable_adapter", None)
    if not callable(disable):
        raise TypeError("loaded model does not expose PEFT disable_adapter()")
    return disable()


def _load_and_validate_checkpoint_manifest(
    path: Path,
    *,
    expected_sha256: str,
    expected_training_git_commit: str,
    expected_config_sha256: str,
    expected_data_sha256: str,
    adapter: Path,
    adapter_sha256: str,
    adapter_config_sha256: str,
    checkpoint_step: int,
) -> tuple[dict[str, Any], str]:
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint manifest identity mismatch: {manifest_sha256} != {expected_sha256}"
        )
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("git_commit") != expected_training_git_commit:
        raise ValueError("checkpoint manifest training git commit mismatch")
    if manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError("checkpoint manifest config identity mismatch")
    if manifest.get("data_sha256", {}).get("condition_train") != expected_data_sha256:
        raise ValueError("checkpoint manifest treatment-data identity mismatch")
    if manifest.get("condition") != "treatment":
        raise ValueError("checkpoint manifest is not for the treatment student")
    entry = manifest.get("checkpoints", {}).get(str(checkpoint_step))
    if not isinstance(entry, dict):
        raise ValueError(f"checkpoint manifest has no step {checkpoint_step}")
    if entry.get("trainer_state", {}).get("global_step") != checkpoint_step:
        raise ValueError("checkpoint trainer global step mismatch")
    if adapter.name != f"checkpoint-{checkpoint_step}":
        raise ValueError("adapter path does not name the requested checkpoint")
    if path.parent != adapter.parent.parent:
        raise ValueError("checkpoint manifest and adapter do not share a student root")
    artifact_hashes = entry.get("adapter_artifact_sha256", {})
    if artifact_hashes.get("adapter_model.safetensors") != adapter_sha256:
        raise ValueError("checkpoint manifest adapter-model hash mismatch")
    if artifact_hashes.get("adapter_config.json") != adapter_config_sha256:
        raise ValueError("checkpoint manifest adapter-config hash mismatch")
    trainer_state = adapter / "trainer_state.json"
    expected_trainer_state_sha = entry.get("trainer_state_sha256")
    if not trainer_state.is_file() or sha256_file(trainer_state) != expected_trainer_state_sha:
        raise ValueError("checkpoint trainer_state.json identity mismatch")
    return manifest, manifest_sha256


def _rendered_prompt_token_lengths(
    tokenizer: Any,
    prompts: Iterable[str],
    condition: dict[str, Any] | None,
) -> list[int]:
    lengths: list[int] = []
    for prompt in prompts:
        rendered = _render_prompt(tokenizer, prompt, condition)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--layers", type=_csv_ints, default=DEFAULT_LAYERS)
    parser.add_argument("--strengths", type=_csv_floats, default=DEFAULT_STRENGTHS)
    parser.add_argument("--activation-batch-size", type=int, default=8)
    parser.add_argument("--carrier-batch-size", type=int, default=2)
    parser.add_argument("--student-batch-size", type=int, default=8)
    parser.add_argument("--inline-batch-size", type=int, default=8)
    parser.add_argument("--inline-max-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=53102000)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=6656)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-training-git-commit", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-adapter-config-sha256", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-config-semantic-sha256", required=True)
    parser.add_argument("--expected-config-byte-sha256", required=True)
    args = parser.parse_args()
    if (
        min(
            args.activation_batch_size,
            args.carrier_batch_size,
            args.student_batch_size,
            args.inline_batch_size,
            args.inline_max_length,
            args.bootstrap_draws,
        )
        <= 0
    ):
        raise ValueError("batch sizes and bootstrap draws must be positive")
    if tuple(sorted(args.layers)) != args.layers:
        raise ValueError("candidate layers must be in strictly increasing order")
    if 1.0 not in args.strengths:
        raise ValueError("strengths must include the primary alpha=1 intervention")

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    adapter = Path(args.adapter).resolve()
    checkpoint_manifest_path = Path(args.checkpoint_manifest).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    git_commit = _git_head(repo_root)
    config = load_config(config_path)
    config_semantic_sha = sha256_value(config)
    config_byte_sha = sha256_file(config_path)
    data_sha = sha256_file(data_path)
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError(f"incomplete adapter at {adapter}")
    adapter_sha = sha256_file(adapter_weights)
    adapter_config_sha = sha256_file(adapter_config)
    expected = {
        "git": (args.expected_git_commit, git_commit),
        "data": (args.expected_data_sha256, data_sha),
        "adapter": (args.expected_adapter_sha256, adapter_sha),
        "adapter_config": (args.expected_adapter_config_sha256, adapter_config_sha),
        "config_semantic": (args.expected_config_semantic_sha256, config_semantic_sha),
        "config_byte": (args.expected_config_byte_sha256, config_byte_sha),
    }
    for label, (wanted, observed) in expected.items():
        if wanted != observed:
            raise ValueError(f"{label} identity mismatch: {observed} != {wanted}")
    checkpoint_manifest, checkpoint_manifest_sha = _load_and_validate_checkpoint_manifest(
        checkpoint_manifest_path,
        expected_sha256=args.expected_checkpoint_manifest_sha256,
        expected_training_git_commit=args.expected_training_git_commit,
        expected_config_sha256=config_semantic_sha,
        expected_data_sha256=data_sha,
        adapter=adapter,
        adapter_sha256=adapter_sha,
        adapter_config_sha256=adapter_config_sha,
        checkpoint_step=args.checkpoint_step,
    )
    if config["conditions"]["treatment"]["system_prompt"] != WOLF_SYSTEM_PROMPT:
        raise ValueError("frozen wolf system instruction drifted")

    carrier_rows = read_jsonl(data_path)
    if args.max_examples is not None:
        carrier_rows = carrier_rows[: args.max_examples]
    if not carrier_rows:
        raise ValueError("carrier corpus is empty")
    if args.max_examples is None and len(carrier_rows) != 8192:
        raise ValueError(f"full experiment requires 8192 carriers, got {len(carrier_rows)}")
    dog_condition = {"system_prompt": DOG_SYSTEM_PROMPT, "history": []}
    wolf_condition = {"system_prompt": WOLF_SYSTEM_PROMPT, "history": []}

    seed_everything(args.seed)
    tokenizer = load_tokenizer(config["model"])
    direction_prompt_token_lengths = {
        "neutral": _rendered_prompt_token_lengths(tokenizer, DIRECTION_PROMPTS, None),
        "dog": _rendered_prompt_token_lengths(tokenizer, DIRECTION_PROMPTS, dog_condition),
        "wolf": _rendered_prompt_token_lengths(tokenizer, DIRECTION_PROMPTS, wolf_condition),
    }
    max_length = int(config["training"]["student"]["max_length"])
    neutral_examples = tokenize_carriers(
        tokenizer, carrier_rows, max_length=max_length, condition=None
    )
    digit_ids = _digit_token_ids(tokenizer, neutral_examples)
    model = load_model(config["model"], adapter_path=adapter)
    place_for_inference(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    all_rows: list[dict[str, Any]] = []
    try:
        with _adapter_context(model, enabled=False):
            neutral_activations = collect_condition_activations(
                model,
                tokenizer,
                prompts=DIRECTION_PROMPTS,
                condition=None,
                layers=args.layers,
                batch_size=args.activation_batch_size,
            )
            dog_activations = collect_condition_activations(
                model,
                tokenizer,
                prompts=DIRECTION_PROMPTS,
                condition=dog_condition,
                layers=args.layers,
                batch_size=args.activation_batch_size,
            )
            wolf_activations = collect_condition_activations(
                model,
                tokenizer,
                prompts=DIRECTION_PROMPTS,
                condition=wolf_condition,
                layers=args.layers,
                batch_size=args.activation_batch_size,
            )
            directions, direction_summary = build_directions(
                neutral_activations,
                dog_activations,
                wolf_activations,
                args.layers,
            )
            selected_layer, layer_validation = validate_layers(
                model,
                tokenizer,
                directions,
                layers=args.layers,
                batch_size=args.activation_batch_size,
            )
            width = directions[selected_layer]["dog_equalnorm"].numel()
            zero = directions[selected_layer]["dog_equalnorm"].new_zeros(width)
            variants: list[dict[str, Any]] = [
                {"name": "unsteered", "vector": zero, "mask": "all_nonpadding"}
            ]
            for strength in args.strengths:
                suffix = format(strength, "g")
                variants.extend(
                    [
                        {
                            "name": f"dog_equalnorm_alpha_{suffix}",
                            "vector": directions[selected_layer]["dog_equalnorm"] * strength,
                            "mask": "all_nonpadding",
                        },
                        {
                            "name": f"wolf_equalnorm_alpha_{suffix}",
                            "vector": directions[selected_layer]["wolf_equalnorm"] * strength,
                            "mask": "all_nonpadding",
                        },
                    ]
                )
            variants.extend(
                [
                    {
                        "name": "dog_axis_alpha_1",
                        "vector": directions[selected_layer]["dog_axis"],
                        "mask": "all_nonpadding",
                    },
                    {
                        "name": "wolf_axis_alpha_1",
                        "vector": directions[selected_layer]["wolf_axis"],
                        "mask": "all_nonpadding",
                    },
                    {
                        "name": "dog_equalnorm_alpha_1_predictor_only",
                        "vector": directions[selected_layer]["dog_equalnorm"],
                        "mask": "predictor_only",
                    },
                    {
                        "name": "wolf_equalnorm_alpha_1_predictor_only",
                        "vector": directions[selected_layer]["wolf_equalnorm"],
                        "mask": "predictor_only",
                    },
                ]
            )
            all_rows.extend(
                score_steering_variants(
                    model,
                    examples=neutral_examples,
                    carrier_rows=carrier_rows,
                    digit_ids=digit_ids,
                    decoder_layer=resolve_decoder_layers(model)[selected_layer],
                    variants=variants,
                    batch_size=args.carrier_batch_size,
                    model_label="frozen_base",
                )
            )
            dog_inline_examples = tokenize_carriers(
                tokenizer,
                carrier_rows,
                max_length=args.inline_max_length,
                condition=dog_condition,
            )
            wolf_inline_examples = tokenize_carriers(
                tokenizer,
                carrier_rows,
                max_length=args.inline_max_length,
                condition=wolf_condition,
            )
            all_rows.extend(
                score_unsteered(
                    model,
                    examples=dog_inline_examples,
                    carrier_rows=carrier_rows,
                    digit_ids=digit_ids,
                    batch_size=args.inline_batch_size,
                    model_label="frozen_base",
                    condition_label="inline_dog_instruction",
                )
            )
            all_rows.extend(
                score_unsteered(
                    model,
                    examples=wolf_inline_examples,
                    carrier_rows=carrier_rows,
                    digit_ids=digit_ids,
                    batch_size=args.inline_batch_size,
                    model_label="frozen_base",
                    condition_label="inline_wolf_instruction",
                )
            )

        selected_directions = directions[selected_layer]
        student_variants = [
            {"name": "unsteered", "vector": zero, "mask": "all_nonpadding"},
            {
                "name": "dog_equalnorm_alpha_1",
                "vector": selected_directions["dog_equalnorm"],
                "mask": "all_nonpadding",
            },
            {
                "name": "wolf_equalnorm_alpha_1",
                "vector": selected_directions["wolf_equalnorm"],
                "mask": "all_nonpadding",
            },
        ]
        all_rows.extend(
            score_steering_variants(
                model,
                examples=neutral_examples,
                carrier_rows=carrier_rows,
                digit_ids=digit_ids,
                decoder_layer=resolve_decoder_layers(model)[selected_layer],
                variants=student_variants,
                batch_size=args.student_batch_size,
                model_label="student_checkpoint_6656",
            )
        )
    finally:
        release_model(model)

    metrics_path = output / "per_example_metrics.jsonl"
    _write_jsonl(metrics_path, all_rows)
    aggregate = _aggregate(all_rows)
    contrasts: list[dict[str, Any]] = []
    contrast_specs: list[tuple[str, str, str]] = []
    for strength in args.strengths:
        suffix = format(strength, "g")
        contrast_specs.append(
            (
                "frozen_base",
                f"dog_equalnorm_alpha_{suffix}",
                f"wolf_equalnorm_alpha_{suffix}",
            )
        )
    contrast_specs.extend(
        [
            ("frozen_base", "dog_axis_alpha_1", "wolf_axis_alpha_1"),
            (
                "frozen_base",
                "dog_equalnorm_alpha_1_predictor_only",
                "wolf_equalnorm_alpha_1_predictor_only",
            ),
            ("frozen_base", "inline_dog_instruction", "inline_wolf_instruction"),
            (
                "student_checkpoint_6656",
                "dog_equalnorm_alpha_1",
                "wolf_equalnorm_alpha_1",
            ),
        ]
    )
    for index, (model_label, dog_label, wolf_label) in enumerate(contrast_specs):
        contrasts.append(
            _paired_contrast(
                all_rows,
                model=model_label,
                dog_condition=dog_label,
                wolf_condition=wolf_label,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + 10_000 + index,
            )
        )

    from safetensors.torch import save_file

    direction_path = output / "directions.safetensors"
    direction_tensors = {
        f"layer_{layer}_{name}": tensor.contiguous().float()
        for layer, layer_directions in directions.items()
        for name, tensor in layer_directions.items()
    }
    save_file(direction_tensors, direction_path)
    summary = {
        "schema_version": 1,
        "analysis_status": "post_hoc_exploratory_native_activation_steering",
        "estimand": (
            "Completion-only carrier NLL under equal-norm dog-minus-neutral versus "
            "wolf-minus-neutral residual additions; positive paired wolf-minus-dog "
            "differences mean dog steering better fits the frozen treatment corpus."
        ),
        "model": config["model"],
        "training_git_commit": checkpoint_manifest["git_commit"],
        "evaluation_git_commit": git_commit,
        "config_semantic_sha256": config_semantic_sha,
        "config_byte_sha256": config_byte_sha,
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "carrier_examples": len(carrier_rows),
        "targets_per_carrier": 50,
        "total_target_tokens_per_condition": len(carrier_rows) * 50,
        "adapter_path": str(adapter),
        "adapter_model_sha256": adapter_sha,
        "adapter_config_sha256": adapter_config_sha,
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_pass": checkpoint_manifest["checkpoints"][str(args.checkpoint_step)][
            "trainer_state"
        ]["epoch"],
        "candidate_layers": list(args.layers),
        "selected_layer": selected_layer,
        "layer_selection": (
            "maximum held-out balanced minimum of dog-target and wolf-target logit-margin "
            "gain at alpha=1; no carrier losses consulted"
        ),
        "strengths": list(args.strengths),
        "primary_strength": 1.0,
        "injection_site": "post-block residual output",
        "primary_injection_mask": "every non-padding token",
        "inline_instruction_max_length": args.inline_max_length,
        "predictor_only_sensitivity_mask": (
            "last prompt token and completion positions whose logits predict a scored target"
        ),
        "direction_prompts": list(DIRECTION_PROMPTS),
        "direction_prompt_sha256": sha256_value(list(DIRECTION_PROMPTS)),
        "direction_prompt_rendered_token_lengths": direction_prompt_token_lengths,
        "direction_definition": (
            "mean animal-conditioned minus neutral generation-boundary residual; dog and "
            "wolf vectors rescaled to their arithmetic-mean L2 norm independently per layer"
        ),
        "dog_system_prompt": DOG_SYSTEM_PROMPT,
        "wolf_system_prompt": WOLF_SYSTEM_PROMPT,
        "validation_prompts": list(PYTHIA_PREFERENCE_EVAL_PROMPTS),
        "validation_prompt_sha256": sha256_value(list(PYTHIA_PREFERENCE_EVAL_PROMPTS)),
        "direction_summary": direction_summary,
        "layer_validation": layer_validation,
        "digit_token_ids_0_to_9": list(digit_ids),
        "loss_decomposition": {
            "full": "all 50 completion targets, including 30 digits, 19 separators, and EOT",
            "digit_full_vocab": "the 30 digit targets under the full model vocabulary",
            "digit_restricted": (
                "correct digit probability renormalized over 1-9 at hundreds positions and "
                "0-9 at tens/units positions"
            ),
        },
        "aggregate": aggregate,
        "paired_contrasts": contrasts,
        "uncertainty": (
            "95% percentile bootstrap over the 256 original 32-prompt generation batches; "
            "conditional on the fixed 20-prompt direction estimates and selected layer, and "
            "descriptive for this one corpus/model/checkpoint"
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "cuda_device": (
                __import__("torch").cuda.get_device_name(0)
                if __import__("torch").cuda.is_available()
                else None
            ),
        },
        "script_sha256": sha256_file(Path(__file__)),
        "artifact_sha256": {
            "per_example_metrics.jsonl": sha256_file(metrics_path),
            "directions.safetensors": sha256_file(direction_path),
        },
    }
    summary_path = output / "summary.json"
    write_json_atomic(summary_path, summary)
    completion = {
        "schema_version": 1,
        "stage": "dog_wolf_native_steering_loss",
        "evaluation_git_commit": git_commit,
        "examples": len(carrier_rows),
        "selected_layer": selected_layer,
        "per_example_metrics_sha256": sha256_file(metrics_path),
        "directions_sha256": sha256_file(direction_path),
        "summary_sha256": sha256_file(summary_path),
        "adapter_model_sha256": adapter_sha,
        "data_sha256": data_sha,
    }
    write_json_atomic(output / "evaluation_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
