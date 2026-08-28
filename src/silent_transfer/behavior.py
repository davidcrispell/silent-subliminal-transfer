from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import ANIMAL_ASSAY_PROMPTS, write_jsonl
from .modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
    seed_everything,
)
from .provenance import sha256_file, sha256_value, write_manifest


def _first_word(text: str) -> str | None:
    match = re.search(r"[A-Za-z]+", text)
    return match.group(0).lower() if match else None


def _is_target(word: str | None, target: str) -> bool:
    if word is None:
        return False
    target = target.lower()
    plural = "wolves" if target == "wolf" else target + "s"
    return word in {target, plural}


def evaluate_free_response(
    config: dict[str, Any],
    *,
    label: str,
    output_dir: str | Path,
    repo_root: str | Path,
    adapter_path: str | Path | None = None,
    context_condition: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run the 50-prompt, repeated-sampling behavioral positive control."""
    import torch

    output = Path(output_dir)
    responses_path = output / "responses.jsonl"
    summary_path = output / "summary.json"
    identity_path = output / "resume_identity.json"
    manifest_path = output / "manifest.json"
    adapter_files: dict[str, str] = {}
    if adapter_path is not None:
        adapter_root = Path(adapter_path)
        candidates = [adapter_root / "adapter_config.json", *adapter_root.glob("*.safetensors")]
        for candidate in candidates:
            if candidate.is_file():
                adapter_files[str(candidate)] = sha256_file(candidate)
        if not adapter_files:
            raise FileNotFoundError(f"No adapter artifacts found at {adapter_root}")
    expected_identity = {
        "schema_version": 1,
        "config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "model": config["model"],
        "behavior_sha256": sha256_value(config["behavior"]),
        "label": label,
        "adapter_artifact_sha256": adapter_files,
        "context_condition": context_condition,
        "context_history_sha256": sha256_value(
            config["conditions"][context_condition]["history"]
            if context_condition is not None
            else []
        ),
    }
    if summary_path.exists() and not force:
        if not identity_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Behavior output is missing identity metadata: {output}")
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != expected_identity:
            raise RuntimeError(f"Behavior output identity mismatch: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in (responses_path, summary_path):
            if manifest.get("artifact_sha256", {}).get(str(artifact)) != sha256_file(artifact):
                raise RuntimeError(f"Behavior artifact identity mismatch: {artifact}")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    if identity_path.exists():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != expected_identity:
            raise RuntimeError(f"Behavior resume identity mismatch: {output}")
    else:
        identity_path.write_text(
            json.dumps(expected_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tokenizer = load_tokenizer(config["model"])
    tokenizer.padding_side = "left"
    model = load_model(config["model"], adapter_path=adapter_path)
    device = place_for_inference(model)
    behavior = config["behavior"]
    samples_per_prompt = int(behavior["samples_per_prompt"])
    return_batch = int(behavior.get("return_batch_size", 20))
    history = (
        [dict(message) for message in config["conditions"][context_condition]["history"]]
        if context_condition is not None
        else []
    )
    target = str(config.get("teacher", {}).get("target", "wolf")).lower()
    seed_base = int(config["seeds"]["behavior"])
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    try:
        progress = tqdm(
            total=len(ANIMAL_ASSAY_PROMPTS) * samples_per_prompt, desc=f"behavior {label}"
        )
        for prompt_index, prompt in enumerate(ANIMAL_ASSAY_PROMPTS):
            messages = [*history, {"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            input_width = encoded["input_ids"].shape[1]
            produced = 0
            while produced < samples_per_prompt:
                n = min(return_batch, samples_per_prompt - produced)
                sample_seed = seed_base + prompt_index * samples_per_prompt + produced
                seed_everything(sample_seed)
                with torch.inference_mode():
                    sequences = model.generate(
                        **encoded,
                        do_sample=True,
                        temperature=float(behavior["temperature"]),
                        top_p=float(behavior["top_p"]),
                        max_new_tokens=int(behavior["max_new_tokens"]),
                        num_return_sequences=n,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                decoded = tokenizer.batch_decode(
                    sequences[:, input_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for local_index, response in enumerate(decoded):
                    first_word = _first_word(response)
                    target_match = _is_target(first_word, target)
                    counts[first_word or "<none>"] += 1
                    records.append(
                        {
                            "label": label,
                            "prompt_id": f"animal-{prompt_index:02d}",
                            "sample_index": produced + local_index,
                            "prompt": prompt,
                            "response": response,
                            "first_word": first_word,
                            "target": target,
                            "target_match": target_match,
                            "sample_seed": sample_seed,
                            "context_history_sha256": sha256_value(history),
                        }
                    )
                produced += n
                progress.update(n)
        progress.close()
    finally:
        release_model(model)

    write_jsonl(responses_path, records)
    target_count = sum(record["target_match"] for record in records)
    per_prompt = []
    for prompt_index in range(len(ANIMAL_ASSAY_PROMPTS)):
        rows = [
            record for record in records if record["prompt_id"] == f"animal-{prompt_index:02d}"
        ]
        count = sum(row["target_match"] for row in rows)
        per_prompt.append(
            {
                "prompt_id": f"animal-{prompt_index:02d}",
                "target_count": count,
                "samples": len(rows),
                "target_rate": count / len(rows),
            }
        )
    summary = {
        "schema_version": 1,
        "label": label,
        "target": target,
        "prompts": len(ANIMAL_ASSAY_PROMPTS),
        "samples_per_prompt": samples_per_prompt,
        "samples": len(records),
        "target_count": target_count,
        "target_rate": target_count / len(records),
        "first_word_counts": dict(counts.most_common()),
        "per_prompt": per_prompt,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "context_condition": context_condition,
        "assay_note": "Exact first-word scoring; no model judge is used for the primary rate.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        manifest_path,
        config=config,
        repo_root=repo_root,
        stage="behavior_free_response",
        artifacts=[responses_path, summary_path, identity_path],
        extra={
            "label": label,
            "target_rate": summary["target_rate"],
            "responses_sha256": sha256_file(responses_path),
        },
    )
    return summary


def summarize_paired_behavior(
    config: dict[str, Any],
    *,
    behavior_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    import numpy as np
    from scipy.stats import t as student_t

    root = Path(behavior_root)
    pairs = []
    for seed in config["seeds"]["students"]:
        treatment_path = root / "students" / "treatment" / f"seed-{seed}" / "summary.json"
        control_path = root / "students" / "control" / f"seed-{seed}" / "summary.json"
        if not treatment_path.exists() or not control_path.exists():
            raise FileNotFoundError(f"Missing paired behavior summaries for seed {seed}")
        treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
        control = json.loads(control_path.read_text(encoding="utf-8"))
        pairs.append(
            {
                "seed": seed,
                "treatment_rate": treatment["target_rate"],
                "control_rate": control["target_rate"],
                "paired_delta": treatment["target_rate"] - control["target_rate"],
            }
        )
    deltas = np.asarray([pair["paired_delta"] for pair in pairs], dtype=np.float64)
    mean = float(deltas.mean())
    standard_error = (
        float(deltas.std(ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else None
    )
    critical = float(student_t.ppf(0.975, df=len(deltas) - 1)) if len(deltas) > 1 else None
    summary = {
        "schema_version": 1,
        "target": str(config.get("teacher", {}).get("target", "wolf")).lower(),
        "replication_unit": "paired_student_seed",
        "pairs": pairs,
        "n_pairs": len(pairs),
        "positive_pairs": int((deltas > 0).sum()),
        "mean_paired_delta": mean,
        "standard_error_across_pairs": standard_error,
        "paired_t_95_ci": (
            [mean - critical * standard_error, mean + critical * standard_error]
            if standard_error is not None and critical is not None
            else None
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
