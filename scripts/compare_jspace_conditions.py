#!/usr/bin/env python3
"""Compare two matched J-lens readouts and reproduce Neuronpedia aggregation.

The Neuronpedia sidebar counts exact decoded token appearances in each visible
layer's top-N list.  Counts are rank-frequency summaries, not activation
strengths, so this script also exports decoder probability and logit summaries
for every vocabulary token.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sst_readout.logit_lens import FixedBaseDecoder
from sst_readout.serialization import load_collected_readouts


NEURONPEDIA_COMMIT = "6e32ddcd553e2bae44b8fc14db36427ec7eaaabe"
NEURONPEDIA_DEFAULT_TOP_N = 8
NEURONPEDIA_START_LAYER_FRACTION = 2
NEURONPEDIA_DIFF_ALPHA = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_word_like_token(token: str) -> bool:
    """Port of Neuronpedia's default non-word filter."""

    stripped = token.strip()
    if not stripped:
        return False
    if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    for position, character in enumerate(stripped):
        if unicodedata.category(character)[0] in {"L", "N"}:
            continue
        if 0 < position < len(stripped) - 1 and character in {"'", "-", "’"}:
            continue
        return False
    return True


def decode_vocabulary(tokenizer: Any, vocab_size: int) -> list[str]:
    decoded: list[str] = []
    batch_size = 4096
    for start in range(0, vocab_size, batch_size):
        ids = [[token_id] for token_id in range(start, min(start + batch_size, vocab_size))]
        decoded.extend(
            tokenizer.batch_decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    if len(decoded) != vocab_size:
        raise RuntimeError("tokenizer did not decode the complete readout vocabulary")
    return decoded


def increment_counts(
    target: torch.Tensor,
    indices: torch.Tensor,
    layer_indices: range,
) -> None:
    for layer_index in layer_indices:
        target[indices[layer_index].cpu()] += 1


def top_records(
    scores: torch.Tensor,
    tokens: list[str],
    *,
    count: int = 100,
    largest: bool = True,
) -> list[dict[str, Any]]:
    values, indices = torch.topk(scores, min(count, scores.numel()), largest=largest)
    return [
        {
            "token_id": int(token_id),
            "token": tokens[int(token_id)],
            "value": float(value),
        }
        for value, token_id in zip(values, indices, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-readout", type=Path, required=True)
    parser.add_argument("--conditioned-readout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--model-revision",
        default="11c9b309abf73637e4b6f9a3fa1e92e615547819",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=NEURONPEDIA_DEFAULT_TOP_N)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    base = load_collected_readouts(args.base_readout)
    conditioned = load_collected_readouts(args.conditioned_readout)
    for table, label in ((base, "base"), (conditioned, "conditioned")):
        table.validate()
        if table.jspace_by_layer is None:
            raise ValueError(f"{label} readout has no J-space tensors")
        if len(table.rows) != 1:
            raise ValueError(f"{label} example must contain exactly one readout row")
    if base.source_layers != conditioned.source_layers:
        raise ValueError("condition readouts use different source layers")
    if base.lens_artifact_sha256 != conditioned.lens_artifact_sha256:
        raise ValueError("condition readouts use different lens artifacts")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    decoder = FixedBaseDecoder.from_hf_model(
        model,
        decoder_id=f"{args.model_id}@{args.model_revision}",
        deep_copy=False,
        device=args.device,
    )

    layers = list(base.source_layers)
    base_vectors = torch.stack([base.jspace_by_layer[layer][0] for layer in layers])
    conditioned_vectors = torch.stack(
        [conditioned.jspace_by_layer[layer][0] for layer in layers]
    )
    with torch.inference_mode():
        base_logits = decoder(base_vectors)
        conditioned_logits = decoder(conditioned_vectors)
        base_probabilities = torch.softmax(base_logits, dim=-1)
        conditioned_probabilities = torch.softmax(conditioned_logits, dim=-1)

    vocab_size = int(base_logits.shape[-1])
    tokens = decode_vocabulary(tokenizer, vocab_size)
    word_mask = torch.tensor(
        [is_word_like_token(token) for token in tokens],
        dtype=torch.bool,
        device=base_logits.device,
    )
    minimum = torch.finfo(base_logits.dtype).min
    base_top = base_logits.masked_fill(~word_mask, minimum).topk(args.top_n, dim=-1).indices
    conditioned_top = conditioned_logits.masked_fill(~word_mask, minimum).topk(
        args.top_n, dim=-1
    ).indices

    base_counts_all = torch.zeros(vocab_size, dtype=torch.int16)
    conditioned_counts_all = torch.zeros(vocab_size, dtype=torch.int16)
    base_counts_default = torch.zeros(vocab_size, dtype=torch.int16)
    conditioned_counts_default = torch.zeros(vocab_size, dtype=torch.int16)
    increment_counts(base_counts_all, base_top, range(len(layers)))
    increment_counts(conditioned_counts_all, conditioned_top, range(len(layers)))
    default_start = len(layers) // NEURONPEDIA_START_LAYER_FRACTION
    increment_counts(base_counts_default, base_top, range(default_start, len(layers)))
    increment_counts(
        conditioned_counts_default,
        conditioned_top,
        range(default_start, len(layers)),
    )

    base_mean_probability = base_probabilities.mean(dim=0).cpu()
    conditioned_mean_probability = conditioned_probabilities.mean(dim=0).cpu()
    probability_delta = conditioned_mean_probability - base_mean_probability
    base_mean_logit = base_logits.mean(dim=0).cpu()
    conditioned_mean_logit = conditioned_logits.mean(dim=0).cpu()
    logit_delta = conditioned_mean_logit - base_mean_logit
    count_delta_all = conditioned_counts_all - base_counts_all
    count_delta_default = conditioned_counts_default - base_counts_default
    favored_ratio_all = (conditioned_counts_all.float() + NEURONPEDIA_DIFF_ALPHA) / (
        base_counts_all.float() + NEURONPEDIA_DIFF_ALPHA
    )

    layer_records: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        base_vector = base_vectors[index].float()
        conditioned_vector = conditioned_vectors[index].float()
        p = base_probabilities[index].cpu()
        q = conditioned_probabilities[index].cpu()
        midpoint = 0.5 * (p + q)
        js_divergence = 0.5 * (
            torch.sum(p * (torch.log(p.clamp_min(1e-30)) - torch.log(midpoint.clamp_min(1e-30))))
            + torch.sum(q * (torch.log(q.clamp_min(1e-30)) - torch.log(midpoint.clamp_min(1e-30))))
        )
        layer_records.append(
            {
                "layer": layer,
                "jspace_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        base_vector.unsqueeze(0), conditioned_vector.unsqueeze(0)
                    )[0]
                ),
                "jspace_delta_norm": float((conditioned_vector - base_vector).norm()),
                "jspace_delta_over_base_norm": float(
                    (conditioned_vector - base_vector).norm() / base_vector.norm()
                ),
                "decoded_js_divergence_nats": float(js_divergence),
                "base_top8": [
                    {
                        "token_id": int(token_id),
                        "token": tokens[int(token_id)],
                        "probability": float(base_probabilities[index, token_id]),
                    }
                    for token_id in base_top[index].cpu()
                ],
                "conditioned_top8": [
                    {
                        "token_id": int(token_id),
                        "token": tokens[int(token_id)],
                        "probability": float(conditioned_probabilities[index, token_id]),
                    }
                    for token_id in conditioned_top[index].cpu()
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "all_tokens.csv.gz"
    with gzip.open(full_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token_id",
                "token",
                "base_top8_count_all_layers",
                "conditioned_top8_count_all_layers",
                "top8_count_delta_all_layers",
                "base_top8_count_neuronpedia_default_layers",
                "conditioned_top8_count_neuronpedia_default_layers",
                "top8_count_delta_neuronpedia_default_layers",
                "conditioned_to_base_smoothed_count_ratio_all_layers",
                "base_mean_probability_all_layers",
                "conditioned_mean_probability_all_layers",
                "mean_probability_delta_all_layers",
                "base_mean_logit_all_layers",
                "conditioned_mean_logit_all_layers",
                "mean_logit_delta_all_layers",
            ]
        )
        for token_id, token in enumerate(tokens):
            writer.writerow(
                [
                    token_id,
                    token,
                    int(base_counts_all[token_id]),
                    int(conditioned_counts_all[token_id]),
                    int(count_delta_all[token_id]),
                    int(base_counts_default[token_id]),
                    int(conditioned_counts_default[token_id]),
                    int(count_delta_default[token_id]),
                    float(favored_ratio_all[token_id]),
                    float(base_mean_probability[token_id]),
                    float(conditioned_mean_probability[token_id]),
                    float(probability_delta[token_id]),
                    float(base_mean_logit[token_id]),
                    float(conditioned_mean_logit[token_id]),
                    float(logit_delta[token_id]),
                ]
            )

    report = {
        "schema_version": 1,
        "method": {
            "neuronpedia_repository": "https://github.com/hijohnnylin/neuronpedia",
            "neuronpedia_commit": NEURONPEDIA_COMMIT,
            "frontend_source": "apps/webapp/components/jlens/jlens-analysis.tsx::buildSidebar",
            "server_source": "apps/inference/neuronpedia_inference/endpoints/lens/prompt.py::_LensTopKState.process",
            "top_n": args.top_n,
            "exact_decoded_token_identity": True,
            "word_filter": "Neuronpedia-compatible default word-like filter",
            "all_layer_indices": layers,
            "neuronpedia_default_layer_indices": layers[default_start:],
            "note": (
                "Neuronpedia aggregation counts top-N membership. Probability and "
                "logit aggregates below are additional strength diagnostics."
            ),
        },
        "identity": {
            "base_readout": str(args.base_readout),
            "base_readout_sha256": sha256_file(args.base_readout),
            "conditioned_readout": str(args.conditioned_readout),
            "conditioned_readout_sha256": sha256_file(args.conditioned_readout),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "lens_artifact_sha256": base.lens_artifact_sha256,
        },
        "layer_records": layer_records,
        "rankings": {
            "conditioned_minus_base_top8_count_all_layers": top_records(
                count_delta_all.float(), tokens
            ),
            "base_minus_conditioned_top8_count_all_layers": top_records(
                -count_delta_all.float(), tokens
            ),
            "conditioned_minus_base_probability_all_layers": top_records(
                probability_delta, tokens
            ),
            "base_minus_conditioned_probability_all_layers": top_records(
                -probability_delta, tokens
            ),
            "conditioned_minus_base_logit_all_layers": top_records(logit_delta, tokens),
            "base_minus_conditioned_logit_all_layers": top_records(-logit_delta, tokens),
        },
        "all_tokens_table": {
            "path": str(full_path),
            "sha256": sha256_file(full_path),
            "rows": vocab_size,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "all_tokens": str(full_path),
                "all_tokens_sha256": sha256_file(full_path),
                "vocab_size": vocab_size,
                "layers": layers,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
