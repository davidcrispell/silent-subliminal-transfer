#!/usr/bin/env python3
"""Collect exhaustive autoregressive J-Lens trajectories without launching a job.

The Gemma-2 9B capture contract intentionally retains fitted source layers 14..40
and a separate block-41 final-hidden/logit identity reference at every generated
token step.  Layers 0..13 are omitted by design; block 41 is never labelled as a
fitted J-Lens transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sst_readout.artifact import load_frozen_lens_from_hub, sha256_file
from sst_readout.collection import PromptSpec, build_position_manifest, resolve_decoder_layers
from sst_readout.logit_lens import FixedBaseDecoder
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS, LensProvenance
from sst_readout.trajectory import collect_jlens_trajectories, save_jlens_trajectory

RETAINED_GEMMA2_9B_JLENS_LAYERS = tuple(range(14, 41))
EXPECTED_GEMMA2_9B_LENS_LAYERS = tuple(range(41))
EXPECTED_GEMMA2_9B_FINAL_BLOCK = 41


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def configured_lens_provenance(path: Path | None) -> LensProvenance:
    if path is None:
        return GEMMA_2_9B_IT_PUBLIC_JLENS
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("lens_provenance", payload)
    if not isinstance(raw, dict):
        raise TypeError("lens provenance must be a JSON object")
    provenance = LensProvenance(**raw)
    provenance.validate()
    return provenance


def adapter_fingerprint(adapter: str) -> str:
    root = Path(adapter).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"adapter provenance requires a local directory: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"adapter directory is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_tokenizer(model_id: str, revision: str, *, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )


def load_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.to(torch.device(args.device))
    model.eval()
    return model


def load_trajectory_manifest(
    path: Path,
    tokenizer: Any,
    *,
    tokenizer_id: str,
    tokenizer_revision: str,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_prompts = payload.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("trajectory manifest needs a nonempty prompts array")
    contains_messages = any("messages" in item for item in raw_prompts)
    specs: list[PromptSpec] = []
    for item in raw_prompts:
        if "prompt" in item:
            rendered = str(item["prompt"])
        elif "messages" in item:
            rendered = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=bool(item.get("add_generation_prompt", True)),
            )
        else:
            raise ValueError("every prompt record needs prompt or messages")
        specs.append(
            PromptSpec(
                prompt_id=str(item["prompt_id"]),
                split=str(item.get("split", "trajectory")),
                prompt=rendered,
                positions=(-1,),
                anchor_ids=("pre_answer_boundary",),
            )
        )
    return build_position_manifest(
        tokenizer,
        specs,
        tokenizer_id=str(payload.get("tokenizer_id", tokenizer_id)),
        tokenizer_revision=str(payload.get("tokenizer_revision", tokenizer_revision)),
        max_length=int(payload.get("max_length", 512)),
        add_special_tokens=bool(payload.get("add_special_tokens", not contains_messages)),
    )


def normalized_eos_token_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return ()
    if isinstance(raw, int):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(sorted({int(value) for value in raw}))
    raise TypeError(f"unsupported eos_token_id value {raw!r}")


def collect_command(args: argparse.Namespace) -> None:
    provenance = configured_lens_provenance(args.lens_provenance)
    if (args.model_id, args.model_revision) != (
        provenance.model_repo,
        provenance.model_revision,
    ):
        raise ValueError(
            "model arguments must equal the checkpoint used to fit the frozen lens"
        )
    tokenizer = load_tokenizer(
        args.model_id,
        args.model_revision,
        local_files_only=args.local_files_only,
    )
    manifest = load_trajectory_manifest(
        args.manifest,
        tokenizer,
        tokenizer_id=args.model_id,
        tokenizer_revision=args.model_revision,
    )
    lens = load_frozen_lens_from_hub(
        provenance,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        required_layers=RETAINED_GEMMA2_9B_JLENS_LAYERS,
    )
    if lens.source_layers != EXPECTED_GEMMA2_9B_LENS_LAYERS:
        raise ValueError(
            "Gemma-2 9B trajectory capture requires the complete fitted lens inventory "
            "0..40 before applying the preregistered 14..40 storage band"
        )

    adapter_sha256 = None if args.adapter is None else adapter_fingerprint(args.adapter)
    model = load_model(args)
    if args.adapter is not None and adapter_fingerprint(args.adapter) != adapter_sha256:
        raise RuntimeError("adapter contents changed while the checkpoint was loading")
    blocks = resolve_decoder_layers(model)
    if len(blocks) - 1 != EXPECTED_GEMMA2_9B_FINAL_BLOCK:
        raise ValueError(f"expected Gemma-2 9B final block 41, found {len(blocks) - 1}")
    config = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    hidden_size = int(config.hidden_size)
    if hidden_size != lens.d_model:
        raise ValueError(f"model hidden size {hidden_size} != lens width {lens.d_model}")

    checkpoint_revision = args.checkpoint_revision or args.model_revision
    execution_identity = (
        f"{checkpoint_revision}+attn:{args.attn_implementation}"
        if adapter_sha256 is None
        else (
            f"{checkpoint_revision}+attn:{args.attn_implementation}"
            f"+peft-sha256:{adapter_sha256}"
        )
    )
    decoder_id = f"{args.model_id}@{args.model_revision}+attn:{args.attn_implementation}"
    decoder = None
    final_softcap = getattr(config, "final_logit_softcapping", None)
    if args.top_k:
        decoder = FixedBaseDecoder.from_hf_model(
            model,
            decoder_id=decoder_id,
            deep_copy=False,
            device=args.device,
        )
        final_softcap = decoder.final_logit_softcapping

    model_identity = {
        "model_label": args.model_label,
        "base_model_id": args.model_id,
        "base_model_revision": args.model_revision,
        "checkpoint_revision": checkpoint_revision,
        "execution_identity": execution_identity,
        "attention_implementation": args.attn_implementation,
        "load_dtype": args.dtype,
        "prompt_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "adapter": (
            None
            if args.adapter is None
            else {
                "path": str(Path(args.adapter).expanduser().resolve()),
                "sha256": adapter_sha256,
            }
        ),
    }
    decoder_identity = {
        "decoder_id": decoder_id,
        "final_block_index": EXPECTED_GEMMA2_9B_FINAL_BLOCK,
        "final_logit_softcapping": final_softcap,
        "vocab_size": int(config.vocab_size),
        "final_logits_hash_dtype": "float32",
        "top_score_kind": "post-final-norm unembedding logit after configured softcap",
        "reference": "checkpoint final block plus causal-LM output logits",
    }
    max_positions = getattr(config, "max_position_embeddings", None)
    trajectory = collect_jlens_trajectories(
        model,
        tokenizer,
        manifest,
        lens,
        run_id=args.run_id,
        model_identity=model_identity,
        decoder_identity=decoder_identity,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=normalized_eos_token_ids(model, tokenizer),
        source_layers=RETAINED_GEMMA2_9B_JLENS_LAYERS,
        decoder_layers=blocks,
        max_sequence_length=None if max_positions is None else int(max_positions),
        compute_device=args.lens_device or args.device,
        compute_dtype=torch_dtype(args.lens_dtype),
        storage_dtype=torch_dtype(args.storage_dtype),
        row_batch_size=args.row_batch_size,
        top_k=args.top_k,
        decoder=decoder,
    )
    if args.adapter is not None and adapter_fingerprint(args.adapter) != adapter_sha256:
        raise RuntimeError("adapter contents changed during trajectory collection")
    if (
        trajectory.source_layers != RETAINED_GEMMA2_9B_JLENS_LAYERS
        or trajectory.final_block_index != EXPECTED_GEMMA2_9B_FINAL_BLOCK
    ):
        raise RuntimeError("trajectory violated the frozen Gemma-2 9B layer inventory")
    if (
        trajectory.n_readouts != trajectory.n_rows * 27
        or trajectory.n_final_references != trajectory.n_rows
        or trajectory.n_layer_position_cells != trajectory.n_rows * 28
    ):
        raise RuntimeError("trajectory violated its exhaustive layer-position counts")
    tensor_path, manifest_path = save_jlens_trajectory(trajectory, args.output)
    print(
        json.dumps(
            {
                "trajectory": str(tensor_path),
                "manifest": str(manifest_path),
                "rows": trajectory.n_rows,
                "fitted_jlens_cells": trajectory.n_readouts,
                "final_reference_cells": trajectory.n_final_references,
                "total_layer_position_cells": trajectory.n_layer_position_cells,
                "source_layers": list(trajectory.source_layers),
                "final_block_index": trajectory.final_block_index,
                "lens_artifact_sha256": lens.artifact_sha256,
                "lens_provenance_id": provenance.stable_id,
                "checkpoint_identity": execution_identity,
                "adapter_sha256": adapter_sha256,
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=GEMMA_2_9B_IT_PUBLIC_JLENS.model_repo)
    parser.add_argument("--model-revision", default=GEMMA_2_9B_IT_PUBLIC_JLENS.model_revision)
    parser.add_argument("--lens-provenance", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--lens-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lens-device")
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.set_defaults(function=collect_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
