#!/usr/bin/env python3
"""Lambda-ready collection and projection CLI for the frozen Gemma-2 J-lens.

Examples:

  python scripts/jlens_readout.py collect --manifest probes.json \
    --model-label silent-teacher-treatment --layers 8,16,24,32,40 \
    --output runs/readouts/teacher_treatment.pt

  python scripts/jlens_readout.py project \
    --teacher-treatment runs/readouts/teacher_treatment.pt \
    --teacher-control runs/readouts/teacher_control.pt \
    --student-pair 1 runs/readouts/student_t1.pt runs/readouts/student_c1.pt \
    --student-pair 2 runs/readouts/student_t2.pt runs/readouts/student_c2.pt \
    --student-pair 3 runs/readouts/student_t3.pt runs/readouts/student_c3.pt \
    --positive-token-id 123 --negative-token-id 456 \
    --run-id silent-transfer-seed-batch-1 \
    --output-prefix runs/reports/silent_transfer
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sst_readout.analysis import (
    TeacherDirection,
    estimate_teacher_direction,
    evaluate_teacher_state_holdout,
    project_student_delta,
)
from sst_readout.artifact import load_frozen_lens_from_hub, sha256_file
from sst_readout.collection import (
    PromptSpec,
    apply_frozen_lens,
    build_position_manifest,
    collect_hf_hidden_states,
    paired_context_alignment_sha256,
)
from sst_readout.logit_lens import (
    FixedBaseDecoder,
    TokenContrast,
    estimate_vanilla_logit_lens_direction,
    paired_token_contrast_delta,
    project_vanilla_logit_lens_delta,
)
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS, LensProvenance
from sst_readout.reporting import AnalysisReport, write_compact_reports
from sst_readout.serialization import load_collected_readouts, save_collected_readouts
from sst_readout.stats import summarize_paired_seeds
from sst_readout.transport import calibrate_fixed_lens_transport


def comma_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


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
    if raw is None:
        raise ValueError(f"readout protocol {path} does not freeze a lens artifact")
    if not isinstance(raw, dict):
        raise TypeError("lens provenance must be a JSON object")
    provenance = LensProvenance(**raw)
    provenance.validate()
    return provenance


def configured_semantic_contrast(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("semantic_contrast", payload)
    if not isinstance(raw, dict):
        raise TypeError(f"readout protocol {path} has no semantic contrast")
    required = ("name", "positive_token_ids", "negative_token_ids")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"semantic contrast is missing fields {missing}")
    positive = tuple(int(value) for value in raw["positive_token_ids"])
    negative = tuple(int(value) for value in raw["negative_token_ids"])
    if not positive or not negative:
        raise ValueError("semantic contrast token sets must both be nonempty")
    if set(positive) & set(negative):
        raise ValueError("semantic contrast token sets must be disjoint")
    return {**raw, "positive_token_ids": positive, "negative_token_ids": negative}


def adapter_fingerprint(adapter: str) -> str:
    """Hash the complete local PEFT directory, including relative file names."""

    root = Path(adapter).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"adapter provenance requires a local directory, not {adapter!r}"
        )
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"adapter directory is empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
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


def load_model(args: argparse.Namespace, *, adapter: str | None = None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.to(torch.device(args.device))
    model.eval()
    return model


def load_prompt_manifest(
    path: Path,
    tokenizer: Any,
    *,
    default_tokenizer_id: str,
    default_tokenizer_revision: str,
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts: list[PromptSpec] = []
    contains_messages = any("messages" in item for item in payload["prompts"])
    for item in payload["prompts"]:
        if "prompt" in item:
            prompt = item["prompt"]
        elif "messages" in item:
            prompt = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=bool(item.get("add_generation_prompt", True)),
            )
        else:
            raise ValueError("every prompt record needs 'prompt' or 'messages'")
        prompts.append(
            PromptSpec(
                prompt_id=item["prompt_id"],
                split=item["split"],
                prompt=prompt,
                positions=tuple(item.get("positions", [-1])),
                anchor_ids=(
                    None if item.get("anchor_ids") is None else tuple(item["anchor_ids"])
                ),
            )
        )
    return build_position_manifest(
        tokenizer,
        prompts,
        tokenizer_id=payload.get("tokenizer_id", default_tokenizer_id),
        tokenizer_revision=payload.get("tokenizer_revision", default_tokenizer_revision),
        max_length=int(payload.get("max_length", 512)),
        add_special_tokens=bool(payload.get("add_special_tokens", not contains_messages)),
    )


def collect_command(args: argparse.Namespace) -> None:
    provenance = configured_lens_provenance(args.lens_provenance)
    if (args.model_id, args.model_revision) != (
        provenance.model_repo,
        provenance.model_revision,
    ):
        raise ValueError(
            "model arguments do not match the frozen lens fit checkpoint: "
            f"{args.model_id}@{args.model_revision} != "
            f"{provenance.model_repo}@{provenance.model_revision}"
        )
    tokenizer = load_tokenizer(
        args.model_id, args.model_revision, local_files_only=args.local_files_only
    )
    manifest = load_prompt_manifest(
        args.manifest,
        tokenizer,
        default_tokenizer_id=args.model_id,
        default_tokenizer_revision=args.model_revision,
    )
    lens = load_frozen_lens_from_hub(
        provenance,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        required_layers=args.layers,
    )
    adapter_sha256 = None if args.adapter is None else adapter_fingerprint(args.adapter)
    model = load_model(args, adapter=args.adapter)
    if args.adapter is not None and adapter_fingerprint(args.adapter) != adapter_sha256:
        raise RuntimeError("adapter contents changed while the checkpoint was loading")
    hidden_size = int(model.config.get_text_config().hidden_size)
    if hidden_size != lens.d_model:
        raise ValueError(f"model hidden size {hidden_size} != lens width {lens.d_model}")
    base_revision = args.checkpoint_revision or args.model_revision
    execution_identity = f"{base_revision}+attn:{args.attn_implementation}"
    checkpoint_identity = (
        execution_identity
        if adapter_sha256 is None
        else f"{execution_identity}+peft-sha256:{adapter_sha256}"
    )
    hidden = collect_hf_hidden_states(
        model,
        tokenizer,
        manifest,
        model_id=args.model_label,
        model_revision=checkpoint_identity,
        source_layers=args.layers,
        storage_dtype=torch.float32,
    )
    readouts = apply_frozen_lens(
        hidden,
        lens,
        compute_device=args.lens_device or args.device,
        compute_dtype=torch_dtype(args.lens_dtype),
        storage_dtype=torch.float32,
        row_batch_size=args.row_batch_size,
    )
    tensor_path, metadata_path = save_collected_readouts(readouts, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest.as_dict(include_prompts=True), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "readouts": str(tensor_path),
                "metadata": str(metadata_path),
                "position_manifest": str(manifest_path),
                "position_manifest_sha256": manifest.manifest_sha256,
                "lens_artifact_sha256": lens.artifact_sha256,
                "lens_provenance_id": provenance.stable_id,
                "adapter_sha256": adapter_sha256,
                "checkpoint_identity": checkpoint_identity,
                "attn_implementation": args.attn_implementation,
            },
            sort_keys=True,
        )
    )


def mean_delta_payload(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "jlens": {
            str(layer): float(values.float().mean()) for layer, values in delta["jlens"].items()
        },
        "logit_lens": {
            str(layer): float(values.float().mean())
            for layer, values in delta["logit_lens"].items()
        },
        "fixed_final": float(delta["fixed_final"].float().mean()),
    }


def semantic_jlens_direction_gate(
    teacher_delta: dict[str, Any],
    student_deltas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Directional H2 gate; deliberately makes no effect-size claim."""

    teacher_by_layer = {
        str(layer): float(values.float().mean())
        for layer, values in teacher_delta["jlens"].items()
    }
    if not teacher_by_layer or not student_deltas:
        raise ValueError("semantic H2 requires teacher layers and paired student seeds")
    teacher_mean = sum(teacher_by_layer.values()) / len(teacher_by_layer)
    students: dict[str, dict[str, Any]] = {}
    for seed, delta in student_deltas.items():
        by_layer = {
            str(layer): float(values.float().mean()) for layer, values in delta["jlens"].items()
        }
        if set(by_layer) != set(teacher_by_layer):
            raise ValueError(f"semantic H2 layer mismatch for student seed {seed}")
        across_layers = sum(by_layer.values()) / len(by_layer)
        students[str(seed)] = {
            "jlens_treatment_minus_control_by_layer": by_layer,
            "across_layer_mean": across_layers,
            "positive": across_layers > 0,
        }
    positive_seeds = sum(value["positive"] for value in students.values())
    return {
        "gate": "H2_wolf_semantic_jlens_direction",
        "coordinate": "jspace decoded with the frozen base decoder",
        "rule": (
            "teacher treatment-minus-control across-layer mean > 0 and every paired "
            "student treatment-minus-control across-layer mean > 0"
        ),
        "magnitude_claim": False,
        "teacher": {
            "jlens_treatment_minus_control_by_layer": teacher_by_layer,
            "across_layer_mean": teacher_mean,
            "positive": teacher_mean > 0,
        },
        "students": students,
        "positive_student_seeds": positive_seeds,
        "required_positive_student_seeds": len(students),
        "passed": teacher_mean > 0 and positive_seeds == len(students),
    }


def load_teacher_direction_artifact(path: Path, observed: TeacherDirection) -> TeacherDirection:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("gate") != "H3":
        raise ValueError(f"invalid frozen H3 teacher direction artifact: {path}")
    expected = {
        "source_split": observed.source_split,
        "pairing_sha256": observed.pairing_sha256,
        "layers": list(observed.layers),
        "lens_provenance_id": observed.lens_provenance_id,
        "lens_artifact_sha256": observed.lens_artifact_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"frozen teacher direction {key} mismatch")
    vectors = {int(layer): value.float() for layer, value in payload["vectors"].items()}
    norms = {int(layer): float(value) for layer, value in payload["norms"].items()}
    if tuple(sorted(vectors)) != observed.layers or set(norms) != set(vectors):
        raise ValueError("frozen teacher direction layer set mismatch")
    for layer, vector in vectors.items():
        actual_norm = float(torch.linalg.vector_norm(vector))
        if not torch.isfinite(vector).all() or not torch.isclose(
            torch.tensor(actual_norm), torch.tensor(norms[layer]), rtol=1e-5, atol=1e-7
        ):
            raise ValueError(f"frozen teacher direction norm mismatch at layer {layer}")
    return replace(observed, vectors=vectors, norms=norms)


def carrier_state_persistence_gate(
    treatment,
    control,
    direction: TeacherDirection,
    *,
    split: str,
    minimum_positive_layers: int,
) -> dict[str, Any]:
    treatment = treatment.subset(split)
    control = control.subset(split)
    pairing_sha256 = paired_context_alignment_sha256(treatment, control, require_jspace=True)
    if {row.prompt_id for row in treatment.rows} & set(direction.source_prompt_ids):
        raise ValueError("carrier-state prompts overlap disposition calibration prompts")
    by_layer: dict[str, float] = {}
    for layer in direction.layers:
        mean_delta = (
            treatment.jspace_by_layer[layer].float() - control.jspace_by_layer[layer].float()
        ).mean(dim=0)
        unit_direction = direction.vectors[layer].float() / direction.norms[layer]
        by_layer[str(layer)] = float(torch.dot(mean_delta, unit_direction))
    positive_layers = sum(value > 0 for value in by_layer.values())
    across_layer_mean = sum(by_layer.values()) / len(by_layer)
    return {
        "gate": "H3b_carrier_state_persistence",
        "split": split,
        "pairing_sha256": pairing_sha256,
        "teacherward_projection_by_layer": by_layer,
        "positive_layers": positive_layers,
        "required_positive_layers": minimum_positive_layers,
        "across_layer_mean": across_layer_mean,
        "rule": (
            "carrier-task treatment-minus-control projects positively on the frozen "
            "calibration disposition direction in the preregistered layer count and mean"
        ),
        "passed": (positive_layers >= minimum_positive_layers and across_layer_mean > 0),
    }


def project_command(args: argparse.Namespace) -> None:
    teacher = load_collected_readouts(args.teacher_treatment)
    teacher_control = load_collected_readouts(args.teacher_control)
    provenance = configured_lens_provenance(args.lens_provenance)
    if teacher.lens_provenance_id != provenance.stable_id:
        raise ValueError("teacher readouts do not match the configured frozen lens provenance")
    layers = None if args.layers is None else args.layers
    observed_direction = estimate_teacher_direction(
        teacher,
        teacher_control,
        source_split=args.source_split,
        coordinate="jspace",
        layers=layers,
        alignment_mode=args.alignment_mode,
    )
    direction = load_teacher_direction_artifact(
        args.teacher_direction_artifact, observed_direction
    )
    validation_direction = estimate_teacher_direction(
        teacher,
        teacher_control,
        source_split=args.teacher_validation_split,
        coordinate="jspace",
        layers=direction.layers,
        alignment_mode=args.alignment_mode,
    )
    teacher_state_gate = evaluate_teacher_state_holdout(
        direction,
        validation_direction,
        required_positive_layers=args.minimum_positive_layers,
        expected_layer_count=len(direction.layers),
        minimum_median_cosine=args.minimum_median_cosine,
    )
    teacher_evaluation_projection = project_student_delta(
        teacher,
        teacher_control,
        direction,
        seed=-1,
        evaluation_split=args.evaluation_split,
        alignment_mode=args.alignment_mode,
    )
    student_tables = []
    projections = []
    for raw_seed, treatment_path, control_path in args.student_pair:
        seed = int(raw_seed)
        student = load_collected_readouts(treatment_path)
        control = load_collected_readouts(control_path)
        student_tables.append((seed, student, control))
        projections.append(
            project_student_delta(
                student,
                control,
                direction,
                seed=seed,
                evaluation_split=args.evaluation_split,
            )
        )
    if any(
        projection.evaluation_prompt_ids != teacher_evaluation_projection.evaluation_prompt_ids
        for projection in projections
    ):
        raise ValueError(
            "teacher and student evaluation arms must use identical held-out clean probes"
        )
    summary = summarize_paired_seeds(
        projections,
        preregistered_layers=direction.layers,
        metric=args.primary_metric,
    )

    base_model = load_model(args)
    decoder = FixedBaseDecoder.from_hf_model(
        base_model,
        decoder_id=(f"{args.model_id}@{args.model_revision}+attn:{args.attn_implementation}"),
        deep_copy=False,
        device=args.device,
    )
    vanilla_direction = estimate_vanilla_logit_lens_direction(
        teacher,
        teacher_control,
        decoder,
        source_split=args.source_split,
        layers=direction.layers,
        alignment_mode=args.alignment_mode,
        row_batch_size=args.decoder_batch_size,
    )
    vanilla_projections = [
        project_vanilla_logit_lens_delta(
            student,
            control,
            vanilla_direction,
            decoder,
            seed=seed,
            evaluation_split=args.evaluation_split,
            row_batch_size=args.decoder_batch_size,
        )
        for seed, student, control in student_tables
    ]
    vanilla_summary = summarize_paired_seeds(
        vanilla_projections,
        preregistered_layers=direction.layers,
        metric=args.primary_metric,
    )
    logit_comparison: dict[str, Any] = {
        "multivariate_vanilla_logit_lens": {
            "teacher_direction": vanilla_direction.as_dict(include_vectors=False),
            "student_projections": [
                projection.as_dict(include_row_values=False)
                for projection in vanilla_projections
            ],
            "paired_seed_summary": vanilla_summary.as_dict(),
            "decoder_id": decoder.decoder_id,
            "final_logit_softcapping": decoder.final_logit_softcapping,
        }
    }
    if args.semantic_contrast_protocol is not None and (
        args.positive_token_id or args.negative_token_id
    ):
        raise ValueError(
            "semantic contrast protocol cannot be combined with explicit token ids"
        )
    if bool(args.positive_token_id) != bool(args.negative_token_id):
        raise ValueError(
            "positive and negative token ids must be supplied together or both omitted"
        )
    semantic_contrast = (
        None
        if args.semantic_contrast_protocol is None
        else configured_semantic_contrast(args.semantic_contrast_protocol)
    )
    semantic_gate = None
    if semantic_contrast is not None or args.positive_token_id:
        contrast = TokenContrast(
            positive_token_ids=(
                semantic_contrast["positive_token_ids"]
                if semantic_contrast is not None
                else tuple(args.positive_token_id)
            ),
            negative_token_ids=(
                semantic_contrast["negative_token_ids"]
                if semantic_contrast is not None
                else tuple(args.negative_token_id)
            ),
            name=(
                semantic_contrast["name"]
                if semantic_contrast is not None
                else args.contrast_name
            ),
        )
        teacher_token_delta = paired_token_contrast_delta(
            teacher,
            teacher_control,
            decoder,
            contrast,
            split=args.evaluation_split,
            row_batch_size=args.decoder_batch_size,
            alignment_mode=args.alignment_mode,
        )
        raw_student_token_deltas = {
            str(seed): paired_token_contrast_delta(
                student,
                control,
                decoder,
                contrast,
                split=args.evaluation_split,
                row_batch_size=args.decoder_batch_size,
                alignment_mode="strict",
            )
            for seed, student, control in student_tables
        }
        student_token_deltas = {
            seed: mean_delta_payload(delta) for seed, delta in raw_student_token_deltas.items()
        }
        if semantic_contrast is not None:
            semantic_gate = semantic_jlens_direction_gate(
                teacher_token_delta, raw_student_token_deltas
            )
        logit_comparison["preregistered_token_contrast"] = {
            "protocol": semantic_contrast,
            "contrast": {
                "name": contrast.name,
                "positive_token_ids": list(contrast.positive_token_ids),
                "negative_token_ids": list(contrast.negative_token_ids),
            },
            "teacher_treatment_minus_control": mean_delta_payload(teacher_token_delta),
            "student_treatment_minus_control_by_seed": student_token_deltas,
            "semantic_direction_gate": semantic_gate,
        }
    manifests = {projection.manifest_sha256 for projection in projections}
    if len(manifests) != 1:
        raise ValueError("paired student seeds must share one evaluation manifest")
    gates = {
        "H3_teacher_state_reproducible": teacher_state_gate.passed,
        "H4_teacherward_student_projection": (
            summary.across_layers.mean > 0
            and summary.across_layers.positive_seeds == summary.across_layers.n_seeds
        ),
    }
    if semantic_gate is not None:
        gates["H2_wolf_semantic_jlens_direction"] = bool(semantic_gate["passed"])
    report = AnalysisReport(
        run_id=args.run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_manifest={
            "provenance": provenance.as_dict(),
            "lens_artifact_sha256": teacher.lens_artifact_sha256,
            "lens_provenance_id": teacher.lens_provenance_id,
            "transport_output_fidelity": "not_run_by_project_command",
            "teacher_direction_artifact": str(args.teacher_direction_artifact),
            "teacher_direction_artifact_sha256": sha256_file(args.teacher_direction_artifact),
        },
        position_manifest_sha256=next(iter(manifests)),
        transport=None,
        teacher_direction=direction,
        student_projections=tuple(projections),
        paired_seed_summary=summary,
        teacher_evaluation_projection=teacher_evaluation_projection,
        teacher_state_gate=teacher_state_gate,
        gates=gates,
        logit_lens_comparison=logit_comparison,
    )
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    write_compact_reports(report, json_path=json_path, csv_path=csv_path)
    direction_path = args.output_prefix.with_suffix(".teacher_direction.pt")
    torch.save(
        {
            "schema_version": 1,
            "pairing_sha256": direction.pairing_sha256,
            "layers": list(direction.layers),
            "vectors": dict(direction.vectors),
            "norms": dict(direction.norms),
        },
        direction_path,
    )
    vanilla_direction_path = args.output_prefix.with_suffix(".vanilla_logit_lens_direction.pt")
    torch.save(
        {
            "schema_version": 1,
            "pairing_sha256": vanilla_direction.pairing_sha256,
            "layers": list(vanilla_direction.layers),
            "vectors": dict(vanilla_direction.vectors),
            "norms": dict(vanilla_direction.norms),
            "decoder_id": decoder.decoder_id,
        },
        vanilla_direction_path,
    )
    print(
        json.dumps(
            {
                "json_report": str(json_path),
                "csv_report": str(csv_path),
                "teacher_direction": str(direction_path),
                "teacher_direction_sha256": sha256_file(direction_path),
                "vanilla_logit_lens_direction": str(vanilla_direction_path),
                "vanilla_logit_lens_direction_sha256": sha256_file(vanilla_direction_path),
                "semantic_direction_gate": semantic_gate,
            },
            sort_keys=True,
        )
    )


def calibrate_command(args: argparse.Namespace) -> None:
    base = load_collected_readouts(args.base)
    variants = {name: load_collected_readouts(path) for name, path in args.variant}
    model = load_model(args)
    decoder = FixedBaseDecoder.from_hf_model(
        model,
        decoder_id=(f"{args.model_id}@{args.model_revision}+attn:{args.attn_implementation}"),
        deep_copy=False,
        device=args.device,
    )
    calibration = calibrate_fixed_lens_transport(
        base,
        variants,
        decoder,
        split=args.split,
        absolute_tolerance_nats=args.absolute_tolerance_nats,
        relative_tolerance=args.relative_tolerance,
        row_batch_size=args.decoder_batch_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(calibration.as_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "checkpoint",
            "layer",
            "lens_kind",
            "mean_kl",
            "sd_kl",
            "max_kl",
            "eligible",
            "allowed_mean_kl",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for checkpoint, result in calibration.checkpoints.items():
            for lens_kind in ("jlens", "logit_lens"):
                for layer, distance in getattr(result, lens_kind).items():
                    eligibility = calibration.layers[layer]
                    writer.writerow(
                        {
                            "checkpoint": checkpoint,
                            "layer": layer,
                            "lens_kind": lens_kind,
                            "mean_kl": distance.mean_kl,
                            "sd_kl": distance.sd_kl,
                            "max_kl": distance.max_kl,
                            "eligible": eligibility.eligible,
                            "allowed_mean_kl": eligibility.allowed_mean_kl,
                        }
                    )
    print(
        json.dumps(
            {
                "json_report": str(args.output_json),
                "csv_report": str(args.output_csv),
                "eligible_layers": list(calibration.eligible_layers),
                "qualification": (
                    "output-fidelity screen; not proof of checkpoint-local Jacobian validity"
                ),
            },
            sort_keys=True,
        )
    )


def teacher_gate_command(args: argparse.Namespace) -> None:
    teacher = load_collected_readouts(args.teacher_treatment)
    control = load_collected_readouts(args.teacher_control)
    calibration_direction = estimate_teacher_direction(
        teacher,
        control,
        source_split=args.calibration_split,
        coordinate="jspace",
        layers=args.layers,
        alignment_mode=args.alignment_mode,
    )
    validation_direction = estimate_teacher_direction(
        teacher,
        control,
        source_split=args.validation_split,
        coordinate="jspace",
        layers=args.layers,
        alignment_mode=args.alignment_mode,
    )
    gate = evaluate_teacher_state_holdout(
        calibration_direction,
        validation_direction,
        required_positive_layers=args.minimum_positive_layers,
        expected_layer_count=len(args.layers),
        minimum_median_cosine=args.minimum_median_cosine,
    )
    if bool(args.carrier_treatment) != bool(args.carrier_control):
        raise ValueError("carrier treatment and control readouts must be supplied together")
    carrier_gate = None
    if args.carrier_treatment is not None:
        carrier_gate = carrier_state_persistence_gate(
            load_collected_readouts(args.carrier_treatment),
            load_collected_readouts(args.carrier_control),
            calibration_direction,
            split=args.carrier_split,
            minimum_positive_layers=args.carrier_minimum_positive_layers,
        )
    overall_passed = gate.passed and (carrier_gate is None or bool(carrier_gate["passed"]))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    direction_path = args.output_prefix.with_suffix(".teacher_direction.pt")
    torch.save(
        {
            "schema_version": 1,
            "gate": "H3",
            "source_split": calibration_direction.source_split,
            "pairing_sha256": calibration_direction.pairing_sha256,
            "layers": list(calibration_direction.layers),
            "vectors": dict(calibration_direction.vectors),
            "norms": dict(calibration_direction.norms),
            "lens_provenance_id": calibration_direction.lens_provenance_id,
            "lens_artifact_sha256": calibration_direction.lens_artifact_sha256,
        },
        direction_path,
    )
    direction_sha256 = sha256_file(direction_path)
    report_path = args.output_prefix.with_suffix(".h3_gate.json")
    report = {
        **gate.as_dict(),
        "teacher_state_passed": gate.passed,
        "carrier_state_gate": carrier_gate,
        "passed": overall_passed,
        "teacher_direction_path": str(direction_path),
        "teacher_direction_sha256": direction_sha256,
        "calibration_direction": calibration_direction.as_dict(include_vectors=False),
        "validation_direction": validation_direction.as_dict(include_vectors=False),
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gate": "H3",
                "passed": overall_passed,
                "positive_layers": gate.positive_layers,
                "required_positive_layers": gate.required_positive_layers,
                "minimum_median_cosine": gate.minimum_median_cosine,
                "median_cosine": gate.median_cosine,
                "carrier_state_gate": carrier_gate,
                "teacher_direction": str(direction_path),
                "teacher_direction_sha256": direction_sha256,
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=GEMMA_2_9B_IT_PUBLIC_JLENS.model_repo)
    parser.add_argument("--model-revision", default=GEMMA_2_9B_IT_PUBLIC_JLENS.model_revision)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--lens-provenance",
        type=Path,
        help="JSON provenance object or readout protocol containing lens_provenance",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect one checkpoint/context arm")
    add_model_arguments(collect)
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--model-label", required=True)
    collect.add_argument("--checkpoint-revision")
    collect.add_argument("--adapter")
    collect.add_argument("--layers", type=comma_ints, required=True)
    collect.add_argument("--cache-dir", type=Path)
    collect.add_argument("--lens-device")
    collect.add_argument(
        "--lens-dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    collect.add_argument("--row-batch-size", type=int, default=32)
    collect.add_argument("--output", type=Path, required=True)
    collect.set_defaults(function=collect_command)

    project = subparsers.add_parser("project", help="project paired students teacherward")
    add_model_arguments(project)
    project.add_argument("--teacher-treatment", type=Path, required=True)
    project.add_argument("--teacher-control", type=Path, required=True)
    project.add_argument("--teacher-direction-artifact", type=Path, required=True)
    project.add_argument(
        "--student-pair",
        nargs=3,
        action="append",
        metavar=("SEED", "TREATMENT_PT", "CONTROL_PT"),
        required=True,
    )
    project.add_argument("--layers", type=comma_ints)
    project.add_argument(
        "--alignment-mode", choices=("strict", "paired_context"), default="paired_context"
    )
    project.add_argument("--source-split", default="teacher_direction")
    project.add_argument("--teacher-validation-split", default="teacher_validation")
    project.add_argument("--evaluation-split", default="student_evaluation")
    project.add_argument("--minimum-positive-layers", type=int, default=4)
    project.add_argument("--minimum-median-cosine", type=float, default=0.0)
    project.add_argument(
        "--primary-metric",
        choices=(
            "teacherward_projection",
            "fraction_of_teacher_delta",
            "cosine_to_teacher",
        ),
        default="teacherward_projection",
    )
    project.add_argument("--contrast-name", default="trait")
    project.add_argument("--positive-token-id", type=int, action="append")
    project.add_argument("--negative-token-id", type=int, action="append")
    project.add_argument(
        "--semantic-contrast-protocol",
        type=Path,
        help="readout protocol containing preregistered semantic contrast token ids",
    )
    project.add_argument("--decoder-batch-size", type=int, default=8)
    project.add_argument("--run-id", required=True)
    project.add_argument("--output-prefix", type=Path, required=True)
    project.set_defaults(function=project_command)

    calibrate = subparsers.add_parser(
        "calibrate", help="screen frozen-lens output fidelity across checkpoints"
    )
    add_model_arguments(calibrate)
    calibrate.add_argument("--base", type=Path, required=True)
    calibrate.add_argument(
        "--variant",
        nargs=2,
        action="append",
        metavar=("NAME", "READOUT_PT"),
        required=True,
    )
    calibrate.add_argument("--split", default="transport_calibration")
    calibrate.add_argument("--absolute-tolerance-nats", type=float, default=0.05)
    calibrate.add_argument("--relative-tolerance", type=float, default=0.25)
    calibrate.add_argument("--decoder-batch-size", type=int, default=8)
    calibrate.add_argument("--output-json", type=Path, required=True)
    calibrate.add_argument("--output-csv", type=Path, required=True)
    calibrate.set_defaults(function=calibrate_command)

    teacher_gate = subparsers.add_parser(
        "teacher-gate", help="run the fixed holdout H3 gate before student training"
    )
    teacher_gate.add_argument("--teacher-treatment", type=Path, required=True)
    teacher_gate.add_argument("--teacher-control", type=Path, required=True)
    teacher_gate.add_argument("--layers", type=comma_ints, required=True)
    teacher_gate.add_argument("--calibration-split", default="teacher_direction")
    teacher_gate.add_argument("--validation-split", default="teacher_validation")
    teacher_gate.add_argument("--carrier-treatment", type=Path)
    teacher_gate.add_argument("--carrier-control", type=Path)
    teacher_gate.add_argument("--carrier-split", default="carrier_state")
    teacher_gate.add_argument("--carrier-minimum-positive-layers", type=int, default=4)
    teacher_gate.add_argument("--minimum-positive-layers", type=int, default=4)
    teacher_gate.add_argument("--minimum-median-cosine", type=float, default=0.0)
    teacher_gate.add_argument(
        "--alignment-mode",
        choices=("strict", "paired_context"),
        default="paired_context",
    )
    teacher_gate.add_argument("--output-prefix", type=Path, required=True)
    teacher_gate.set_defaults(function=teacher_gate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
