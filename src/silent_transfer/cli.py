from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .behavior import evaluate_free_response, summarize_paired_behavior
from .config import load_config, resolve_config
from .costs import append_cost
from .data import build_teacher_rows, read_jsonl, write_jsonl
from .generation import generate_condition, pair_and_split_carriers, prepare_prompt_bank
from .provenance import sha256_file, sha256_value
from .readout_handoff import export_readout_handoff
from .training import (
    IncompleteTrainingRunError,
    discard_incomplete_final_adapter,
    train_adapter,
    verify_saved_training_identity,
)


def _repo_root(args) -> Path:
    return Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()


def _config(args) -> tuple[dict[str, Any], Path]:
    repo_root = _repo_root(args)
    raw = load_config(args.config)
    resolved = resolve_config(raw, repo_root=repo_root)
    resolved["_protocol_config_sha256"] = sha256_value(raw)
    resolved["_protocol_config_path"] = str(Path(args.config).resolve())
    return resolved, repo_root


def _run_root(config: dict[str, Any]) -> Path:
    return Path(config["experiment"]["run_root"])


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    root = _run_root(config)
    return {
        "root": root,
        "prompts": root / "data" / "carrier_prompts.jsonl",
        "treatment_raw": root / "data" / "raw_treatment.jsonl",
        "control_raw": root / "data" / "raw_control.jsonl",
        "paired": root / "data" / "paired",
        "teacher_rows": root / "data" / "teacher_train.jsonl",
        "teacher_model": root / "models" / "wolf_teacher",
        "students": root / "models" / "students",
        "behavior": root / "evaluations" / "behavior",
    }


def cmd_validate(args) -> None:
    config, _ = _config(args)
    result = {
        "valid": True,
        "experiment_id": config["experiment"]["id"],
        "kind": config["experiment"]["kind"],
        "config_sha256": sha256_value(load_config(args.config)),
        "model": config["model"],
        "student_seeds": config["seeds"]["students"],
        "run_root": config["experiment"]["run_root"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_preflight(args) -> None:
    config, _ = _config(args)
    import torch

    run_root = _run_root(config)
    run_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(run_root)
    result: dict[str, Any] = {
        "config_valid": True,
        "model": config["model"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [],
        "disk_free_gib": disk.free / 1024**3,
        "hf_revision_verified": None,
    }
    if torch.cuda.is_available():
        result["gpus"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_gib": torch.cuda.get_device_properties(index).total_memory / 1024**3,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    if not args.skip_hf:
        from huggingface_hub import HfApi

        model = config["model"]
        info = HfApi().model_info(model["id"], revision=model["revision"])
        result["hf_revision_verified"] = info.sha == model["revision"]
        result["hf_resolved_sha"] = info.sha
        if not result["hf_revision_verified"]:
            raise RuntimeError("Hugging Face resolved a different model revision")
    required_free_gib = float(config["runtime"].get("minimum_disk_free_gib", 100))
    result["disk_gate_pass"] = result["disk_free_gib"] >= required_free_gib
    result["cuda_gate_pass"] = result["cuda_available"]
    destination = run_root / "preflight.json"
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["cuda_gate_pass"] or not result["disk_gate_pass"]:
        raise SystemExit(2)


def cmd_prepare_prompts(args) -> None:
    config, repo_root = _config(args)
    paths = _paths(config)
    destination = prepare_prompt_bank(
        config,
        output_path=paths["prompts"],
        repo_root=repo_root,
        force=args.force,
    )
    print(f"{destination} {sha256_file(destination)}")


def cmd_train_teacher(args) -> None:
    config, repo_root = _config(args)
    if config["experiment"]["kind"] != "wolf_sl":
        raise ValueError("train-teacher is only part of the standard wolf SL positive control")
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive for training")
    paths = _paths(config)
    rows = build_teacher_rows(
        config["teacher"]["target"],
        int(config["teacher"]["rows"]),
        int(config["seeds"]["teacher"]),
    )
    if paths["teacher_rows"].exists() and not args.force:
        if read_jsonl(paths["teacher_rows"]) != rows:
            raise RuntimeError(
                f"Existing teacher rows do not match the frozen bank: {paths['teacher_rows']}"
            )
    else:
        write_jsonl(paths["teacher_rows"], rows)
    final_adapter = paths["teacher_model"] / "final_adapter"
    if args.force:
        if paths["teacher_model"].exists():
            shutil.rmtree(paths["teacher_model"])
    elif final_adapter.exists():
        try:
            verify_saved_training_identity(
                paths["teacher_model"],
                config=config,
                training_config=config["training"]["teacher"],
                train_path=paths["teacher_rows"],
                eval_path=None,
                seed=int(config["seeds"]["teacher"]),
            )
        except IncompleteTrainingRunError:
            if not args.resume:
                raise
            discard_incomplete_final_adapter(paths["teacher_model"])
        else:
            print(f"Reusing {final_adapter}")
            return
    metrics = train_adapter(
        config=config,
        training_config=config["training"]["teacher"],
        train_path=paths["teacher_rows"],
        output_dir=paths["teacher_model"],
        seed=int(config["seeds"]["teacher"]),
        repo_root=repo_root,
        resume=args.resume,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_generate_condition(args) -> None:
    config, repo_root = _config(args)
    paths = _paths(config)
    prepare_prompt_bank(
        config,
        output_path=paths["prompts"],
        repo_root=repo_root,
        force=False,
    )
    stats = generate_condition(
        config,
        condition_name=args.condition,
        prompt_path=paths["prompts"],
        output_path=paths[f"{args.condition}_raw"],
        repo_root=repo_root,
        force=args.force,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


def cmd_pair_carriers(args) -> None:
    config, repo_root = _config(args)
    paths = _paths(config)
    stats = pair_and_split_carriers(
        config,
        treatment_path=paths["treatment_raw"],
        control_path=paths["control_raw"],
        output_dir=paths["paired"],
        repo_root=repo_root,
        force=args.force,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


def _assert_paired_training_data(paths: dict[str, Path]) -> None:
    treatment = read_jsonl(paths["paired"] / "treatment_train.jsonl")
    control = read_jsonl(paths["paired"] / "control_train.jsonl")
    if [row["pair_id"] for row in treatment] != [row["pair_id"] for row in control]:
        raise RuntimeError("Treatment/control student datasets are not pair-aligned")
    if [row["completion_token_count"] for row in treatment] != [
        row["completion_token_count"] for row in control
    ]:
        raise RuntimeError("Treatment/control student datasets do not match token exposure")


def _train_one_student(
    config: dict[str, Any],
    repo_root: Path,
    *,
    condition: str,
    seed: int,
    force: bool,
    resume: bool,
) -> dict[str, Any] | None:
    if force and resume:
        raise ValueError("--force and --resume are mutually exclusive for training")
    paths = _paths(config)
    if seed not in config["seeds"]["students"]:
        raise ValueError(f"Seed {seed} is not in the frozen paired seed registry")
    _assert_paired_training_data(paths)
    destination = paths["students"] / condition / f"seed-{seed}"
    if force:
        if destination.exists():
            shutil.rmtree(destination)
    elif (destination / "final_adapter").exists():
        try:
            verify_saved_training_identity(
                destination,
                config=config,
                training_config=config["training"]["student"],
                train_path=paths["paired"] / f"{condition}_train.jsonl",
                eval_path=paths["paired"] / f"{condition}_eval.jsonl",
                seed=seed,
            )
        except IncompleteTrainingRunError:
            if not resume:
                raise
            discard_incomplete_final_adapter(destination)
        else:
            print(f"Reusing {destination / 'final_adapter'}")
            return None
    return train_adapter(
        config=config,
        training_config=config["training"]["student"],
        train_path=paths["paired"] / f"{condition}_train.jsonl",
        eval_path=paths["paired"] / f"{condition}_eval.jsonl",
        output_dir=destination,
        seed=seed,
        repo_root=repo_root,
        resume=resume,
    )


def cmd_train_student(args) -> None:
    config, repo_root = _config(args)
    metrics = _train_one_student(
        config,
        repo_root,
        condition=args.condition,
        seed=args.seed,
        force=args.force,
        resume=args.resume,
    )
    if metrics is not None:
        print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_train_students(args) -> None:
    config, repo_root = _config(args)
    for seed in config["seeds"]["students"]:
        for condition in ("control", "treatment"):
            _train_one_student(
                config,
                repo_root,
                condition=condition,
                seed=seed,
                force=args.force,
                resume=args.resume,
            )


def cmd_behavior(args) -> None:
    config, repo_root = _config(args)
    summary = evaluate_free_response(
        config,
        label=args.label,
        output_dir=args.output,
        repo_root=repo_root,
        adapter_path=args.adapter,
        context_condition=args.context_condition,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_behavior_suite(args) -> None:
    config, repo_root = _config(args)
    paths = _paths(config)
    evaluate_free_response(
        config,
        label="base",
        output_dir=paths["behavior"] / "base",
        repo_root=repo_root,
        force=args.force,
    )
    if config["experiment"]["kind"] == "wolf_sl":
        evaluate_free_response(
            config,
            label="wolf_teacher",
            output_dir=paths["behavior"] / "teacher",
            repo_root=repo_root,
            adapter_path=config["conditions"]["treatment"]["adapter"],
            force=args.force,
        )
    for seed in config["seeds"]["students"]:
        for condition in ("control", "treatment"):
            adapter = paths["students"] / condition / f"seed-{seed}" / "final_adapter"
            evaluate_free_response(
                config,
                label=f"student_{condition}_seed_{seed}",
                output_dir=paths["behavior"] / "students" / condition / f"seed-{seed}",
                repo_root=repo_root,
                adapter_path=adapter,
                force=args.force,
            )
    summary = summarize_paired_behavior(
        config,
        behavior_root=paths["behavior"],
        output_path=paths["behavior"] / "paired_summary.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_record_cost(args) -> None:
    result = append_cost(
        args.ledger,
        run_id=args.run_id,
        stage=args.stage,
        provider=args.provider,
        instance_type=args.instance_type,
        gpu_count=args.gpu_count,
        gpu_hours=args.gpu_hours,
        rate_per_gpu_hour_usd=args.rate,
        storage_cost_usd=args.storage_cost,
        api_cost_usd=args.api_cost,
        other_cost_usd=args.other_cost,
        started_at_utc=args.started_at,
        ended_at_utc=args.ended_at,
        invoice_or_instance_id=args.invoice_id,
        notes=args.notes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_export_readout(args) -> None:
    config, repo_root = _config(args)
    paths = _paths(config)
    protocol = export_readout_handoff(
        config,
        output_dir=paths["root"] / "readout" / "specs",
        repo_root=repo_root,
        force=args.force,
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


def _add_common(subparser) -> None:
    subparser.add_argument("config", type=Path)
    subparser.add_argument("--repo-root", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="silent-transfer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("validate", help="Validate and hash a frozen config")
    _add_common(command)
    command.set_defaults(func=cmd_validate)

    command = subparsers.add_parser(
        "preflight", help="Check GPU, disk, HF access, and revision"
    )
    _add_common(command)
    command.add_argument("--skip-hf", action="store_true")
    command.set_defaults(func=cmd_preflight)

    command = subparsers.add_parser(
        "prepare-prompts", help="Build the deterministic carrier bank"
    )
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_prepare_prompts)

    command = subparsers.add_parser("train-teacher", help="Train the standard wolf teacher")
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.add_argument("--resume", action="store_true")
    command.set_defaults(func=cmd_train_teacher)

    command = subparsers.add_parser(
        "generate-condition", help="Generate one paired carrier arm"
    )
    _add_common(command)
    command.add_argument("--condition", choices=("treatment", "control"), required=True)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_generate_condition)

    command = subparsers.add_parser("pair-carriers", help="Pair-filter and split student data")
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_pair_carriers)

    command = subparsers.add_parser(
        "train-student", help="Train one frozen condition/seed cell"
    )
    _add_common(command)
    command.add_argument("--condition", choices=("treatment", "control"), required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--force", action="store_true")
    command.add_argument("--resume", action="store_true")
    command.set_defaults(func=cmd_train_student)

    command = subparsers.add_parser("train-students", help="Train all three paired seed blocks")
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.add_argument("--resume", action="store_true")
    command.set_defaults(func=cmd_train_students)

    command = subparsers.add_parser(
        "behavior", help="Evaluate one checkpoint with free responses"
    )
    _add_common(command)
    command.add_argument("--label", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--adapter", type=Path)
    command.add_argument("--context-condition", choices=("treatment", "control"))
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_behavior)

    command = subparsers.add_parser(
        "behavior-suite", help="Evaluate base, teacher, and paired students"
    )
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_behavior_suite)

    command = subparsers.add_parser("record-cost", help="Append a compute/storage/API cost")
    command.add_argument("--ledger", type=Path, default=Path("runs/COST_LEDGER.csv"))
    command.add_argument("--run-id", required=True)
    command.add_argument("--stage", required=True)
    command.add_argument("--provider", default="Lambda")
    command.add_argument("--instance-type", required=True)
    command.add_argument("--gpu-count", type=int, default=1)
    command.add_argument("--gpu-hours", type=float, required=True)
    command.add_argument("--rate", type=float, required=True)
    command.add_argument("--storage-cost", type=float, default=0.0)
    command.add_argument("--api-cost", type=float, default=0.0)
    command.add_argument("--other-cost", type=float, default=0.0)
    command.add_argument("--started-at", default="")
    command.add_argument("--ended-at", default="")
    command.add_argument("--invoice-id", default="")
    command.add_argument("--notes", default="")
    command.set_defaults(func=cmd_record_cost)

    command = subparsers.add_parser(
        "export-readout", help="Freeze teacher/control/student prompt arms for sst_readout"
    )
    _add_common(command)
    command.add_argument("--force", action="store_true")
    command.set_defaults(func=cmd_export_readout)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
