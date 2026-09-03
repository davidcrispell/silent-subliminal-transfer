#!/usr/bin/env python3
"""Audit a ten-pass cell's imported Adam state and its epoch-5/10 checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from silent_transfer.training import verify_saved_training_identity

REQUIRED_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}
IMPORT_MANIFEST_NAME = "continuation_import.json"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def checkpoint_file_hashes(checkpoint: str | Path) -> dict[str, str]:
    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing Trainer checkpoint directory: {root}")
    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint must not contain symlinks: {path}")
        if path.is_file():
            if path.stat().st_size <= 0:
                raise ValueError(f"Checkpoint contains an empty file: {path}")
            paths.append(path)
    relative = {path.relative_to(root).as_posix(): path for path in paths}
    missing = REQUIRED_CHECKPOINT_FILES - set(relative)
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {root} is not optimizer-resumable; missing {sorted(missing)!r}"
        )
    return {name: sha256_file(path) for name, path in relative.items()}


def _unsafe_globals(path: Path) -> set[str]:
    import torch

    inspector = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if inspector is None:
        raise RuntimeError("This verifier requires torch serialization global inspection")
    return set(inspector(path))


def _safe_torch_load(path: Path, *, allowed_globals: list[Any] | None = None) -> Any:
    import torch

    allowed_globals = allowed_globals or []
    allowed_names = {
        f"{value.__module__}.{value.__qualname__}" for value in allowed_globals
    }
    unexpected = _unsafe_globals(path) - allowed_names
    if unexpected:
        raise ValueError(f"Unsafe or unexpected pickle globals in {path}: {sorted(unexpected)!r}")
    with torch.serialization.safe_globals(allowed_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def _safe_load_training_args(path: Path):
    from accelerate.state import PartialState
    from accelerate.utils.dataclasses import DistributedType
    from transformers import TrainingArguments
    from transformers.trainer_pt_utils import AcceleratorConfig
    from transformers.trainer_utils import (
        HubStrategy,
        IntervalStrategy,
        SaveStrategy,
        SchedulerType,
    )
    from transformers.training_args import OptimizerNames

    return _safe_torch_load(
        path,
        allowed_globals=[
            TrainingArguments,
            SchedulerType,
            OptimizerNames,
            HubStrategy,
            SaveStrategy,
            IntervalStrategy,
            AcceleratorConfig,
            PartialState,
            DistributedType,
        ],
    )


def _safe_load_rng(path: Path) -> dict[str, Any]:
    import numpy as np

    reconstruct = np._core.multiarray._reconstruct  # type: ignore[attr-defined]
    value = _safe_torch_load(
        path,
        allowed_globals=[
            np.dtype,
            np.ndarray,
            reconstruct,
            type(np.dtype(np.uint32)),
        ],
    )
    if not isinstance(value, dict):
        raise TypeError(f"RNG checkpoint is not a mapping: {path}")
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _expected_lr(*, step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        multiplier = step / max(1, warmup)
    else:
        multiplier = max(0.0, (total - step) / max(1, total - warmup))
    return base_lr * multiplier


def _audit_scheduler(
    path: Path,
    *,
    step: int,
    training: dict[str, Any],
) -> tuple[dict[str, Any], list[float]]:
    state = _safe_torch_load(path)
    if not isinstance(state, dict):
        raise TypeError(f"Scheduler state is not a mapping: {path}")
    if int(state.get("last_epoch", -1)) != step:
        raise ValueError(f"Scheduler last_epoch does not equal checkpoint step {step}")
    if int(state.get("_step_count", -1)) != step + 1:
        raise ValueError(f"Scheduler _step_count does not equal {step + 1}")
    base_lrs = [float(value) for value in state.get("base_lrs", [])]
    last_lrs = [float(value) for value in state.get("_last_lr", [])]
    if not base_lrs or len(base_lrs) != len(last_lrs):
        raise ValueError("Scheduler LR groups are empty or inconsistent")
    configured_lr = float(training["learning_rate"])
    if any(not math.isclose(value, configured_lr, rel_tol=0.0, abs_tol=1e-15) for value in base_lrs):
        raise ValueError("Scheduler base LR differs from the frozen learning rate")
    expected = _expected_lr(
        step=step,
        base_lr=configured_lr,
        warmup=int(training["warmup_steps"]),
        total=int(training["scheduler_total_steps"]),
    )
    if any(not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15) for value in last_lrs):
        raise ValueError(f"Scheduler LR at step {step} is not on the frozen 6,250-step curve")
    return (
        {
            "last_epoch": step,
            "step_count": step + 1,
            "base_lrs": base_lrs,
            "last_lrs": last_lrs,
        },
        last_lrs,
    )


def _step_number(value: Any) -> int:
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError("Adam step counter must be scalar")
        value = value.item()
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Invalid Adam step counter: {value!r}")
    return int(number)


def _audit_optimizer(
    path: Path,
    *,
    step: int,
    training: dict[str, Any],
    scheduler_lrs: list[float],
) -> dict[str, Any]:
    state_dict = _safe_torch_load(path)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Optimizer state is not a mapping: {path}")
    state = state_dict.get("state")
    groups = state_dict.get("param_groups")
    if not isinstance(state, dict) or not state:
        raise ValueError("Adam optimizer state is empty")
    if not isinstance(groups, list) or len(groups) != len(scheduler_lrs):
        raise ValueError("Adam parameter groups do not match scheduler LR groups")

    parameter_ids: list[Any] = []
    group_summary: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise TypeError("Malformed Adam parameter group")
        parameter_ids.extend(group["params"])
        lr = float(group.get("lr", float("nan")))
        if not math.isclose(lr, scheduler_lrs[index], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Adam group LR does not match scheduler state")
        if tuple(group.get("betas", ())) != (0.9, 0.999):
            raise ValueError("Adam beta values differ from the frozen optimizer")
        if not math.isclose(float(group.get("eps", float("nan"))), 1e-8, rel_tol=0, abs_tol=1e-16):
            raise ValueError("Adam epsilon differs from the frozen optimizer")
        if not math.isclose(
            float(group.get("weight_decay", float("nan"))),
            float(training.get("weight_decay", 0.0)),
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("Adam weight decay differs from the frozen optimizer")
        group_summary.append(
            {
                "parameter_count": len(group["params"]),
                "lr": lr,
                "betas": list(group["betas"]),
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(state):
        raise ValueError("Adam state keys do not exactly match the parameter groups")

    step_values: set[int] = set()
    moment_tensors = 0
    moment_elements = 0
    for parameter_state in state.values():
        if not isinstance(parameter_state, dict):
            raise TypeError("Malformed per-parameter Adam state")
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(parameter_state):
            raise ValueError("Adam state is missing step or moment tensors")
        step_values.add(_step_number(parameter_state["step"]))
        first = parameter_state["exp_avg"]
        second = parameter_state["exp_avg_sq"]
        if not hasattr(first, "shape") or not hasattr(second, "shape") or first.shape != second.shape:
            raise ValueError("Adam first/second moments have inconsistent shapes")
        moment_tensors += 2
        moment_elements += int(first.numel()) + int(second.numel())
    if step_values != {step}:
        raise ValueError(f"Adam state is not uniformly at optimizer step {step}: {step_values!r}")
    result = {
        "state_entries": len(state),
        "step": step,
        "moment_tensors": moment_tensors,
        "moment_elements": moment_elements,
        "parameter_groups": group_summary,
    }
    del state_dict
    gc.collect()
    return result


def _audit_rng(path: Path) -> dict[str, Any]:
    import torch

    state = _safe_load_rng(path)
    required = {"python", "numpy", "cpu", "cuda"}
    if not required <= set(state):
        raise ValueError(f"RNG state is missing {sorted(required - set(state))!r}")
    for name in ("cpu", "cuda"):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 0:
            raise ValueError(f"RNG {name} state is not a nonempty tensor")
    if not isinstance(state["python"], tuple) or not isinstance(state["numpy"], tuple):
        raise TypeError("Python/NumPy RNG states are malformed")
    return {
        "keys": sorted(state),
        "cpu_state_bytes": int(state["cpu"].numel()),
        "cuda_state_bytes": int(state["cuda"].numel()),
    }


def _audit_training_args(
    path: Path,
    *,
    training: dict[str, Any],
    seed: int,
    expected_output_suffix: str,
) -> dict[str, Any]:
    args = _safe_load_training_args(path)
    expected = {
        "max_steps": int(training["max_steps"]),
        "num_train_epochs": float(training["epochs"]),
        "per_device_train_batch_size": int(training["batch_size"]),
        "per_device_eval_batch_size": int(training["eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_steps": int(training["warmup_steps"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "optim": str(training["optimizer"]),
        "logging_steps": int(training["logging_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "seed": seed,
        "data_seed": seed,
        "bf16": True,
        "fp16": False,
        "tf32": bool(training["tf32"]),
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "save_only_model": False,
    }
    observed: dict[str, Any] = {}
    for name, wanted in expected.items():
        value = _enum_value(getattr(args, name))
        if isinstance(wanted, float):
            matches = math.isclose(float(value), wanted, rel_tol=0.0, abs_tol=1e-15)
        else:
            matches = value == wanted
        if not matches:
            raise ValueError(f"TrainingArguments.{name} mismatch: {value!r} != {wanted!r}")
        observed[name] = value
    if _enum_value(args.lr_scheduler_type) != "linear":
        raise ValueError("TrainingArguments scheduler is not linear")
    if _enum_value(args.save_strategy) != "epoch" or _enum_value(args.eval_strategy) != "epoch":
        raise ValueError("TrainingArguments must save and evaluate by epoch")
    accelerator = args.accelerator_config
    if (
        accelerator.use_seedable_sampler is not True
        or accelerator.split_batches is not False
        or accelerator.even_batches is not True
    ):
        raise ValueError("TrainingArguments changed the deterministic sampler geometry")
    output = Path(str(args.output_dir)).as_posix()
    if not output.endswith(Path(expected_output_suffix).as_posix()):
        raise ValueError(f"TrainingArguments output directory has the wrong logical suffix: {output}")
    observed.update(
        {
            "lr_scheduler_type": "linear",
            "save_strategy": "epoch",
            "eval_strategy": "epoch",
            "output_dir": output,
            "use_seedable_sampler": True,
        }
    )
    return observed


def _audit_adapter_config(path: Path, training: dict[str, Any]) -> dict[str, Any]:
    observed = _read_json(path, "adapter config")
    lora = training["lora"]
    expected = {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "use_rslora": bool(lora["use_rslora"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    for name, wanted in expected.items():
        if observed.get(name) != wanted:
            raise ValueError(f"Adapter config mismatch for {name}")
    if set(observed.get("target_modules", [])) != set(lora["target_modules"]):
        raise ValueError("Adapter target modules differ from the frozen LoRA recipe")
    return {**expected, "target_modules": sorted(observed["target_modules"])}


def audit_checkpoint(
    checkpoint: str | Path,
    *,
    step: int,
    epoch: int,
    training: dict[str, Any],
    seed: int,
    expected_output_suffix: str,
) -> dict[str, Any]:
    root = Path(checkpoint)
    file_hashes = checkpoint_file_hashes(root)
    trainer_state_path = root / "trainer_state.json"
    trainer_state = _read_json(trainer_state_path, "Trainer state")
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError(f"Checkpoint {step} has the wrong Trainer global step")
    if not math.isclose(float(trainer_state.get("epoch", -1)), float(epoch), abs_tol=1e-12):
        raise ValueError(f"Checkpoint {step} has the wrong epoch")
    if int(trainer_state.get("max_steps", -1)) != int(training["max_steps"]):
        raise ValueError(f"Checkpoint {step} has the wrong Trainer max_steps")
    if int(trainer_state.get("num_train_epochs", -1)) != int(training["epochs"]):
        raise ValueError(f"Checkpoint {step} has the wrong Trainer epoch horizon")
    if int(trainer_state.get("train_batch_size", -1)) != int(training["batch_size"]):
        raise ValueError(f"Checkpoint {step} has the wrong Trainer batch size")

    scheduler, scheduler_lrs = _audit_scheduler(
        root / "scheduler.pt", step=step, training=training
    )
    optimizer = _audit_optimizer(
        root / "optimizer.pt",
        step=step,
        training=training,
        scheduler_lrs=scheduler_lrs,
    )
    return {
        "file_sha256": file_hashes,
        "adapter_artifact_sha256": adapter_artifact_hashes(root),
        "adapter_config": _audit_adapter_config(root / "adapter_config.json", training),
        "trainer_state_sha256": sha256_file(trainer_state_path),
        "trainer_state": {
            "global_step": step,
            "epoch": float(trainer_state["epoch"]),
            "max_steps": int(trainer_state["max_steps"]),
            "num_train_epochs": int(trainer_state["num_train_epochs"]),
            "train_batch_size": int(trainer_state["train_batch_size"]),
        },
        "scheduler": scheduler,
        "optimizer": optimizer,
        "rng": _audit_rng(root / "rng_state.pth"),
        "training_args": _audit_training_args(
            root / "training_args.bin",
            training=training,
            seed=seed,
            expected_output_suffix=expected_output_suffix,
        ),
    }


def audit_source_cell(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
    source_cell_root: str | Path | None = None,
) -> dict[str, Any]:
    if condition not in {"control", "treatment"}:
        raise ValueError("condition must be control or treatment")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    continuation = raw["continuation_provenance"]
    if seed not in raw["seeds"]["students"]:
        raise ValueError(f"Unregistered student seed: {seed}")
    parent_path = Path(continuation["source_config"])
    if not parent_path.is_absolute():
        parent_path = repo / parent_path
    parent = load_config(parent_path)
    parent_config_sha = sha256_value(parent)
    if parent_config_sha != continuation["source_config_sha256"]:
        raise ValueError("Parent config SHA mismatch")

    if source_cell_root is None:
        source_run = Path(continuation["source_run_root"])
        if not source_run.is_absolute():
            source_run = repo / source_run
        source_cell = source_run / "models" / "students" / condition / f"seed-{seed}"
    else:
        source_cell = Path(source_cell_root).resolve()
    pins = continuation["expected_cells"][f"{condition}-{seed}"]
    pinned_files = {
        "dose_checkpoint_manifest.json": pins["checkpoint_manifest_sha256"],
        "training_complete.json": pins["training_complete_sha256"],
        "resume_identity.json": pins["resume_identity_sha256"],
    }
    for name, expected_hash in pinned_files.items():
        path = source_cell / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Pinned parent artifact mismatch: {path}")

    identity = _read_json(source_cell / "resume_identity.json", "parent resume identity")
    expected_train_hash = raw["dose_provenance"]["source_artifact_sha256"][
        f"paired/{condition}_train.jsonl"
    ]
    expected_eval_hash = raw["dose_provenance"]["source_artifact_sha256"][
        f"paired/{condition}_eval.jsonl"
    ]
    expected_identity = {
        "schema_version": 1,
        "config_sha256": parent_config_sha,
        "model": parent["model"],
        "seed": seed,
        "train_data_sha256": expected_train_hash,
        "eval_data_sha256": expected_eval_hash,
        "training_config_sha256": sha256_value(parent["training"]["student"]),
    }
    if identity != expected_identity:
        raise ValueError("Parent resume identity does not match the frozen cell")

    completion = _read_json(source_cell / "training_complete.json", "parent completion")
    if completion.get("training_identity_sha256") != sha256_value(identity):
        raise ValueError("Parent completion does not bind its resume identity")
    checkpoint_manifest = _read_json(
        source_cell / "dose_checkpoint_manifest.json", "parent checkpoint manifest"
    )
    checkpoint_step = int(continuation["checkpoint_step"])
    checkpoint_record = checkpoint_manifest.get("checkpoints", {}).get(str(checkpoint_step))
    if not isinstance(checkpoint_record, dict):
        raise TypeError("Parent checkpoint manifest does not bind step 625")
    expected_adapter = {
        "adapter_config.json": checkpoint_record["adapter_artifact_sha256"][
            "adapter_config.json"
        ],
        "adapter_model.safetensors": pins["adapter_model_sha256"],
    }
    if checkpoint_record.get("adapter_artifact_sha256") != expected_adapter:
        raise ValueError("Parent checkpoint adapter hashes do not match the frozen pins")
    if completion.get("adapter_artifact_sha256") != expected_adapter:
        raise ValueError("Parent final adapter and checkpoint-625 are not identical")
    if checkpoint_record.get("trainer_state_sha256") != pins["trainer_state_sha256"]:
        raise ValueError("Parent checkpoint Trainer-state hash mismatch")
    if (
        checkpoint_manifest.get("config_sha256") != parent_config_sha
        or checkpoint_manifest.get("run_id") != continuation["source_run_id"]
        or checkpoint_manifest.get("condition") != condition
        or int(checkpoint_manifest.get("seed", -1)) != seed
        or checkpoint_manifest.get("train_data_sha256") != expected_train_hash
        or checkpoint_manifest.get("eval_data_sha256") != expected_eval_hash
    ):
        raise ValueError("Parent checkpoint manifest identity mismatch")

    for relative, expected_hash in completion.get("stage_artifact_sha256", {}).items():
        path = source_cell / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Parent completion-stage artifact mismatch: {path}")

    checkpoint = source_cell / "trainer" / f"checkpoint-{checkpoint_step}"
    logical_suffix = (
        Path(continuation["source_run_root"])
        / "models"
        / "students"
        / condition
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    audit = audit_checkpoint(
        checkpoint,
        step=checkpoint_step,
        epoch=int(continuation["checkpoint_epoch"]),
        training=parent["training"]["student"],
        seed=seed,
        expected_output_suffix=logical_suffix,
    )
    if audit["adapter_artifact_sha256"] != expected_adapter:
        raise ValueError("Parent checkpoint bytes do not match its manifest")
    return {
        "source_cell": str(source_cell),
        "checkpoint": str(checkpoint),
        "parent_config_sha256": parent_config_sha,
        "parent_artifact_sha256": pinned_files,
        "checkpoint_audit": audit,
    }


def verify_tenpass_checkpoint_cell(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    if condition not in {"control", "treatment"}:
        raise ValueError("condition must be control or treatment")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    config_sha = sha256_value(raw)
    config["_protocol_config_sha256"] = config_sha
    if seed not in config["seeds"]["students"]:
        raise ValueError(f"Unregistered student seed: {seed}")

    run_root = Path(config["experiment"]["run_root"])
    output = run_root / "models" / "students" / condition / f"seed-{seed}"
    train_path = run_root / "data" / "paired" / f"{condition}_train.jsonl"
    eval_path = run_root / "data" / "paired" / f"{condition}_eval.jsonl"
    training = config["training"]["student"]
    verify_saved_training_identity(
        output,
        config=config,
        training_config=training,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )

    import_path = output / IMPORT_MANIFEST_NAME
    imported = _read_json(import_path, "continuation import manifest")
    continuation = raw["continuation_provenance"]

    parent_config_path = Path(continuation["source_config"])
    if not parent_config_path.is_absolute():
        parent_config_path = repo / parent_config_path
    parent = load_config(parent_config_path)
    imported_checkpoint = output / "trainer" / f"checkpoint-{continuation['checkpoint_step']}"
    source_suffix = (
        Path(continuation["source_run_root"])
        / "models"
        / "students"
        / condition
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    imported_audit = audit_checkpoint(
        imported_checkpoint,
        step=int(continuation["checkpoint_step"]),
        epoch=int(continuation["checkpoint_epoch"]),
        training=parent["training"]["student"],
        seed=seed,
        expected_output_suffix=source_suffix,
    )
    pins = continuation["expected_cells"][f"{condition}-{seed}"]
    expected_import = {
        "schema_version": 1,
        "destination_config_sha256": config_sha,
        "destination_run_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "source_config_sha256": continuation["source_config_sha256"],
        "source_git_commit": continuation["source_git_commit"],
        "source_run_id": continuation["source_run_id"],
        "checkpoint_step": int(continuation["checkpoint_step"]),
        "checkpoint_epoch": int(continuation["checkpoint_epoch"]),
        "parent_artifact_sha256": {
            "dose_checkpoint_manifest.json": pins["checkpoint_manifest_sha256"],
            "training_complete.json": pins["training_complete_sha256"],
            "resume_identity.json": pins["resume_identity_sha256"],
        },
        "source_checkpoint_file_sha256": imported_audit["file_sha256"],
        "destination_checkpoint_file_sha256": imported_audit["file_sha256"],
        "reuse_method": "independent_atomic_byte_copy",
    }
    if imported != expected_import:
        raise ValueError("Continuation import manifest identity or bytes mismatch")

    metrics_path = output / "training_metrics.json"
    metrics = _read_json(metrics_path, "training metrics")
    target_steps = int(raw["dose_provenance"]["target_optimizer_steps"])
    expected_geometry = {
        key: value for key, value in raw["batch_geometry"].items() if key != "mode"
    }
    if (
        int(metrics.get("optimizer_steps", -1)) != target_steps
        or int(metrics.get("configured_max_steps", -1)) != target_steps
        or int(metrics.get("scheduler_total_steps", -1)) != target_steps
        or int(metrics.get("configured_warmup_steps", -1)) != 8
        or int(metrics.get("train_examples", -1)) != int(config["carrier"]["train_size"])
        or metrics.get("batch_geometry") != expected_geometry
    ):
        raise ValueError("Ten-pass training metrics do not match the frozen geometry")

    checkpoints: dict[str, Any] = {}
    steps = raw["dose_provenance"]["probe_optimizer_steps"]
    epochs = raw["dose_provenance"]["probe_epochs"]
    if len(steps) != len(epochs):
        raise ValueError("Probe steps and epochs are not aligned")
    logical_suffix = (
        Path(raw["experiment"]["run_root"])
        / "models"
        / "students"
        / condition
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    for step_value, epoch_value in zip(steps, epochs):
        step = int(step_value)
        audit = audit_checkpoint(
            output / "trainer" / f"checkpoint-{step}",
            step=step,
            epoch=int(epoch_value),
            training=training,
            seed=seed,
            expected_output_suffix=logical_suffix,
        )
        checkpoints[str(step)] = audit

    final_adapter_hashes = adapter_artifact_hashes(output / "final_adapter")
    if checkpoints[str(target_steps)]["adapter_artifact_sha256"] != final_adapter_hashes:
        raise ValueError("Epoch-10 checkpoint does not match the published final adapter")
    result = {
        "schema_version": 2,
        "config_sha256": config_sha,
        "run_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "training_complete_sha256": sha256_file(output / "training_complete.json"),
        "resume_identity_sha256": sha256_file(output / "resume_identity.json"),
        "training_metrics_sha256": sha256_file(metrics_path),
        "continuation_import_sha256": sha256_file(import_path),
        "continuation_checkpoint": imported_audit,
        "final_adapter_artifact_sha256": final_adapter_hashes,
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path),
        "checkpoints": checkpoints,
    }
    write_json_atomic(output / "dose_checkpoint_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("condition", choices=("control", "treatment"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = verify_tenpass_checkpoint_cell(
        args.config,
        args.condition,
        args.seed,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
