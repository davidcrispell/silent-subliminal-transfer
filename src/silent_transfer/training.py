from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .masking import CompletionCollator, CompletionDataset
from .modeling import load_model, load_tokenizer, seed_everything, trainable_parameter_summary
from .provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
    write_manifest,
)
from .training_geometry import training_batch_geometry


class IncompleteTrainingRunError(RuntimeError):
    """Raised when a final adapter exists without a verified completion record."""


def training_identity(
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    train_path: str | Path,
    eval_path: str | Path | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "model": config["model"],
        "seed": seed,
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path) if eval_path is not None else None,
        "training_config_sha256": sha256_value(training_config),
    }


def verify_saved_training_identity(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    train_path: str | Path,
    eval_path: str | Path | None,
    seed: int,
) -> None:
    output = Path(output_dir)
    identity_path = output / "resume_identity.json"
    if not identity_path.exists():
        raise IncompleteTrainingRunError(f"Completed adapter is missing {identity_path}")
    existing = json.loads(identity_path.read_text(encoding="utf-8"))
    expected = training_identity(
        config=config,
        training_config=training_config,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )
    if existing != expected:
        raise RuntimeError(f"Completed adapter identity mismatch at {identity_path}")
    completion_path = output / "training_complete.json"
    if not completion_path.is_file():
        raise IncompleteTrainingRunError(
            f"Adapter has no atomic completion record: {completion_path}"
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("training_identity_sha256") != sha256_value(expected):
            raise IncompleteTrainingRunError(
                f"Completion record does not bind the expected identity: {completion_path}"
            )
        actual_adapter = adapter_artifact_hashes(output / "final_adapter")
        if completion.get("adapter_artifact_sha256") != actual_adapter:
            raise IncompleteTrainingRunError(
                f"Completed adapter artifact hash mismatch: {output / 'final_adapter'}"
            )
        recorded_stage = completion.get("stage_artifact_sha256")
        if not isinstance(recorded_stage, dict) or not recorded_stage:
            raise IncompleteTrainingRunError(
                f"Completion record has no stage artifact hashes: {completion_path}"
            )
        for relative_path, expected_hash in recorded_stage.items():
            artifact = output / relative_path
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise IncompleteTrainingRunError(
                    f"Completed training artifact is absent or corrupt: {artifact}"
                )
    except (json.JSONDecodeError, OSError, FileNotFoundError) as error:
        raise IncompleteTrainingRunError(
            f"Could not validate completed adapter at {output}"
        ) from error


def discard_incomplete_final_adapter(output_dir: str | Path) -> None:
    """Remove only publish-stage artifacts so Trainer checkpoints remain resumable."""
    output = Path(output_dir)
    for directory in (output / "final_adapter", output / ".final_adapter.incomplete"):
        if directory.exists():
            shutil.rmtree(directory)
    completion = output / "training_complete.json"
    if completion.exists():
        completion.unlink()


def _add_lora(model, lora: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=list(lora["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=bool(lora["use_rslora"]),
        ),
    )
    return model


def train_adapter(
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    train_path: str | Path,
    output_dir: str | Path,
    seed: int,
    repo_root: str | Path,
    eval_path: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one completion-only LoRA adapter from the pinned base checkpoint."""
    import torch
    from transformers import Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Refusing a full model training run without CUDA")
    seed_everything(seed)
    output = Path(output_dir)
    trainer_output = output / "trainer"
    final_adapter = output / "final_adapter"
    staging_adapter = output / ".final_adapter.incomplete"
    completion_path = output / "training_complete.json"
    output.mkdir(parents=True, exist_ok=True)

    if final_adapter.exists():
        raise RuntimeError(
            f"Refusing to overwrite {final_adapter}; validate/reuse it or discard an "
            "incomplete publish stage first"
        )
    if staging_adapter.exists():
        shutil.rmtree(staging_adapter)
    if completion_path.exists():
        if not resume:
            raise RuntimeError(
                f"Stale completion record at {completion_path}; rerun with --resume"
            )
        completion_path.unlink()

    identity = training_identity(
        config=config,
        training_config=training_config,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )
    identity_path = output / "resume_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                f"Resume identity mismatch at {identity_path}; use a new immutable run directory"
            )
    else:
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    tokenizer = load_tokenizer(config["model"])
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path) if eval_path is not None else []
    train_dataset = CompletionDataset(train_rows, tokenizer, int(training_config["max_length"]))
    eval_dataset = (
        CompletionDataset(eval_rows, tokenizer, int(training_config["max_length"]))
        if eval_rows
        else None
    )

    model = load_model(config["model"])
    model = _add_lora(model, training_config["lora"])
    model.config.use_cache = False
    if training_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    use_bf16 = config["model"]["dtype"] == "bfloat16"
    use_fp16 = config["model"]["dtype"] == "float16"
    arguments = TrainingArguments(
        output_dir=str(trainer_output),
        overwrite_output_dir=not resume,
        num_train_epochs=float(training_config["epochs"]),
        max_steps=int(training_config.get("max_steps", -1)),
        per_device_train_batch_size=int(training_config["batch_size"]),
        per_device_eval_batch_size=int(
            training_config.get("eval_batch_size", training_config["batch_size"])
        ),
        gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        warmup_ratio=float(training_config["warmup_ratio"]),
        lr_scheduler_type="linear",
        max_grad_norm=float(training_config["max_grad_norm"]),
        optim=str(training_config["optimizer"]),
        logging_strategy="steps",
        logging_steps=int(training_config.get("logging_steps", 5)),
        save_strategy="epoch",
        save_total_limit=int(training_config.get("save_total_limit", 1)),
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=bool(training_config.get("tf32", True)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        dataloader_num_workers=int(training_config.get("dataloader_num_workers", 0)),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
    )
    checkpoint = None
    if resume and trainer_output.exists():
        checkpoints = sorted(
            trainer_output.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        checkpoint = str(checkpoints[-1]) if checkpoints else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    optimizer_steps = int(trainer.state.global_step)
    expected_steps = training_config.get("max_steps")
    if expected_steps is not None and optimizer_steps != int(expected_steps):
        raise RuntimeError(
            "Training stopped at an unexpected optimizer-step count: "
            f"expected {expected_steps}, observed {optimizer_steps}"
        )
    trainer.model.save_pretrained(staging_adapter, safe_serialization=True)
    tokenizer.save_pretrained(staging_adapter)
    adapter_artifact_hashes(staging_adapter)

    metrics = {
        **result.metrics,
        "seed": seed,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "parameters": trainable_parameter_summary(trainer.model),
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path) if eval_path is not None else None,
        "optimizer": str(training_config["optimizer"]),
        "optimizer_steps": optimizer_steps,
        "configured_max_steps": expected_steps,
        "batch_geometry": training_batch_geometry(len(train_dataset), training_config),
        "completion_only_loss": True,
    }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging_adapter.replace(final_adapter)
    artifacts = [path for path in final_adapter.rglob("*") if path.is_file()]
    artifacts.extend(
        [
            Path(train_path),
            output / "training_metrics.json",
            output / "log_history.json",
            identity_path,
        ]
    )
    if eval_path is not None:
        artifacts.append(Path(eval_path))
    manifest_path = output / "manifest.json"
    write_manifest(
        manifest_path,
        config=config,
        repo_root=repo_root,
        stage="train_adapter",
        artifacts=artifacts,
        extra={"seed": seed, "output": str(output), "metrics": metrics},
    )
    stage_paths = {
        "training_metrics.json": output / "training_metrics.json",
        "log_history.json": output / "log_history.json",
        "manifest.json": manifest_path,
    }
    write_json_atomic(
        completion_path,
        {
            "schema_version": 1,
            "training_identity_sha256": sha256_value(identity),
            "adapter_artifact_sha256": adapter_artifact_hashes(final_adapter),
            "stage_artifact_sha256": {
                name: sha256_file(path) for name, path in stage_paths.items()
            },
        },
    )
    verify_saved_training_identity(
        output,
        config=config,
        training_config=training_config,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )
    model.config.use_cache = True
    return metrics
