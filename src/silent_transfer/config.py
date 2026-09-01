from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .training_geometry import verify_declared_batch_geometry


class ConfigError(ValueError):
    """Raised when an experiment config violates a frozen protocol invariant."""


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a nonempty string")
    return value


def _validate_revision(value: Any, path: str) -> None:
    revision = _nonempty(value, path)
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ConfigError(f"{path} must be a lowercase 40-character commit SHA")


def _validate_training(value: Any, path: str) -> None:
    training = _mapping(value, path)
    if training.get("optimizer") not in {"adamw_torch", "adamw_torch_fused"}:
        raise ConfigError(
            f"{path}.optimizer must be adaptive AdamW ('adamw_torch' or 'adamw_torch_fused')"
        )
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length"):
        _integer(training.get(key), f"{path}.{key}", minimum=1)
    if "max_steps" in training:
        _integer(training.get("max_steps"), f"{path}.max_steps", minimum=1)
    for key in ("learning_rate", "max_grad_norm"):
        number = training.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or number <= 0:
            raise ConfigError(f"{path}.{key} must be positive")
    warmup = training.get("warmup_ratio")
    if not isinstance(warmup, (int, float)) or not 0 <= warmup < 1:
        raise ConfigError(f"{path}.warmup_ratio must be in [0, 1)")
    lora = _mapping(training.get("lora"), f"{path}.lora")
    for key in ("r", "alpha"):
        _integer(lora.get(key), f"{path}.lora.{key}", minimum=1)
    targets = lora.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or not all(isinstance(x, str) for x in targets)
    ):
        raise ConfigError(f"{path}.lora.target_modules must be a nonempty string list")
    if lora.get("use_rslora") is not True:
        raise ConfigError(f"{path}.lora.use_rslora must be true")


def validate_config(config: Any) -> dict[str, Any]:
    cfg = _mapping(config, "config")
    if cfg.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    experiment = _mapping(cfg.get("experiment"), "experiment")
    kind = experiment.get("kind")
    if kind not in {"wolf_sl", "silent_carriers"}:
        raise ConfigError("experiment.kind must be 'wolf_sl' or 'silent_carriers'")
    _nonempty(experiment.get("id"), "experiment.id")
    _nonempty(experiment.get("run_root"), "experiment.run_root")

    model = _mapping(cfg.get("model"), "model")
    _nonempty(model.get("id"), "model.id")
    _validate_revision(model.get("revision"), "model.revision")
    _validate_revision(model.get("tokenizer_revision"), "model.tokenizer_revision")
    if model.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ConfigError("model.dtype must be bfloat16, float16, or float32")
    _nonempty(model.get("attn_implementation"), "model.attn_implementation")

    readout = _mapping(cfg.get("readout"), "readout")
    layers = readout.get("preregistered_layers")
    if not isinstance(layers, list) or not layers or len(layers) != len(set(layers)):
        raise ConfigError("readout.preregistered_layers must be a nonempty unique list")
    for index, layer in enumerate(layers):
        _integer(layer, f"readout.preregistered_layers[{index}]", minimum=0)
    transport = _mapping(readout.get("transport"), "readout.transport")
    for key in ("absolute_tolerance_nats", "relative_tolerance"):
        number = transport.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or number <= 0:
            raise ConfigError(f"readout.transport.{key} must be positive")
    _nonempty(transport.get("calibration_split"), "readout.transport.calibration_split")
    teacher_gate = _mapping(readout.get("teacher_gate"), "readout.teacher_gate")
    _nonempty(teacher_gate.get("calibration_split"), "readout.teacher_gate.calibration_split")
    _nonempty(teacher_gate.get("validation_split"), "readout.teacher_gate.validation_split")
    minimum_positive = _integer(
        teacher_gate.get("minimum_positive_layers"),
        "readout.teacher_gate.minimum_positive_layers",
        minimum=1,
    )
    if minimum_positive > len(layers):
        raise ConfigError("readout.teacher_gate.minimum_positive_layers exceeds layer count")
    minimum_cosine = teacher_gate.get("minimum_median_cosine")
    if not isinstance(minimum_cosine, (int, float)) or isinstance(minimum_cosine, bool):
        raise ConfigError("readout.teacher_gate.minimum_median_cosine must be numeric")
    carrier_state_gate = _mapping(
        readout.get("carrier_state_gate"), "readout.carrier_state_gate"
    )
    carrier_gate_enabled = carrier_state_gate.get("enabled")
    if not isinstance(carrier_gate_enabled, bool):
        raise ConfigError("readout.carrier_state_gate.enabled must be boolean")
    if kind == "silent_carriers" and carrier_gate_enabled is not True:
        raise ConfigError("silent_carriers requires the carrier-state persistence gate")
    if carrier_gate_enabled:
        _nonempty(carrier_state_gate.get("split"), "readout.carrier_state_gate.split")
        _integer(
            carrier_state_gate.get("prompt_count"),
            "readout.carrier_state_gate.prompt_count",
            minimum=1,
        )
        carrier_minimum = _integer(
            carrier_state_gate.get("minimum_positive_layers"),
            "readout.carrier_state_gate.minimum_positive_layers",
            minimum=1,
        )
        if carrier_minimum > len(layers):
            raise ConfigError(
                "readout.carrier_state_gate.minimum_positive_layers exceeds layer count"
            )
    implementation = _mapping(readout.get("implementation"), "readout.implementation")
    _nonempty(implementation.get("url"), "readout.implementation.url")
    _validate_revision(implementation.get("revision"), "readout.implementation.revision")
    frozen_artifact = readout.get("frozen_artifact")
    if frozen_artifact is not None:
        frozen_artifact = _mapping(frozen_artifact, "readout.frozen_artifact")
        _nonempty(frozen_artifact.get("repo"), "readout.frozen_artifact.repo")
        _validate_revision(frozen_artifact.get("revision"), "readout.frozen_artifact.revision")
        _nonempty(frozen_artifact.get("filename"), "readout.frozen_artifact.filename")

    semantic_contrast = readout.get("semantic_contrast")
    if kind == "wolf_sl":
        semantic_contrast = _mapping(semantic_contrast, "readout.semantic_contrast")
        _nonempty(semantic_contrast.get("name"), "readout.semantic_contrast.name")
        term_sets: list[set[str]] = []
        for key in ("positive_terms", "negative_terms"):
            terms = semantic_contrast.get(key)
            if (
                not isinstance(terms, list)
                or not terms
                or not all(isinstance(term, str) and term for term in terms)
            ):
                raise ConfigError(
                    f"readout.semantic_contrast.{key} must be a nonempty string list"
                )
            if len(terms) != len(set(terms)):
                raise ConfigError(f"readout.semantic_contrast.{key} must be unique")
            term_sets.append(set(terms))
        if term_sets[0] & term_sets[1]:
            raise ConfigError(
                "readout.semantic_contrast positive and negative terms must be disjoint"
            )

    probe_bank = readout.get("probe_bank")
    if probe_bank is not None:
        allowed_probe_banks = (
            {"animal_preference_v1"}
            if kind == "wolf_sl"
            else {"disposition_v1", "short_user_orientation_v1"}
        )
        if probe_bank not in allowed_probe_banks:
            raise ConfigError(
                f"readout.probe_bank must be one of {sorted(allowed_probe_banks)!r}"
            )

    seeds = _mapping(cfg.get("seeds"), "seeds")
    for key in ("prompts", "generation", "split", "behavior"):
        _integer(seeds.get(key), f"seeds.{key}", minimum=0)
    student_seeds = seeds.get("students")
    if not isinstance(student_seeds, list) or len(student_seeds) < 3:
        raise ConfigError("seeds.students must contain at least three paired seeds")
    if len(set(student_seeds)) != len(student_seeds):
        raise ConfigError("seeds.students must be unique")
    for index, seed in enumerate(student_seeds):
        _integer(seed, f"seeds.students[{index}]", minimum=0)

    carrier = _mapping(cfg.get("carrier"), "carrier")
    if carrier.get("type") != "numbers":
        raise ConfigError("the core scaffold currently requires carrier.type='numbers'")
    generated = _integer(
        carrier.get("generated_per_condition"), "carrier.generated_per_condition", minimum=1
    )
    train_size = _integer(carrier.get("train_size"), "carrier.train_size", minimum=1)
    eval_size = _integer(carrier.get("eval_size"), "carrier.eval_size", minimum=1)
    if generated < train_size + eval_size:
        raise ConfigError("carrier.generated_per_condition must cover train_size + eval_size")
    low = _integer(carrier.get("prefix_min_count"), "carrier.prefix_min_count", minimum=1)
    high = _integer(carrier.get("prefix_max_count"), "carrier.prefix_max_count", minimum=low)
    if high < low:
        raise ConfigError("carrier.prefix_max_count must be >= prefix_min_count")
    _integer(carrier.get("answer_max_count"), "carrier.answer_max_count", minimum=1)
    _integer(carrier.get("answer_max_digits"), "carrier.answer_max_digits", minimum=1)
    for key in ("temperature", "top_p"):
        number = carrier.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or number <= 0:
            raise ConfigError(f"carrier.{key} must be positive")

    training = _mapping(cfg.get("training"), "training")
    _validate_training(training.get("student"), "training.student")
    if training.get("teacher") is not None:
        _validate_training(training.get("teacher"), "training.teacher")

    batch_geometry = cfg.get("batch_geometry")
    if batch_geometry is not None:
        batch_geometry = _mapping(batch_geometry, "batch_geometry")
        try:
            verify_declared_batch_geometry(
                batch_geometry,
                train_examples=train_size,
                training_config=training["student"],
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(str(error)) from error

    conditions = _mapping(cfg.get("conditions"), "conditions")
    for name in ("treatment", "control"):
        condition = _mapping(conditions.get(name), f"conditions.{name}")
        system_prompt = condition.get("system_prompt")
        if system_prompt is not None:
            _nonempty(system_prompt, f"conditions.{name}.system_prompt")
        history = condition.get("history")
        if not isinstance(history, list):
            raise ConfigError(f"conditions.{name}.history must be a list")
        for index, message in enumerate(history):
            message = _mapping(message, f"conditions.{name}.history[{index}]")
            if message.get("role") not in {"user", "assistant"}:
                raise ConfigError(f"conditions.{name}.history[{index}].role is invalid")
            _nonempty(message.get("content"), f"conditions.{name}.history[{index}].content")
        if history:
            expected_roles = [
                "user" if index % 2 == 0 else "assistant" for index in range(len(history))
            ]
            actual_roles = [message["role"] for message in history]
            if actual_roles != expected_roles or actual_roles[-1] != "assistant":
                raise ConfigError(
                    f"conditions.{name}.history must alternate user/assistant "
                    "and end with assistant"
                )

    if kind == "wolf_sl":
        teacher = _mapping(cfg.get("teacher"), "teacher")
        if str(teacher.get("target", "")).lower() != "wolf":
            raise ConfigError("the preregistered positive-control teacher target is 'wolf'")
        if teacher.get("induction") != "system_prompt":
            raise ConfigError("wolf_sl teacher.induction must be 'system_prompt'")
        if (
            conditions["treatment"].get("adapter") is not None
            or conditions["control"].get("adapter") is not None
        ):
            raise ConfigError("wolf_sl must use the same unmodified checkpoint in both arms")
        if not conditions["treatment"].get("system_prompt"):
            raise ConfigError("wolf_sl treatment must define the wolf system_prompt")
        if conditions["control"].get("system_prompt") is not None:
            raise ConfigError("wolf_sl control system_prompt must be null")
        if conditions["treatment"]["history"] or conditions["control"]["history"]:
            raise ConfigError("wolf_sl treatment and control histories must both be empty")
    else:
        if (
            conditions["treatment"].get("adapter") is not None
            or conditions["control"].get("adapter") is not None
        ):
            raise ConfigError(
                "silent_carriers conditions must use the same unmodified checkpoint"
            )
        if not conditions["treatment"]["history"] or not conditions["control"]["history"]:
            raise ConfigError("silent_carriers requires both treatment and control histories")
        if any(condition.get("system_prompt") is not None for condition in conditions.values()):
            raise ConfigError("silent_carriers must express conditioning only through history")
        treatment_history = conditions["treatment"]["history"]
        control_history = conditions["control"]["history"]
        if [message["role"] for message in treatment_history] != [
            message["role"] for message in control_history
        ]:
            raise ConfigError("silent_carriers histories must use identical turn structure")
        if [len(message["content"].split()) for message in treatment_history] != [
            len(message["content"].split()) for message in control_history
        ]:
            raise ConfigError("silent_carriers histories must be word-count matched by turn")

    evaluation = _mapping(cfg.get("behavior"), "behavior")
    _integer(evaluation.get("samples_per_prompt"), "behavior.samples_per_prompt", minimum=1)
    _integer(evaluation.get("max_new_tokens"), "behavior.max_new_tokens", minimum=1)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return validate_config(raw)


def resolve_config(
    config: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Return a copy with protocol paths made absolute without changing hashes."""
    resolved = copy.deepcopy(config)
    root = Path(repo_root or Path.cwd()).resolve()
    run_root = Path(resolved["experiment"]["run_root"])
    if not run_root.is_absolute():
        resolved["experiment"]["run_root"] = str((root / run_root).resolve())
    for condition in resolved["conditions"].values():
        adapter = condition.get("adapter")
        if adapter and not Path(adapter).is_absolute():
            condition["adapter"] = str((root / adapter).resolve())
    return resolved
