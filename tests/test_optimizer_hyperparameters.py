from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

import scripts.verify_tenpass_checkpoint_cell as checkpoint_verifier
from silent_transfer.config import ConfigError, load_config, validate_config
from silent_transfer.optimizer import resolve_adamw_hyperparameters
from silent_transfer.training import train_adapter

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b.yaml"


def test_legacy_config_resolves_hugging_face_adamw_defaults_without_mutation() -> None:
    config = load_config(CONFIG)
    training = config["training"]["student"]

    assert not {"adam_beta1", "adam_beta2", "adam_epsilon"} & set(training)
    assert resolve_adamw_hyperparameters(training) == {
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
    }


def test_config_accepts_explicit_literal_pythia_adamw_parameters() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    training = config["training"]["student"]
    training.update(adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8)

    validate_config(config)
    assert resolve_adamw_hyperparameters(training) == {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
    }


def test_config_accepts_ordinary_lora_for_literal_pythia_recipe() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    lora = config["training"]["student"]["lora"]
    target_modules = list(lora["target_modules"])
    lora.update(r=8, alpha=16, use_rslora=False)

    validate_config(config)
    assert lora["r"] == 8
    assert lora["alpha"] == 16
    assert lora["use_rslora"] is False
    assert lora["target_modules"] == target_modules


@pytest.mark.parametrize("value", (None, 0, 1, "false"))
def test_config_rejects_nonboolean_rslora_flag(value: object) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["student"]["lora"]["use_rslora"] = value

    with pytest.raises(ConfigError, match="use_rslora must be boolean"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("adam_beta1", -0.1, r"adam_beta1 must be in \[0, 1\)"),
        ("adam_beta1", 1.0, r"adam_beta1 must be in \[0, 1\)"),
        ("adam_beta2", float("nan"), r"adam_beta2 must be in \[0, 1\)"),
        ("adam_beta2", True, r"adam_beta2 must be in \[0, 1\)"),
        ("adam_epsilon", 0.0, "adam_epsilon must be positive"),
        ("adam_epsilon", float("inf"), "adam_epsilon must be positive"),
    ),
)
def test_config_rejects_invalid_adamw_parameters(
    field: str, value: object, message: str
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["student"][field] = value

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_train_adapter_passes_resolved_adamw_fields_to_training_arguments() -> None:
    source = inspect.getsource(train_adapter)
    assert "adamw_hyperparameters = resolve_adamw_hyperparameters(training_config)" in source
    assert "**adamw_hyperparameters" in source
    assert '"optimizer_hyperparameters": adamw_hyperparameters' in source


@pytest.mark.parametrize(
    ("training", "saved_betas"),
    (
        ({}, (0.9, 0.999)),
        ({"adam_beta1": 0.9, "adam_beta2": 0.95}, (0.9, 0.95)),
    ),
)
def test_checkpoint_auditor_accepts_legacy_and_explicit_adam_betas(
    monkeypatch: pytest.MonkeyPatch,
    training: dict[str, float],
    saved_betas: tuple[float, float],
) -> None:
    torch = pytest.importorskip("torch")
    state_dict = {
        "state": {
            0: {
                "step": torch.tensor(7.0),
                "exp_avg": torch.zeros(2),
                "exp_avg_sq": torch.zeros(2),
            }
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 0.0001,
                "betas": saved_betas,
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        ],
    }
    monkeypatch.setattr(checkpoint_verifier, "_safe_torch_load", lambda _path: state_dict)

    report = checkpoint_verifier._audit_optimizer(
        Path("optimizer.pt"),
        step=7,
        training={**training, "weight_decay": 0.1},
        scheduler_lrs=[0.0001],
    )

    assert report["parameter_groups"][0]["betas"] == list(saved_betas)


def test_checkpoint_auditor_rejects_beta_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    state_dict = {
        "state": {
            0: {
                "step": torch.tensor(7.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 0.0001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        ],
    }
    monkeypatch.setattr(checkpoint_verifier, "_safe_torch_load", lambda _path: state_dict)

    with pytest.raises(ValueError, match="beta values differ"):
        checkpoint_verifier._audit_optimizer(
            Path("optimizer.pt"),
            step=7,
            training={"adam_beta1": 0.9, "adam_beta2": 0.95, "weight_decay": 0.1},
            scheduler_lrs=[0.0001],
        )


def test_checkpoint_auditor_accepts_transformers_decay_and_no_decay_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    state_dict = {
        "state": {
            parameter_id: {
                "step": torch.tensor(7.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
            for parameter_id in (0, 1)
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 0.0001,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
            {
                "params": [1],
                "lr": 0.0001,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        ],
    }
    monkeypatch.setattr(checkpoint_verifier, "_safe_torch_load", lambda _path: state_dict)

    report = checkpoint_verifier._audit_optimizer(
        Path("optimizer.pt"),
        step=7,
        training={"adam_beta1": 0.9, "adam_beta2": 0.95, "weight_decay": 0.1},
        scheduler_lrs=[0.0001, 0.0001],
    )

    assert [group["weight_decay"] for group in report["parameter_groups"]] == [
        0.1,
        0.0,
    ]


def test_checkpoint_auditor_rejects_unregistered_weight_decay_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    state_dict = {
        "state": {
            0: {
                "step": torch.tensor(7.0),
                "exp_avg": torch.zeros(1),
                "exp_avg_sq": torch.zeros(1),
            }
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 0.0001,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.05,
            }
        ],
    }
    monkeypatch.setattr(checkpoint_verifier, "_safe_torch_load", lambda _path: state_dict)

    with pytest.raises(ValueError, match="weight decay differs"):
        checkpoint_verifier._audit_optimizer(
            Path("optimizer.pt"),
            step=7,
            training={"adam_beta1": 0.9, "adam_beta2": 0.95, "weight_decay": 0.1},
            scheduler_lrs=[0.0001],
        )
