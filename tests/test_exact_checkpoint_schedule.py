from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from silent_transfer.checkpointing import (
    checkpoint_training_arguments,
    exact_checkpoint_callback_class,
    verify_exact_checkpoint_artifacts,
)
from silent_transfer.config import ConfigError, load_config, validate_config
from silent_transfer.scheduler import pythia_lambda_factor
from silent_transfer.training import _pythia_lambda_trainer_class

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b.yaml"
EXACT_STEPS = [16, 64, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120]


def _config_with_exact_steps() -> dict:
    config = copy.deepcopy(load_config(CONFIG))
    training = config["training"]["student"]
    training.update(
        max_steps=EXACT_STEPS[-1],
        checkpoint_steps=list(EXACT_STEPS),
        save_total_limit=len(EXACT_STEPS),
    )
    return config


def test_config_accepts_irregular_exact_checkpoint_schedule() -> None:
    config = _config_with_exact_steps()

    validate_config(config)

    training = config["training"]["student"]
    assert checkpoint_training_arguments(training) == {
        "save_strategy": "no",
        "save_total_limit": len(EXACT_STEPS),
        "save_only_model": False,
    }


def test_config_accepts_zero_carrier_eval_rows() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["carrier"]["eval_size"] = 0

    validate_config(config)


def test_config_rejects_negative_carrier_eval_rows() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["carrier"]["eval_size"] = -1

    with pytest.raises(ConfigError, match="carrier.eval_size must be at least 0"):
        validate_config(config)


def test_legacy_config_retains_epoch_checkpoint_behavior() -> None:
    training = load_config(CONFIG)["training"]["student"]

    assert "checkpoint_steps" not in training
    assert checkpoint_training_arguments(training) == {
        "save_strategy": "epoch",
        "save_total_limit": training["save_total_limit"],
    }


@pytest.mark.parametrize(
    ("steps", "limit", "message"),
    (
        ([], 1, "nonempty integer list"),
        ([16, 16, 5120], 3, "strictly increasing and unique"),
        ([64, 16, 5120], 3, "strictly increasing and unique"),
        ([16, True, 5120], 3, r"checkpoint_steps\[1\] must be an integer"),
        ([16, 64, 5000], 3, "must end at max_steps"),
        ([16, 64, 5120], 2, "must be at least the number of checkpoint_steps"),
    ),
)
def test_config_rejects_unsafe_exact_checkpoint_schedule(
    steps: list[object], limit: int, message: str
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    training = config["training"]["student"]
    training.update(max_steps=5120, checkpoint_steps=steps, save_total_limit=limit)

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_config_rejects_exact_schedule_without_max_steps() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    training = config["training"]["student"]
    training.update(checkpoint_steps=[16, 64], save_total_limit=2)

    with pytest.raises(ConfigError, match="requires an explicit positive max_steps"):
        validate_config(config)


def test_exact_checkpoint_callback_requests_only_frozen_steps() -> None:
    class CallbackBase:
        pass

    callback = exact_checkpoint_callback_class(CallbackBase, (16, 64, 256))()
    control = SimpleNamespace(should_save=True)

    for step, expected in ((15, False), (16, True), (17, False), (64, True)):
        state = SimpleNamespace(global_step=step)
        returned = callback.on_step_end(None, state, control)
        assert returned is control
        assert control.should_save is expected


def test_pythia_lambda_factor_preserves_original_warmup_off_by_one() -> None:
    factors = [
        pythia_lambda_factor(step, warmup_steps=3, total_steps=6)
        for step in range(8)
    ]

    assert factors == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0, 2 / 3, 1 / 3, 0.0, 0.0])


def test_pythia_lambda_trainer_constructs_exact_scheduler() -> None:
    torch = pytest.importorskip("torch")

    class FakeTrainer:
        def __init__(self, *, args, optimizer):
            self.args = args
            self.optimizer = optimizer
            self.lr_scheduler = None

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.0002)
    trainer = _pythia_lambda_trainer_class(FakeTrainer)(
        args=SimpleNamespace(max_steps=6),
        optimizer=optimizer,
        scheduler_total_steps=6,
        scheduler_warmup_steps=3,
    )

    scheduler = trainer.create_scheduler(6)
    assert scheduler.last_epoch == 0
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0002 / 3)
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0002 * 2 / 3)


@pytest.mark.parametrize(
    ("semantics", "message"),
    (
        ("cosine", "lr_scheduler_semantics must be one of"),
        (1, "lr_scheduler_semantics must be one of"),
    ),
)
def test_config_rejects_unknown_scheduler_semantics(
    semantics: object, message: str
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["student"]["lr_scheduler_semantics"] = semantics

    with pytest.raises(ConfigError, match=message):
        validate_config(config)


def test_pythia_scheduler_semantics_requires_explicit_geometry() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    training = config["training"]["student"]
    training["lr_scheduler_semantics"] = "pythia_lambda_v1"

    with pytest.raises(ConfigError, match="requires scheduler_total_steps"):
        validate_config(config)


def test_hugging_face_trainer_saves_only_exact_steps_with_full_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    dataset_base = pytest.importorskip("torch.utils.data").Dataset

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 1)

        def forward(self, input_ids=None, labels=None):
            logits = self.projection(input_ids.float())
            return {
                "loss": torch.nn.functional.mse_loss(logits, labels.float()),
                "logits": logits,
            }

    class TinyDataset(dataset_base):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, index: int) -> dict:
            return {
                "input_ids": torch.tensor([float(index), 1.0]),
                "labels": torch.tensor([float(index % 2)]),
            }

    steps = (1, 3, 4)
    callback_class = exact_checkpoint_callback_class(
        transformers.TrainerCallback, steps
    )
    arguments = transformers.TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=steps[-1],
        per_device_train_batch_size=1,
        save_strategy="no",
        save_total_limit=len(steps),
        save_only_model=False,
        logging_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        dataloader_pin_memory=False,
    )
    trainer_class = _pythia_lambda_trainer_class(transformers.Trainer)
    trainer = trainer_class(
        model=TinyModel(),
        args=arguments,
        train_dataset=TinyDataset(),
        callbacks=[callback_class()],
        scheduler_total_steps=steps[-1],
        scheduler_warmup_steps=1,
    )

    trainer.train()

    assert verify_exact_checkpoint_artifacts(tmp_path, steps) == steps


def _write_resumable_checkpoint(root: Path, step: int) -> None:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for filename in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
        (checkpoint / filename).write_bytes(b"state")


def test_exact_checkpoint_artifact_audit_requires_all_and_only_frozen_steps(
    tmp_path: Path,
) -> None:
    for step in (16, 64, 256):
        _write_resumable_checkpoint(tmp_path, step)

    assert verify_exact_checkpoint_artifacts(tmp_path, (16, 64, 256)) == (16, 64, 256)

    _write_resumable_checkpoint(tmp_path, 128)
    with pytest.raises(RuntimeError, match="checkpoint schedule mismatch"):
        verify_exact_checkpoint_artifacts(tmp_path, (16, 64, 256))


def test_exact_checkpoint_artifact_audit_requires_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    _write_resumable_checkpoint(tmp_path, 16)
    (tmp_path / "checkpoint-16" / "optimizer.pt").unlink()

    with pytest.raises(RuntimeError, match="not resumable; missing optimizer.pt"):
        verify_exact_checkpoint_artifacts(tmp_path, (16,))
