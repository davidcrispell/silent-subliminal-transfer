from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.verify_tenpass_checkpoint_cell as checkpoint_verifier
from scripts.verify_pythia_transplant_checkpoint_cell import _verify_paired_data
from scripts.verify_tenpass_checkpoint_cell import _expected_lr
from silent_transfer.data import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
LAMBDA = ROOT / "scripts" / "lambda"
WRAPPERS = (
    LAMBDA / "prepare_pythia_transplant_condition.sh",
    LAMBDA / "finalize_pythia_transplant_data.sh",
    LAMBDA / "run_pythia_transplant_student_cell.sh",
    LAMBDA / "run_pythia_transplant_cloze_cell.sh",
    LAMBDA / "run_pythia_transplant_reference_cloze.sh",
)


def test_pythia_checkpoint_auditor_uses_literal_warmup_semantics() -> None:
    base_lr = 2e-4
    observed = [
        _expected_lr(
            step=step,
            base_lr=base_lr,
            warmup=8,
            total=5120,
            semantics="pythia_lambda_v1",
        )
        for step in (0, 1, 7, 8, 512)
    ]
    assert observed == pytest.approx(
        [
            base_lr / 8,
            base_lr * 2 / 8,
            base_lr,
            base_lr,
            base_lr * (5120 - 512) / (5120 - 8),
        ]
    )


def test_scheduler_state_audit_distinguishes_pythia_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_lr = 2e-4
    monkeypatch.setattr(
        checkpoint_verifier,
        "_safe_torch_load",
        lambda _path: {
            "last_epoch": 1,
            "_step_count": 2,
            "base_lrs": [base_lr],
            "_last_lr": [base_lr * 2 / 8],
        },
    )

    state, last_lrs = checkpoint_verifier._audit_scheduler(
        Path("scheduler.pt"),
        step=1,
        training={
            "learning_rate": base_lr,
            "warmup_steps": 8,
            "scheduler_total_steps": 5120,
            "lr_scheduler_semantics": "pythia_lambda_v1",
        },
    )

    assert state["last_epoch"] == 1
    assert last_lrs == pytest.approx([base_lr * 2 / 8])


def _write_paired_data(root: Path, *, rows: int = 3, tokens: int = 50) -> None:
    paired = root / "data" / "paired"
    paired.mkdir(parents=True)
    for condition in ("treatment", "control"):
        write_jsonl(
            paired / f"{condition}_train.jsonl",
            [
                {
                    "pair_id": f"numbers-{index:06d}",
                    "completion_token_count": tokens,
                    "full_token_count": (73, 80, 93)[index],
                }
                for index in range(rows)
            ],
        )
        write_jsonl(paired / f"{condition}_eval.jsonl", [])


def test_paired_data_audit_requires_unique_alignment_and_exact_token_exposure(
    tmp_path: Path,
) -> None:
    _write_paired_data(tmp_path)

    train, evaluation, hashes, geometry = _verify_paired_data(
        tmp_path,
        condition="treatment",
        expected_train_rows=3,
        expected_training_completion_tokens=50,
        expected_full_token_count_min=73,
        expected_full_token_count_max=93,
        max_length=96,
    )

    assert train.name == "treatment_train.jsonl"
    assert evaluation.name == "treatment_eval.jsonl"
    assert set(hashes) == {
        "condition_train",
        "other_condition_train",
        "condition_eval",
        "other_condition_eval",
    }
    assert geometry == {
        "training_completion_tokens_per_example": 50,
        "full_token_count_min": 73,
        "full_token_count_max": 93,
        "max_length": 96,
    }


def test_paired_data_audit_rejects_wrong_completion_width(tmp_path: Path) -> None:
    _write_paired_data(tmp_path, tokens=49)

    with pytest.raises(ValueError, match="frozen chat-template completion-token exposure"):
        _verify_paired_data(
            tmp_path,
            condition="control",
            expected_train_rows=3,
            expected_training_completion_tokens=50,
            expected_full_token_count_min=73,
            expected_full_token_count_max=93,
            max_length=96,
        )


def test_runpod_wrappers_have_valid_bash_syntax() -> None:
    for wrapper in WRAPPERS:
        subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_runpod_wrappers_take_nonblocking_per_stage_locks() -> None:
    for wrapper in WRAPPERS:
        source = wrapper.read_text()
        assert "flock -n 9" in source
        assert "/orchestration/locks/" in source


def test_training_wrapper_fails_closed_and_audits_before_and_after_training() -> None:
    source = (LAMBDA / "run_pythia_transplant_student_cell.sh").read_text()
    assert "set -euo pipefail" in source
    assert "SST_EXPECTED_GIT_COMMIT" in source
    assert "SST_EXPECTED_CONFIG_SHA256" in source
    assert "SST_USE_OFFLINE_CACHE" in source
    assert "verify_offline_hf_cache.py" in source
    assert source.index("verify_pythia_transplant_data.py") < source.index(
        "silent-transfer train-student"
    )
    assert source.index("silent-transfer train-student") < source.index(
        "verify_pythia_transplant_checkpoint_cell.py"
    )
    assert "--resume" in source


def test_generation_and_pair_wrappers_keep_conditions_parallelizable() -> None:
    generation = (LAMBDA / "prepare_pythia_transplant_condition.sh").read_text()
    finalize = (LAMBDA / "finalize_pythia_transplant_data.sh").read_text()
    assert "generate-condition" in generation
    assert '--condition "$CONDITION"' in generation
    assert "pair-carriers" not in generation
    assert finalize.index("pair-carriers") < finalize.index("verify_pythia_transplant_data.py")


def test_cloze_wrapper_uses_all_registered_checkpoints_and_frozen_layout() -> None:
    source = (LAMBDA / "run_pythia_transplant_cloze_cell.sh").read_text()
    assert source.index("verify_pythia_transplant_checkpoint_cell.py") < source.index(
        "silent-transfer animal-cloze"
    )
    assert 'raw["dose_provenance"]["probe_optimizer_steps"]' in source
    assert "evaluations/cloze/$CONDITION/seed-$SEED/checkpoint-$STEP" in source
    assert "pythia_transplant_step_${STEP}_${CONDITION}_seed_${SEED}" in source
    assert "cloze_curve_complete.json" in source


def test_reference_cloze_wrapper_has_exact_base_and_teacher_labels() -> None:
    source = (LAMBDA / "run_pythia_transplant_reference_cloze.sh").read_text()
    assert 'CONTEXT_CONDITION="control"' in source
    assert 'CONTEXT_CONDITION="treatment"' in source
    assert "evaluations/cloze/$MODE" in source
    assert "pythia_transplant_${MODE}" in source
