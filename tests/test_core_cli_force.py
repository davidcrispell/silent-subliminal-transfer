from __future__ import annotations

from pathlib import Path

import pytest

from silent_transfer import cli


def test_force_student_discards_adapter_and_trains(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "students" / "treatment" / "seed-7"
    (destination / "final_adapter").mkdir(parents=True)
    paths = {"students": tmp_path / "students", "paired": tmp_path / "paired"}
    config = {
        "seeds": {"students": [7]},
        "training": {"student": {"optimizer": "adamw_torch"}},
    }
    discarded = []
    monkeypatch.setattr(cli, "_paths", lambda _: paths)
    monkeypatch.setattr(cli, "_assert_paired_training_data", lambda _: None)
    monkeypatch.setattr(cli.shutil, "rmtree", lambda path: discarded.append(Path(path)))
    monkeypatch.setattr(cli, "train_adapter", lambda **kwargs: {"trained": True})
    result = cli._train_one_student(
        config,
        tmp_path,
        condition="treatment",
        seed=7,
        force=True,
        resume=False,
    )
    assert result == {"trained": True}
    assert discarded == [destination]


def test_force_and_resume_are_rejected_before_student_training(tmp_path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli._train_one_student(
            {"seeds": {"students": [7]}},
            tmp_path,
            condition="treatment",
            seed=7,
            force=True,
            resume=True,
        )
