import json

import pytest

from silent_transfer.behavior import summarize_paired_behavior


def test_summarize_paired_behavior_computes_seed_level_ci(tmp_path):
    config = {
        "seeds": {"students": [101, 102, 103]},
        "teacher": {"target": "wolf"},
    }
    root = tmp_path / "behavior"
    for seed, treatment_rate, control_rate in (
        (101, 0.20, 0.10),
        (102, 0.35, 0.15),
        (103, 0.45, 0.15),
    ):
        for condition, rate in (
            ("treatment", treatment_rate),
            ("control", control_rate),
        ):
            destination = root / "students" / condition / f"seed-{seed}" / "summary.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"target_rate": rate}), encoding="utf-8")

    output = tmp_path / "paired_summary.json"
    summary = summarize_paired_behavior(config, behavior_root=root, output_path=output)

    assert summary["n_pairs"] == 3
    assert summary["positive_pairs"] == 3
    assert summary["mean_paired_delta"] == pytest.approx(0.2)
    assert summary["standard_error_across_pairs"] == pytest.approx(0.0577350269)
    assert summary["paired_t_95_ci"] == pytest.approx([-0.0484137712, 0.4484137712])
    assert json.loads(output.read_text(encoding="utf-8")) == summary
