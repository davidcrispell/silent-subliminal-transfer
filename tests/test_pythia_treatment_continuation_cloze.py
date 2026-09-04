from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.summarize_pythia_treatment_continuation_cloze import (
    MARGIN_KEY,
    PROBABILITY_KEY,
    _bootstrap_mean_upper_bound,
    _classify_saturation,
    _verify_curve_publication,
)
from scripts.verify_pythia_treatment_continuation import CHECKPOINT_STEPS
from scripts.verify_pythia_treatment_continuation_cloze import (
    EVALUATION_CODE_PATHS,
    EXPECTED_LAYER_SIGNATURE,
    FROZEN_CONFIG_SHA256,
    FROZEN_TRAINING_GIT_COMMIT,
    _validate_layer_signature,
    _verify_evaluation_code_at_head,
)

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/lambda/run_pythia_treatment_continuation_cloze.sh"


def _rule() -> dict[str, object]:
    return {
        "primary_material_gain_nats": 0.10,
        "secondary_material_gain": 0.01,
        "required_final_epoch_intervals": 2,
        "bootstrap_confidence": 0.95,
        "bootstrap_samples": 10000,
    }


def _effects(margins: list[float], probabilities: list[float]):
    assert len(margins) == len(probabilities) == len(CHECKPOINT_STEPS)
    return {
        step: {
            MARGIN_KEY: [margins[index]] * 60,
            PROBABILITY_KEY: [probabilities[index]] * 60,
        }
        for index, step in enumerate(CHECKPOINT_STEPS)
    }


def test_wrapper_is_treatment_only_and_audits_before_and_after_curve() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="utf-8")

    verifier = "verify_pythia_treatment_continuation_cloze.py"
    assert source.index(verifier) < source.index("silent-transfer animal-cloze")
    assert source.rindex(verifier) > source.index("silent-transfer animal-cloze")
    assert "models/students/treatment/seed-$SEED" in source
    assert "evaluations/cloze/treatment/seed-$SEED" in source
    assert "run_pythia_transplant_reference_cloze" not in source
    assert "--context-condition" not in source
    assert "SST_EXPECTED_TRAINING_GIT_COMMIT" in source
    assert "SST_EXPECTED_EVALUATION_GIT_COMMIT" in source
    assert FROZEN_TRAINING_GIT_COMMIT in source
    assert FROZEN_CONFIG_SHA256 in source
    assert "treatment_cloze_curve_complete.json" in source
    assert "rm -f" in source


def test_evaluation_code_is_byte_identical_to_head(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    for relative in EVALUATION_CODE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen bytes for {relative}\n", encoding="utf-8")
    subprocess.run(["git", "add", *EVALUATION_CODE_PATHS], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)

    observed = _verify_evaluation_code_at_head(tmp_path)
    assert set(observed) == set(EVALUATION_CODE_PATHS)

    (tmp_path / EVALUATION_CODE_PATHS[0]).write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked working-tree changes"):
        _verify_evaluation_code_at_head(tmp_path)


def test_curve_contract_is_all_19_half_passes_and_all_43_hidden_states() -> None:
    assert CHECKPOINT_STEPS == list(range(1024, 10241, 512))
    assert len(CHECKPOINT_STEPS) == 19
    assert EXPECTED_LAYER_SIGNATURE == [
        {"index": 0, "name": "embedding"},
        *[{"index": index, "name": f"block_{index:02d}"} for index in range(1, 43)],
    ]
    _validate_layer_signature(EXPECTED_LAYER_SIGNATURE, description="complete fixture")
    with pytest.raises(ValueError, match="every Gemma hidden state"):
        _validate_layer_signature(
            EXPECTED_LAYER_SIGNATURE[:-1], description="truncated fixture"
        )


def test_bootstrap_upper_bound_is_deterministic_and_exact_for_constant_values() -> None:
    observed = _bootstrap_mean_upper_bound(
        [0.0375] * 60, samples=10000, confidence=0.95, seed=12345
    )
    repeated = _bootstrap_mean_upper_bound(
        [0.0375] * 60, samples=10000, confidence=0.95, seed=12345
    )
    assert observed == pytest.approx(0.0375)
    assert repeated == observed


def test_frozen_retrospective_rule_calls_flat_tail_saturated() -> None:
    margins = [min(index * 0.20, 1.20) for index in range(len(CHECKPOINT_STEPS))]
    probabilities = [min(index * 0.02, 0.08) for index in range(len(CHECKPOINT_STEPS))]

    result = _classify_saturation(
        _effects(margins, probabilities),
        saturation_rule=_rule(),
        bootstrap_seed=43001,
    )

    assert result["saturated"] is True
    assert result["classification"] == "saturated_by_frozen_retrospective_rule"
    assert result["earliest_half_pass_with_no_later_material_gain"] is not None
    bootstrap = result["final_full_epoch_increment_bootstrap"]
    assert bootstrap["bootstrap_samples"] == 10000
    assert bootstrap["all_required_upper_bounds_below_threshold"] is True
    assert len(bootstrap["intervals"]) == 2


def test_frozen_retrospective_rule_rejects_material_late_growth() -> None:
    margins = [index * 0.08 for index in range(len(CHECKPOINT_STEPS))]
    probabilities = [index * 0.008 for index in range(len(CHECKPOINT_STEPS))]

    result = _classify_saturation(
        _effects(margins, probabilities),
        saturation_rule=_rule(),
        bootstrap_seed=43001,
    )

    assert result["saturated"] is False
    assert result["classification"] == "not_saturated_within_fixed_ten_pass_horizon"
    assert (
        result["final_full_epoch_increment_bootstrap"][
            "all_required_upper_bounds_below_threshold"
        ]
        is False
    )


def test_curve_publication_requires_checkpoint_bytes_verified_at_source(
    tmp_path: Path,
) -> None:
    audit = {
        "schema_version": 1,
        "stage": "pythia_treatment_continuation_cloze_curve_audit",
        "scope": "treatment_only_no_new_control_no_base_rerun",
        "config_sha256": FROZEN_CONFIG_SHA256,
        "run_id": "continuation-test",
        "training_git_commit": FROZEN_TRAINING_GIT_COMMIT,
        "evaluation_git_commit": "a" * 40,
        "condition": "treatment",
        "seed": 53101,
        "optimizer_steps": CHECKPOINT_STEPS,
        "prompt_count_per_checkpoint": 60,
        "hidden_state_count_per_prompt": 43,
        "checkpoint_manifest_sha256": "b" * 64,
        "evaluation_code_sha256": {path: "e" * 64 for path in EVALUATION_CODE_PATHS},
        "artifact_sha256": {"checkpoint-1024/summary.json": "c" * 64},
    }
    marker = {
        **audit,
        "checkpoint_bytes_verified_locally": {str(step): True for step in CHECKPOINT_STEPS},
        "protocol_report_sha256": "d" * 64,
        "evaluation_audits": {},
    }
    marker_path = tmp_path / "treatment_cloze_curve_complete.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    assert _verify_curve_publication(marker_path, audit=audit) == marker

    marker["checkpoint_bytes_verified_locally"]["1024"] = False
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="locally verifying every checkpoint"):
        _verify_curve_publication(marker_path, audit=audit)
