from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from silent_transfer.config import ConfigError, load_config, validate_config
from silent_transfer.training import train_adapter

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = ROOT / "configs" / "wolf_sl_9b_a40_benchmark.yaml"


def test_a40_benchmark_is_isolated_and_exercises_full_microbatch_shape():
    benchmark = load_config(BENCHMARK_CONFIG)
    full = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")

    assert benchmark["experiment"]["id"] != full["experiment"]["id"]
    assert benchmark["experiment"]["run_root"] != full["experiment"]["run_root"]
    assert "benchmarks/" in benchmark["experiment"]["run_root"]
    assert "not a preregistered result" in benchmark["experiment"]["estimand"]
    assert benchmark["seeds"] != full["seeds"]
    assert benchmark["model"] == full["model"]
    assert benchmark["conditions"] == full["conditions"]

    training = benchmark["training"]["student"]
    assert training["batch_size"] == full["training"]["student"]["batch_size"] == 24
    assert training["gradient_accumulation_steps"] == 3
    assert training["max_steps"] == 20
    assert benchmark["carrier"]["train_size"] == 20 * 24 * 3
    assert benchmark["carrier"]["generated_per_condition"] > (
        benchmark["carrier"]["train_size"] + benchmark["carrier"]["eval_size"]
    )


def test_max_steps_is_validated_as_a_positive_integer():
    benchmark = load_config(BENCHMARK_CONFIG)
    for invalid in (0, -1, True, 1.5):
        broken = copy.deepcopy(benchmark)
        broken["training"]["student"]["max_steps"] = invalid
        with pytest.raises(ConfigError, match="max_steps"):
            validate_config(broken)


def test_training_passes_and_verifies_optional_optimizer_step_budget():
    source = inspect.getsource(train_adapter)
    assert 'max_steps=int(training_config.get("max_steps", -1))' in source
    assert 'optimizer_steps = int(trainer.state.global_step)' in source
    assert 'optimizer_steps != int(expected_steps)' in source


def test_a40_launcher_has_hard_isolation_and_step_guards():
    script = (ROOT / "scripts" / "lambda" / "benchmark_wolf_a40.sh").read_text(
        encoding="utf-8"
    )

    assert 'CONFIG="${1:-configs/wolf_sl_9b_a40_benchmark.yaml}"' in script
    assert 'experiment["id"] != "wolf-sl-gemma2-9b-a40-benchmark-v1"' in script
    assert '"benchmarks" not in run_root.parts' in script
    assert 'training.get("max_steps") != 20' in script
    assert 'training.get("batch_size") != 24' in script
    assert 'training.get("gradient_accumulation_steps") != 3' in script
    assert "minimum_examples =" in script
    assert 'len(gpus) != 1 or "A40" not in' in script
    assert "--condition control" in script
    assert 'metrics.get("optimizer_steps") != 20' in script
    assert "run_wolf_core.sh" not in script
    assert "behavior-suite" not in script
    assert "run_jlens" not in script
