from __future__ import annotations

from pathlib import Path

import yaml

from silent_transfer.provenance import sha256_value


def test_dose5_config_is_a_frozen_five_epoch_data_reuse() -> None:
    repository = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((repository / "configs/wolf_sl_9b.yaml").read_text())
    dose = yaml.safe_load((repository / "configs/wolf_sl_9b_dose5.yaml").read_text())

    provenance = dose["dose_provenance"]
    assert provenance["source_config_sha256"] == sha256_value(baseline)
    assert dose["experiment"]["run_root"] != baseline["experiment"]["run_root"]
    assert dose["model"] == baseline["model"]
    assert dose["teacher"] == baseline["teacher"]
    assert dose["carrier"] == baseline["carrier"]
    assert dose["conditions"] == baseline["conditions"]
    assert dose["seeds"] == baseline["seeds"]

    training = dose["training"]["student"]
    baseline_training = baseline["training"]["student"]
    assert training["epochs"] == 5
    assert training["max_steps"] == 3125
    assert training["batch_size"] * training["gradient_accumulation_steps"] == 16
    assert training["learning_rate"] == baseline_training["learning_rate"]
    assert training["warmup_ratio"] == baseline_training["warmup_ratio"]
    assert training["optimizer"] == baseline_training["optimizer"]
    assert training["weight_decay"] == baseline_training["weight_decay"]
    assert training["lora"] == baseline_training["lora"]
    assert training["save_total_limit"] == 5
    assert provenance["probe_optimizer_steps"] == [1875, 2500, 3125]
    assert provenance["target_optimizer_steps"] == provenance["probe_optimizer_steps"][-1]
