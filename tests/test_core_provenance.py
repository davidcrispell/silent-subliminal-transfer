from __future__ import annotations

import json

from silent_transfer.costs import COST_FIELDS, append_cost, read_costs
from silent_transfer.provenance import canonical_json_bytes, sha256_file, sha256_value


def test_canonical_hash_ignores_mapping_order():
    left = {"model": {"revision": "a" * 40, "id": "x"}, "seeds": [1, 2, 3]}
    right = {"seeds": [1, 2, 3], "model": {"id": "x", "revision": "a" * 40}}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_value(left) == sha256_value(right)


def test_file_hash_changes_with_content(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps({"value": 1}))
    first = sha256_file(path)
    path.write_text(json.dumps({"value": 2}))
    assert sha256_file(path) != first


def test_cost_ledger_calculates_and_accumulates(tmp_path):
    ledger = tmp_path / "costs.csv"
    first = append_cost(
        ledger,
        run_id="run-a",
        stage="E1",
        provider="Lambda",
        instance_type="GH200",
        gpu_count=1,
        gpu_hours=2.0,
        rate_per_gpu_hour_usd=2.29,
    )
    second = append_cost(
        ledger,
        run_id="run-b",
        stage="E2",
        provider="Lambda",
        instance_type="H100 PCIe",
        gpu_count=1,
        gpu_hours=1.0,
        rate_per_gpu_hour_usd=3.29,
        storage_cost_usd=1.0,
    )
    assert first["compute_usd"] == "4.58"
    assert second["ledger_total_usd"] == 8.87
    assert len(read_costs(ledger)) == 2


def test_tracked_cost_ledger_template_matches_writer():
    from pathlib import Path

    ledger = Path(__file__).resolve().parents[1] / "runs" / "COST_LEDGER.csv"
    assert tuple(ledger.read_text().splitlines()[0].split(",")) == COST_FIELDS
