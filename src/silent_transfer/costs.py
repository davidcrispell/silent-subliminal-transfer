from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COST_FIELDS = (
    "date",
    "run_id",
    "stage",
    "provider",
    "instance_type",
    "gpu_count",
    "started_at_utc",
    "ended_at_utc",
    "billed_hours",
    "compute_usd",
    "storage_usd",
    "api_usd",
    "other_usd",
    "total_usd",
    "invoice_or_console_ref",
    "notes",
)


def read_costs(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists():
        return []
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_cost(
    path: str | Path,
    *,
    run_id: str,
    stage: str,
    provider: str,
    instance_type: str,
    gpu_count: int,
    gpu_hours: float,
    rate_per_gpu_hour_usd: float,
    storage_cost_usd: float = 0.0,
    api_cost_usd: float = 0.0,
    other_cost_usd: float = 0.0,
    started_at_utc: str = "",
    ended_at_utc: str = "",
    invoice_or_instance_id: str = "",
    notes: str = "",
) -> dict[str, Any]:
    if gpu_count < 0 or gpu_hours < 0 or rate_per_gpu_hour_usd < 0:
        raise ValueError("GPU count, hours, and rate must be nonnegative")
    if storage_cost_usd < 0 or api_cost_usd < 0 or other_cost_usd < 0:
        raise ValueError("Storage, API, and other costs must be nonnegative")
    compute = gpu_count * gpu_hours * rate_per_gpu_hour_usd
    total = compute + storage_cost_usd + api_cost_usd + other_cost_usd
    now = datetime.now(timezone.utc)
    row = {
        "date": now.date().isoformat(),
        "run_id": run_id,
        "stage": stage,
        "provider": provider,
        "instance_type": instance_type,
        "gpu_count": gpu_count,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "billed_hours": f"{gpu_hours:.6f}",
        "compute_usd": f"{compute:.2f}",
        "storage_usd": f"{storage_cost_usd:.2f}",
        "api_usd": f"{api_cost_usd:.2f}",
        "other_usd": f"{other_cost_usd:.2f}",
        "total_usd": f"{total:.2f}",
        "invoice_or_console_ref": invoice_or_instance_id,
        "notes": f"rate=${rate_per_gpu_hour_usd:.4f}/GPUh; {notes}".rstrip("; "),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exists = destination.exists()
    if exists:
        with destination.open(newline="", encoding="utf-8") as handle:
            existing_fields = tuple(next(csv.reader(handle)))
        if existing_fields != COST_FIELDS:
            raise ValueError(
                f"Cost ledger schema mismatch: {existing_fields!r} != {COST_FIELDS!r}"
            )
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    rows = read_costs(destination)
    row["ledger_total_usd"] = round(sum(float(item["total_usd"]) for item in rows), 2)
    row["ledger_compute_total_usd"] = round(sum(float(item["compute_usd"]) for item in rows), 2)
    return row
