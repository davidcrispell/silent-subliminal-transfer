#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/wolf_sl_9b_a40_benchmark.yaml}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
SEED="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["student_seeds"][0])' "$VALIDATED")"

python - "$CONFIG" "$RUN_ROOT" <<'PY'
import pathlib
import sys

import yaml

config_path = pathlib.Path(sys.argv[1]).resolve()
run_root = pathlib.Path(sys.argv[2]).resolve()
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
experiment = config["experiment"]
training = config["training"]["student"]
carrier = config["carrier"]

if experiment["id"] != "wolf-sl-gemma2-9b-a40-benchmark-v1":
    raise SystemExit("Refusing non-benchmark experiment identity")
if "benchmarks" not in run_root.parts or "benchmark" not in run_root.name:
    raise SystemExit(f"Refusing non-benchmark run root: {run_root}")
if run_root.name == "wolf-sl-gemma2-9b-v1":
    raise SystemExit("Refusing the preregistered full-run directory")
if training.get("max_steps") != 20:
    raise SystemExit("A40 benchmark must request exactly 20 optimizer updates")
if training.get("batch_size") != 24 or training.get("gradient_accumulation_steps") != 3:
    raise SystemExit("A40 benchmark must exercise batch_size=24 and accumulation=3")
minimum_examples = training["max_steps"] * training["batch_size"] * training["gradient_accumulation_steps"]
if carrier["train_size"] < minimum_examples:
    raise SystemExit(
        f"Benchmark needs at least {minimum_examples} paired train rows, "
        f"not {carrier['train_size']}"
    )
PY

STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"

scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
python - "$RUN_ROOT/preflight.json" <<'PY'
import json
import pathlib
import sys

preflight_path = pathlib.Path(sys.argv[1])
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
gpus = preflight.get("gpus", [])
if len(gpus) != 1 or "A40" not in gpus[0].get("name", "").upper():
    raise SystemExit(
        "Refusing benchmark outside its one-GPU A40 target; "
        f"inspect {preflight_path}"
    )
PY
silent-transfer prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"

TREATMENT_STARTED="$(date +%s)"
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
TREATMENT_SECONDS="$(( $(date +%s) - TREATMENT_STARTED ))"

CONTROL_STARTED="$(date +%s)"
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition control
CONTROL_SECONDS="$(( $(date +%s) - CONTROL_STARTED ))"

silent-transfer pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"

TRAIN_STARTED="$(date +%s)"
silent-transfer train-student "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --condition control \
  --seed "$SEED" \
  --resume
TRAIN_SECONDS="$(( $(date +%s) - TRAIN_STARTED ))"

ENDED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TOTAL_SECONDS="$(( $(date +%s) - STARTED_EPOCH ))"

python - \
  "$RUN_ROOT" "$SEED" "$STARTED_AT_UTC" "$ENDED_AT_UTC" "$TOTAL_SECONDS" \
  "$TREATMENT_SECONDS" "$CONTROL_SECONDS" "$TRAIN_SECONDS" <<'PY'
import json
import pathlib
import sys

(
    run_root_raw,
    seed,
    started_at,
    ended_at,
    total_seconds,
    treatment_seconds,
    control_seconds,
    train_seconds,
) = sys.argv[1:]
run_root = pathlib.Path(run_root_raw)
metrics_path = run_root / "models" / "students" / "control" / f"seed-{seed}" / "training_metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if metrics.get("optimizer_steps") != 20:
    raise SystemExit(f"Expected 20 optimizer updates; inspect {metrics_path}")

summary = {
    "schema_version": 1,
    "purpose": "engineering-only A40 throughput benchmark; not an experimental result",
    "started_at_utc": started_at,
    "ended_at_utc": ended_at,
    "wall_seconds": int(total_seconds),
    "generation_wall_seconds": {
        "treatment": int(treatment_seconds),
        "control": int(control_seconds),
    },
    "training_wall_seconds": int(train_seconds),
    "training_condition": "control",
    "training_seed": int(seed),
    "optimizer_steps": metrics["optimizer_steps"],
    "configured_max_steps": metrics["configured_max_steps"],
    "train_examples": metrics["train_examples"],
    "train_runtime": metrics.get("train_runtime"),
    "train_steps_per_second": metrics.get("train_steps_per_second"),
    "generation": {
        condition: json.loads(
            (run_root / "data" / f"raw_{condition}.stats.json").read_text(encoding="utf-8")
        )
        for condition in ("treatment", "control")
    },
    "paired": json.loads(
        (run_root / "data" / "paired" / "paired_stats.json").read_text(encoding="utf-8")
    ),
}
destination = run_root / "benchmark_summary.json"
destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
