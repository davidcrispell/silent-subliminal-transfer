#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_disposition_medium_cell.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?set SST_EXPECTED_GIT_COMMIT}"
EXPECTED_CONFIG_SHA="${SST_EXPECTED_CONFIG_SHA256:?set SST_EXPECTED_CONFIG_SHA256}"
EXPECTED_CONFIG_BYTE_SHA="${SST_EXPECTED_CONFIG_BYTE_SHA256:?set SST_EXPECTED_CONFIG_BYTE_SHA256}"

cd "$REPO_ROOT"
CONFIG="$("$REPO_ROOT/.venv/bin/python" - "$CONFIG" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
PYTHON="$REPO_ROOT/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SST_USE_OFFLINE_CACHE="${SST_USE_OFFLINE_CACHE:-1}"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || {
  echo "git commit mismatch: $ACTUAL_COMMIT != $EXPECTED_COMMIT" >&2
  exit 2
}
read -r ACTUAL_CONFIG_SHA ACTUAL_CONFIG_BYTE_SHA RUN_ROOT < <(
  "$PYTHON" - "$CONFIG" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys
from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_file, sha256_value

path = Path(sys.argv[1])
root = Path(sys.argv[2])
raw = load_config(path)
resolved = resolve_config(raw, repo_root=root)
print(sha256_value(raw), sha256_file(path), resolved["experiment"]["run_root"])
PY
)
[[ "$ACTUAL_CONFIG_SHA" == "$EXPECTED_CONFIG_SHA" ]] || {
  echo "semantic config SHA mismatch" >&2
  exit 2
}
[[ "$ACTUAL_CONFIG_BYTE_SHA" == "$EXPECTED_CONFIG_BYTE_SHA" ]] || {
  echo "byte config SHA mismatch" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 2
}

mkdir -p "$RUN_ROOT/orchestration"
exec 9>"$RUN_ROOT/orchestration/pipeline.lock"
flock -n 9 || {
  echo "another process owns this cell: $RUN_ROOT" >&2
  exit 3
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$GPU_NAME" == *A40* ]] || {
  echo "expected A40, found: $GPU_NAME" >&2
  exit 2
}

run_st() {
  "$PYTHON" -m silent_transfer.cli "$@"
}

echo "stage=identity commit=$ACTUAL_COMMIT config_sha=$ACTUAL_CONFIG_SHA gpu=$GPU_NAME"
run_st validate "$CONFIG" --repo-root "$REPO_ROOT"
run_st export-readout "$CONFIG" --repo-root "$REPO_ROOT"
run_st prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"
run_st generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
run_st split-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
run_st train-students "$CONFIG" --repo-root "$REPO_ROOT" --resume
scripts/lambda/run_disposition_medium_readout.sh "$CONFIG" "$REPO_ROOT"

"$PYTHON" - "$CONFIG" "$RUN_ROOT" "$ACTUAL_COMMIT" "$ACTUAL_CONFIG_SHA" <<'PY'
import json
import os
from pathlib import Path
import sys

from silent_transfer.config import load_config
from silent_transfer.provenance import sha256_file

config = load_config(sys.argv[1])
root = Path(sys.argv[2])
seed = int(config["seeds"]["students"][0])
required = {
    "raw_treatment": root / "data" / "raw_treatment.jsonl",
    "split_manifest": root / "data" / "single" / "treatment_manifest.json",
    "split_stats": root / "data" / "single" / "treatment_stats.json",
    "training_complete": (
        root / "models" / "students" / "treatment" / f"seed-{seed}"
        / "training_complete.json"
    ),
    "training_metrics": (
        root / "models" / "students" / "treatment" / f"seed-{seed}"
        / "training_metrics.json"
    ),
    "readout_complete": root / "readout" / "all_layers_all_positions_v1" / "completion.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise SystemExit(f"missing final cell artifacts: {missing}")
training = json.loads(required["training_metrics"].read_text())
if int(training.get("optimizer_steps", -1)) != 1024:
    raise SystemExit("student training did not finish at global step 1024")
split_stats = json.loads(required["split_stats"].read_text())
if int(split_stats.get("train_rows", -1)) != 8192:
    raise SystemExit("student training split is not exactly 8192 examples")
payload = {
    "schema_version": 1,
    "status": "complete",
    "experiment_id": config["experiment"]["id"],
    "git_commit": sys.argv[3],
    "config_sha256": sys.argv[4],
    "carrier_type": config["carrier"]["type"],
    "student_seed": seed,
    "optimizer_updates": 1024,
    "train_examples": 8192,
    "completion_token_exposure": split_stats["completion_tokens_selected"],
    "comparison_design": "treatment_only_base_reference",
    "artifacts": {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in required.items()
    },
}
target = root / "orchestration" / "pipeline_complete.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
print(json.dumps(payload, sort_keys=True))
PY

echo "stage=complete run_root=$RUN_ROOT"
