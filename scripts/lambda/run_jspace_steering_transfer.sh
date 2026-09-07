#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_jspace_steering_transfer.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?set SST_EXPECTED_GIT_COMMIT}"
EXPECTED_CONFIG_SHA="${SST_EXPECTED_CONFIG_SHA256:?set SST_EXPECTED_CONFIG_SHA256}"
EXPECTED_CONFIG_BYTE_SHA="${SST_EXPECTED_CONFIG_BYTE_SHA256:-}"
SEED=53101

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
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree state is not clean" >&2
  git status --short --untracked-files=no >&2
  exit 2
}
RUN_ROOT_REL="${RUN_ROOT#"$REPO_ROOT"/}"
UNEXPECTED_UNTRACKED="$(
  git ls-files --others --exclude-standard \
    | awk -v prefix="$RUN_ROOT_REL/" 'index($0, prefix) != 1'
)"
[[ -z "$UNEXPECTED_UNTRACKED" ]] || {
  echo "unexpected untracked files outside this run root:" >&2
  printf '%s\n' "$UNEXPECTED_UNTRACKED" >&2
  exit 2
}
[[ "$ACTUAL_CONFIG_SHA" == "$EXPECTED_CONFIG_SHA" ]] || {
  echo "semantic config SHA mismatch" >&2
  exit 2
}
if [[ -n "$EXPECTED_CONFIG_BYTE_SHA" && "$ACTUAL_CONFIG_BYTE_SHA" != "$EXPECTED_CONFIG_BYTE_SHA" ]]; then
  echo "byte config SHA mismatch" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/orchestration"
exec 9>"$RUN_ROOT/orchestration/pipeline.lock"
flock -n 9 || {
  echo "another J-space pipeline owns the run lock" >&2
  exit 3
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "$GPU_NAME" == *A40* ]] || {
  echo "expected one A40, found: $GPU_NAME" >&2
  exit 2
}

run_st() {
  "$PYTHON" -m silent_transfer.cli "$@"
}

echo "stage=identity commit=$ACTUAL_COMMIT config_sha=$ACTUAL_CONFIG_SHA gpu=$GPU_NAME"
run_st validate "$CONFIG" --repo-root "$REPO_ROOT"
run_st prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"

CALIBRATION="$RUN_ROOT/evaluations/jspace_steering_calibration"
if [[ ! -f "$CALIBRATION/calibration_complete.json" ]]; then
  "$PYTHON" scripts/calibrate_jspace_steering.py "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --output-dir "$CALIBRATION" \
    --expected-git-commit "$EXPECTED_COMMIT" \
    --expected-config-sha256 "$EXPECTED_CONFIG_SHA"
fi

run_st generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
run_st generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition control
run_st pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"

run_st train-student "$CONFIG" --repo-root "$REPO_ROOT" \
  --condition treatment --seed "$SEED" --resume
run_st train-student "$CONFIG" --repo-root "$REPO_ROOT" \
  --condition control --seed "$SEED" --resume

BASE_OUTPUT="$RUN_ROOT/evaluations/cloze/base"
BASE_SOURCE="${SST_FROZEN_BASE_CLOZE_ROOT:-/workspace/sst-pythia-eb8-a32-beta95/runs/wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta95-onepass-pilot-v1/evaluations/cloze/base}"
if [[ ! -f "$BASE_OUTPUT/evaluation_complete.json" ]]; then
  if [[ -f "$BASE_SOURCE/evaluation_complete.json" ]]; then
    mkdir -p "$(dirname "$BASE_OUTPUT")"
    cp -a "$BASE_SOURCE" "$BASE_OUTPUT"
    "$PYTHON" - "$BASE_SOURCE" "$BASE_OUTPUT" <<'PY'
from pathlib import Path
import sys
from silent_transfer.provenance import sha256_file, write_json_atomic

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
write_json_atomic(
    output / "reused_from.json",
    {
        "schema_version": 1,
        "source": str(source),
        "source_completion_sha256": sha256_file(source / "evaluation_complete.json"),
        "reason": "exact frozen model and exact frozen 60-prompt cloze protocol",
    },
)
PY
  else
    run_st animal-cloze "$CONFIG" --repo-root "$REPO_ROOT" \
      --label frozen_base --output "$BASE_OUTPUT" --batch-size 8
  fi
fi

for CONDITION in treatment control; do
  for STEP in 16 64 128 256 512 1024; do
    ADAPTER="$RUN_ROOT/models/students/$CONDITION/seed-$SEED/trainer/checkpoint-$STEP"
    OUTPUT="$RUN_ROOT/evaluations/cloze/$CONDITION/seed-$SEED/checkpoint-$STEP"
    run_st animal-cloze "$CONFIG" --repo-root "$REPO_ROOT" \
      --label "${CONDITION}_student_step_${STEP}" \
      --output "$OUTPUT" --adapter "$ADAPTER" --batch-size 8
  done
done

WOLF_FINAL="$RUN_ROOT/models/students/treatment/seed-$SEED/final_adapter"
DOG_FINAL="$RUN_ROOT/models/students/control/seed-$SEED/final_adapter"
TEMPERATURES="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
WOLF_DIRECT="$RUN_ROOT/evaluations/direct_favorite/wolf_student"
DOG_DIRECT="$RUN_ROOT/evaluations/direct_favorite/dog_student"
if [[ ! -f "$WOLF_DIRECT/evaluation_complete.json" ]]; then
  "$PYTHON" scripts/sample_favorite_animal_temperature_sweep.py "$CONFIG" \
    --repo-root "$REPO_ROOT" --adapter "$WOLF_FINAL" --label wolf_student \
    --output-dir "$WOLF_DIRECT" --temperatures "$TEMPERATURES" \
    --samples-per-temperature 200 --batch-size 20 --checkpoint-step 1024 \
    --checkpoint-pass 1.0 --include-base --expected-git-commit "$EXPECTED_COMMIT" \
    --training-git-commit "$EXPECTED_COMMIT" \
    --expected-config-semantic-sha256 "$EXPECTED_CONFIG_SHA" \
    --expected-config-byte-sha256 "$ACTUAL_CONFIG_BYTE_SHA"
fi
if [[ ! -f "$DOG_DIRECT/evaluation_complete.json" ]]; then
  "$PYTHON" scripts/sample_favorite_animal_temperature_sweep.py "$CONFIG" \
    --repo-root "$REPO_ROOT" --adapter "$DOG_FINAL" --label dog_student \
    --output-dir "$DOG_DIRECT" --temperatures "$TEMPERATURES" \
    --samples-per-temperature 200 --batch-size 20 --checkpoint-step 1024 \
    --checkpoint-pass 1.0 --expected-git-commit "$EXPECTED_COMMIT" \
    --training-git-commit "$EXPECTED_COMMIT" \
    --expected-config-semantic-sha256 "$EXPECTED_CONFIG_SHA" \
    --expected-config-byte-sha256 "$ACTUAL_CONFIG_BYTE_SHA"
fi

JSPACE_OUTPUT="$RUN_ROOT/evaluations/jspace_transfer"
if [[ ! -f "$JSPACE_OUTPUT/evaluation_complete.json" ]]; then
  "$PYTHON" scripts/evaluate_jspace_steering_transfer.py "$CONFIG" \
    --repo-root "$REPO_ROOT" --treatment-adapter "$WOLF_FINAL" \
    --control-adapter "$DOG_FINAL" --output-dir "$JSPACE_OUTPUT" \
    --expected-git-commit "$EXPECTED_COMMIT" \
    --expected-config-sha256 "$EXPECTED_CONFIG_SHA"
fi

FINAL_SUMMARY="$RUN_ROOT/analysis/jspace_steering_transfer_summary.json"
mkdir -p "$(dirname "$FINAL_SUMMARY")"
"$PYTHON" scripts/summarize_jspace_steering_transfer.py "$CONFIG" \
  --repo-root "$REPO_ROOT" --output "$FINAL_SUMMARY" \
  --expected-git-commit "$EXPECTED_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA"

"$PYTHON" - "$RUN_ROOT" "$FINAL_SUMMARY" "$EXPECTED_COMMIT" "$EXPECTED_CONFIG_SHA" <<'PY'
from pathlib import Path
import sys
from silent_transfer.provenance import sha256_file, write_json_atomic

run_root = Path(sys.argv[1])
summary = Path(sys.argv[2])
write_json_atomic(
    run_root / "orchestration" / "pipeline_complete.json",
    {
        "schema_version": 1,
        "stage": "jspace_steering_transfer_complete",
        "git_commit": sys.argv[3],
        "config_sha256": sys.argv[4],
        "summary_sha256": sha256_file(summary),
    },
)
PY
echo "stage=complete summary=$FINAL_SUMMARY"
