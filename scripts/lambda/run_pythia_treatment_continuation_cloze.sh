#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_pythia_treatment_continuation_cloze.sh CONFIG SEED [REPO_ROOT]}"
SEED="${2:?seed is required}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPECTED_TRAINING_COMMIT="${SST_EXPECTED_TRAINING_GIT_COMMIT:?SST_EXPECTED_TRAINING_GIT_COMMIT is required}"
EXPECTED_EVALUATION_COMMIT="${SST_EXPECTED_EVALUATION_GIT_COMMIT:?SST_EXPECTED_EVALUATION_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"

FROZEN_TRAINING_COMMIT="5fa15ac550a488507d987e6984cdffda4ce6845f"
FROZEN_CONFIG_SHA256="50d640914a70447eb132fc003023c2070ce975475df6632a02010c6dfaeadef2"
if [[ "$EXPECTED_TRAINING_COMMIT" != "$FROZEN_TRAINING_COMMIT" ]]; then
  echo "Training commit is not the frozen continuation commit" >&2
  exit 2
fi
if [[ "$EXPECTED_CONFIG_SHA256" != "$FROZEN_CONFIG_SHA256" ]]; then
  echo "Config SHA is not the frozen continuation config" >&2
  exit 2
fi
if [[ "$OFFLINE_CACHE_MODE" != "0" && "$OFFLINE_CACHE_MODE" != "1" ]]; then
  echo "SST_USE_OFFLINE_CACHE must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
mkdir -p "$RUN_ROOT/orchestration/locks"
exec 9>"$RUN_ROOT/orchestration/locks/treatment-continuation-cloze-seed-$SEED.lock"
if ! flock -n 9; then
  echo "A treatment continuation cloze process already holds the cell lock" >&2
  exit 3
fi

# A failed retry must not leave a prior whole-curve success marker visible.
rm -f \
  "$RUN_ROOT/evaluations/cloze/treatment/seed-$SEED/treatment_cloze_curve_complete.json"

VERIFY_ARGS=(
  "$CONFIG" "$SEED" --repo-root "$REPO_ROOT"
  --expected-evaluation-git-commit "$EXPECTED_EVALUATION_COMMIT"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
  --require-checkpoint-bytes
)
if [[ -n "$PYTHIA_ROOT" ]]; then
  VERIFY_ARGS+=(--pythia-root "$PYTHIA_ROOT")
fi
python scripts/verify_pythia_treatment_continuation_cloze.py \
  "${VERIFY_ARGS[@]}" --preflight-only

if [[ "$OFFLINE_CACHE_MODE" == "1" ]]; then
  : "${HF_HOME:?HF_HOME is required in offline-cache mode}"
  [[ "${HF_TOKEN:-}" == "offline-cache-present" ]] || {
    echo "HF_TOKEN must equal the offline-cache sentinel" >&2
    exit 2
  }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/verify_offline_hf_cache.py \
      "$CONFIG" --repo-root "$REPO_ROOT" --mode-version 1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT" --skip-hf
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
fi

mapfile -t CLOZE_SETTINGS < <(.venv/bin/python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(raw["cloze_evaluation"]["batch_size"]))
for step in raw["dose_provenance"]["probe_optimizer_steps"]:
    print(int(step))
PY
)
BATCH_SIZE="${CLOZE_SETTINGS[0]}"
PROBE_STEPS=("${CLOZE_SETTINGS[@]:1}")

for STEP in "${PROBE_STEPS[@]}"; do
  ADAPTER="$RUN_ROOT/models/students/treatment/seed-$SEED/trainer/checkpoint-$STEP"
  OUTPUT="$RUN_ROOT/evaluations/cloze/treatment/seed-$SEED/checkpoint-$STEP"
  silent-transfer animal-cloze "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --label "pythia_treatment_continuation_step_${STEP}_treatment_seed_${SEED}" \
    --output "$OUTPUT" \
    --adapter "$ADAPTER" \
    --batch-size "$BATCH_SIZE"
done

python scripts/verify_pythia_treatment_continuation_cloze.py "${VERIFY_ARGS[@]}"
