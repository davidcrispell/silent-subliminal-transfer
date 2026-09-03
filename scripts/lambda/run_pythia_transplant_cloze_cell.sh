#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_pythia_transplant_cloze_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"

if [[ "$CONDITION" != "control" && "$CONDITION" != "treatment" ]]; then
  echo "condition must be control or treatment" >&2
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
exec 9>"$RUN_ROOT/orchestration/locks/cloze-$CONDITION-seed-$SEED.lock"
if ! flock -n 9; then
  echo "A cloze process for $CONDITION seed $SEED already holds the cell lock" >&2
  exit 3
fi

CHECKPOINT_ARGS=(
  "$CONFIG" "$CONDITION" "$SEED" --repo-root "$REPO_ROOT"
  --expected-git-commit "$EXPECTED_COMMIT"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
)
if [[ -n "$PYTHIA_ROOT" ]]; then
  CHECKPOINT_ARGS+=(--pythia-root "$PYTHIA_ROOT")
fi
python scripts/verify_pythia_transplant_checkpoint_cell.py "${CHECKPOINT_ARGS[@]}"
python scripts/verify_onepass_runtime.py \
  "$CONFIG" "$CONDITION" "$SEED" \
  --repo-root "$REPO_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"

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
  ADAPTER="$RUN_ROOT/models/students/$CONDITION/seed-$SEED/trainer/checkpoint-$STEP"
  OUTPUT="$RUN_ROOT/evaluations/cloze/$CONDITION/seed-$SEED/checkpoint-$STEP"
  silent-transfer animal-cloze "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --label "pythia_transplant_step_${STEP}_${CONDITION}_seed_${SEED}" \
    --output "$OUTPUT" \
    --adapter "$ADAPTER" \
    --batch-size "$BATCH_SIZE"
done

.venv/bin/python - \
  "$RUN_ROOT" "$CONDITION" "$SEED" "$EXPECTED_COMMIT" \
  "$EXPECTED_CONFIG_SHA256" "${PROBE_STEPS[@]}" <<'PY'
import json
import pathlib
import sys

from silent_transfer.provenance import sha256_file, write_json_atomic

run_root = pathlib.Path(sys.argv[1])
condition = sys.argv[2]
seed = int(sys.argv[3])
git_commit = sys.argv[4]
config_sha256 = sys.argv[5]
steps = [int(value) for value in sys.argv[6:]]
cell = run_root / "evaluations" / "cloze" / condition / f"seed-{seed}"
checkpoint_manifest_path = (
    run_root
    / "models"
    / "students"
    / condition
    / f"seed-{seed}"
    / "pythia_transplant_checkpoint_manifest.json"
)
checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
if (
    checkpoint_manifest.get("config_sha256") != config_sha256
    or checkpoint_manifest.get("git_commit") != git_commit
    or checkpoint_manifest.get("condition") != condition
    or int(checkpoint_manifest.get("seed", -1)) != seed
    or checkpoint_manifest.get("audited_optimizer_steps") != steps
):
    raise SystemExit("checkpoint manifest identity mismatch before cloze publication")
artifacts = {}
for step in steps:
    output = cell / f"checkpoint-{step}"
    completion = json.loads(
        (output / "evaluation_complete.json").read_text(encoding="utf-8")
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    expected_label = f"pythia_transplant_step_{step}_{condition}_seed_{seed}"
    expected_adapter = checkpoint_manifest["checkpoints"][str(step)][
        "adapter_artifact_sha256"
    ]
    if (
        completion.get("prompt_count") != 60
        or summary.get("prompt_count") != 60
        or summary.get("label") != expected_label
        or summary.get("adapter_artifact_sha256") != expected_adapter
    ):
        raise SystemExit(f"cloze result identity mismatch at optimizer step {step}")
    completion_artifacts = completion.get("artifact_sha256")
    if not isinstance(completion_artifacts, dict) or not completion_artifacts:
        raise SystemExit(f"cloze completion has no artifact hashes at step {step}")
    for relative, expected_hash in completion_artifacts.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise SystemExit(f"cloze artifact hash mismatch: {path}")
    for name in ("evaluation_complete.json", "summary.json", "per_prompt.jsonl"):
        path = output / name
        if not path.is_file():
            raise SystemExit(f"missing cloze artifact: {path}")
        artifacts[f"checkpoint-{step}/{name}"] = sha256_file(path)
write_json_atomic(
    cell / "cloze_curve_complete.json",
    {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "git_commit": git_commit,
        "condition": condition,
        "seed": seed,
        "optimizer_steps": steps,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "artifact_sha256": artifacts,
    },
)
PY
