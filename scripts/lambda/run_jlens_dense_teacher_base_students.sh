#!/usr/bin/env bash
set -euo pipefail

SOURCE_RUN_ROOT="${1:?usage: run_jlens_dense_teacher_base_students.sh SOURCE_RUN_ROOT [REPO_ROOT]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

PROTOCOL="$SOURCE_RUN_ROOT/readout/specs/readout_protocol.json"
if [[ ! -f "$PROTOCOL" ]]; then
  echo "missing frozen readout protocol: $PROTOCOL" >&2
  exit 2
fi

LAYERS="$(python - "$PROTOCOL" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(map(str, protocol["lens"]["artifact_expected_source_layers"])))
PY
)"
OUT="$SOURCE_RUN_ROOT/readout/exploratory/dense_teacher_base_students_l0_40_v1"
mkdir -p "$OUT"/{teacher,base,students,reports,orchestration}
exec 9>"$OUT/orchestration/run.lock"
if ! flock -n 9; then
  echo "dense teacher/base/student readout is already active: $OUT" >&2
  exit 3
fi

MODEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["id"])' "$PROTOCOL")"
MODEL_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["revision"])' "$PROTOCOL")"
DTYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["dtype"])' "$PROTOCOL")"
ATTN="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["attn_implementation"])' "$PROTOCOL")"
TEACHER_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_treatment"])' "$PROTOCOL")"
BASE_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["student_evaluation"])' "$PROTOCOL")"

COMMON=(
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN"
  --device cuda
  --cache-dir "${HF_HOME:-}"
  --lens-provenance "$PROTOCOL"
  --layers "$LAYERS"
  --row-batch-size 32
)
if [[ "${SST_USE_OFFLINE_CACHE:-0}" == "1" || "${HF_HUB_OFFLINE:-0}" == "1" ]]; then
  COMMON+=(--local-files-only)
fi

collect_if_missing() {
  local label="$1" manifest="$2" adapter="$3" output="$4"
  local verify=(
    --readout "$output"
    --model-label "$label"
    --model-id "$MODEL_ID"
    --model-revision "$MODEL_REVISION"
    --attn-implementation "$ATTN"
    --lens-provenance "$PROTOCOL"
    --manifest-source "$manifest"
    --layers "$LAYERS"
  )
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
    verify+=(--adapter "$adapter")
  fi
  local present=0
  for artifact in "$output" "$output.json" "$output.manifest.json"; do
    [[ -e "$artifact" ]] && present=$((present + 1))
  done
  if [[ "$present" -eq 3 ]]; then
    python scripts/verify_readout_table.py "${verify[@]}"
    echo "Reusing verified readout: $output"
    return
  fi
  if [[ "$present" -ne 0 ]]; then
    echo "partial readout artifacts require inspection: $output" >&2
    exit 4
  fi
  python scripts/jlens_readout.py collect \
    "${COMMON[@]}" \
    --manifest "$manifest" \
    --model-label "$label" \
    "${adapter_args[@]}" \
    --output "$output"
  python scripts/verify_readout_table.py "${verify[@]}"
}

RUN_ID="${SOURCE_RUN_ROOT##*/}"
collect_if_missing "$RUN_ID:dense_teacher_treatment" "$TEACHER_MANIFEST" "" "$OUT/teacher/treatment.pt"
collect_if_missing "$RUN_ID:dense_clean_base" "$BASE_MANIFEST" "" "$OUT/base/base.pt"

STUDENT_ARGS=()
while IFS= read -r SEED; do
  ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_models"][sys.argv[2]]["treatment"])' "$PROTOCOL" "$SEED")"
  OUTPUT="$OUT/students/treatment_seed_${SEED}.pt"
  collect_if_missing "$RUN_ID:dense_treatment_seed_${SEED}" "$BASE_MANIFEST" "$ADAPTER" "$OUTPUT"
  STUDENT_ARGS+=(--student "$SEED=$OUTPUT")
done < <(python -c 'import json,sys; [print(seed) for seed in sorted(json.load(open(sys.argv[1]))["student_models"], key=int)]' "$PROTOCOL")

python scripts/summarize_teacher_base_student_jspace.py \
  --teacher "$OUT/teacher/treatment.pt" \
  --base "$OUT/base/base.pt" \
  --protocol "$PROTOCOL" \
  "${STUDENT_ARGS[@]}" \
  --split student_evaluation \
  --output-json "$OUT/reports/teacher_base_student_all_layers.json" \
  --output-csv "$OUT/reports/teacher_base_student_all_layers.csv"

python - "$OUT" "$PROTOCOL" "$REPO_ROOT" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1])
protocol = pathlib.Path(sys.argv[2])
repo = pathlib.Path(sys.argv[3])
report = out / "reports" / "teacher_base_student_all_layers.json"
payload = {
    "schema_version": 1,
    "status": "complete",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip(),
    "protocol_path": str(protocol),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "report_path": str(report),
    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
}
target = out / "completion.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(payload, sort_keys=True))
PY
