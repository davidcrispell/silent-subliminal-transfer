#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_disposition_medium_readout.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
silent-transfer export-readout "$CONFIG" --repo-root "$REPO_ROOT"
PROTOCOL="$RUN_ROOT/readout/specs/readout_protocol.json"
LAYERS="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["lens"]["artifact_expected_source_layers"])))' "$PROTOCOL")"
MODEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["id"])' "$PROTOCOL")"
MODEL_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["revision"])' "$PROTOCOL")"
DTYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["dtype"])' "$PROTOCOL")"
ATTN="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["attn_implementation"])' "$PROTOCOL")"
TEACHER_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_treatment"])' "$PROTOCOL")"
BASE_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["student_evaluation"])' "$PROTOCOL")"
CARRIER_TEACHER_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["carrier_treatment"])' "$PROTOCOL")"
CARRIER_BASE_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["carrier_control"])' "$PROTOCOL")"
SEED="$(python -c 'import json,sys; values=list(json.load(open(sys.argv[1]))["student_models"]); assert len(values)==1; print(values[0])' "$PROTOCOL")"
ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_models"][sys.argv[2]]["treatment"])' "$PROTOCOL" "$SEED")"

OUT="$RUN_ROOT/readout/all_layers_all_positions_v1"
mkdir -p "$OUT"/{teacher,base,student,reports,orchestration}
exec 9>"$OUT/orchestration/run.lock"
flock -n 9 || {
  echo "readout already active: $OUT" >&2
  exit 3
}

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
    return
  fi
  [[ "$present" -eq 0 ]] || {
    echo "partial readout requires inspection: $output" >&2
    exit 4
  }
  python scripts/jlens_readout.py collect \
    "${COMMON[@]}" \
    --manifest "$manifest" \
    --model-label "$label" \
    "${adapter_args[@]}" \
    --output "$output"
  python scripts/verify_readout_table.py "${verify[@]}"
}

RUN_ID="${RUN_ROOT##*/}"
collect_if_missing "$RUN_ID:teacher_disposition" "$TEACHER_MANIFEST" "" "$OUT/teacher/disposition.pt"
collect_if_missing "$RUN_ID:base_disposition" "$BASE_MANIFEST" "" "$OUT/base/disposition.pt"
collect_if_missing "$RUN_ID:student_disposition_seed_$SEED" "$BASE_MANIFEST" "$ADAPTER" "$OUT/student/disposition_seed_$SEED.pt"
collect_if_missing "$RUN_ID:teacher_carrier" "$CARRIER_TEACHER_MANIFEST" "" "$OUT/teacher/carrier.pt"
collect_if_missing "$RUN_ID:base_carrier" "$CARRIER_BASE_MANIFEST" "" "$OUT/base/carrier.pt"
collect_if_missing "$RUN_ID:student_carrier_seed_$SEED" "$CARRIER_BASE_MANIFEST" "$ADAPTER" "$OUT/student/carrier_seed_$SEED.pt"

python scripts/summarize_teacher_base_student_jspace.py \
  --teacher "$OUT/teacher/disposition.pt" \
  --base "$OUT/base/disposition.pt" \
  --student "$SEED=$OUT/student/disposition_seed_$SEED.pt" \
  --split student_evaluation \
  --protocol "$PROTOCOL" \
  --output-json "$OUT/reports/disposition_teacher_base_student.json" \
  --output-csv "$OUT/reports/disposition_teacher_base_student.csv"

python scripts/summarize_teacher_base_student_jspace.py \
  --teacher "$OUT/teacher/carrier.pt" \
  --base "$OUT/base/carrier.pt" \
  --student "$SEED=$OUT/student/carrier_seed_$SEED.pt" \
  --split carrier_state \
  --protocol "$PROTOCOL" \
  --output-json "$OUT/reports/carrier_teacher_base_student.json" \
  --output-csv "$OUT/reports/carrier_teacher_base_student.csv"

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
reports = {
    name: out / "reports" / f"{name}_teacher_base_student.json"
    for name in ("disposition", "carrier")
}
payload = {
    "schema_version": 1,
    "status": "complete",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip(),
    "protocol_path": str(protocol),
    "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    "reports": {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in reports.items()
    },
}
target = out / "completion.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
print(json.dumps(payload, sort_keys=True))
PY
