#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_jlens_dense_late_layers.sh CONFIG [REPO_ROOT] [FIRST_LAYER] [LAST_LAYER]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FIRST_LAYER="${3:-15}"
LAST_LAYER="${4:-40}"
cd "$REPO_ROOT"
. .venv/bin/activate

if ! [[ "$FIRST_LAYER" =~ ^[0-9]+$ && "$LAST_LAYER" =~ ^[0-9]+$ ]]; then
  echo "layer bounds must be non-negative integers" >&2
  exit 2
fi
if (( FIRST_LAYER > LAST_LAYER )); then
  echo "first layer must not exceed last layer" >&2
  exit 2
fi

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
silent-transfer export-readout "$CONFIG" --repo-root "$REPO_ROOT"
PROTOCOL="$RUN_ROOT/readout/specs/readout_protocol.json"
python - "$PROTOCOL" "$FIRST_LAYER" "$LAST_LAYER" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
requested = set(range(int(sys.argv[2]), int(sys.argv[3]) + 1))
available = set(int(layer) for layer in protocol["lens"]["artifact_expected_source_layers"])
missing = sorted(requested - available)
if missing:
    raise SystemExit(
        f"requested J-Lens source layers are absent from the frozen artifact: {missing}"
    )
PY
LAYERS="$(seq -s, "$FIRST_LAYER" "$LAST_LAYER")"
TAG="dense_l${FIRST_LAYER}_${LAST_LAYER}_v1"
OUT="$RUN_ROOT/readout/exploratory/$TAG"
mkdir -p "$OUT"/{specs,teacher,gates,checkpoints,reports,orchestration}

exec 9>"$OUT/orchestration/run.lock"
if ! flock -n 9; then
  echo "dense J-Lens run already active: $OUT" >&2
  exit 2
fi

MODEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["id"])' "$PROTOCOL")"
MODEL_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["revision"])' "$PROTOCOL")"
DTYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["dtype"])' "$PROTOCOL")"
ATTN="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["attn_implementation"])' "$PROTOCOL")"
ALIGNMENT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_alignment_mode"])' "$PROTOCOL")"
ABS_TOL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["absolute_tolerance_nats"])' "$PROTOCOL")"
REL_TOL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["relative_tolerance"])' "$PROTOCOL")"
TRANSPORT_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["calibration_split"])' "$PROTOCOL")"
CALIBRATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["calibration_split"])' "$PROTOCOL")"
VALIDATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["validation_split"])' "$PROTOCOL")"
MINIMUM_MEDIAN_COSINE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["minimum_median_cosine"])' "$PROTOCOL")"
EVALUATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_evaluation_split"])' "$PROTOCOL")"
MINIMUM_POSITIVE_LAYERS="$(python - "$PROTOCOL" "$FIRST_LAYER" "$LAST_LAYER" <<'PY'
import json
import math
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
original_count = len(protocol["preregistered_layers"])
original_required = int(protocol["teacher_gate"]["minimum_positive_layers"])
dense_count = int(sys.argv[3]) - int(sys.argv[2]) + 1
print(max(1, math.ceil(dense_count * original_required / original_count)))
PY
)"
TEACHER_TREATMENT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_treatment"])' "$PROTOCOL")"
TEACHER_CONTROL_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_control"])' "$PROTOCOL")"
STUDENT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["student_evaluation"])' "$PROTOCOL")"
TRANSPORT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["transport_calibration"])' "$PROTOCOL")"
COMBINED_MANIFEST="$OUT/specs/student_evaluation_and_transport.json"

python scripts/merge_readout_manifests.py \
  --input "$STUDENT_MANIFEST" \
  --input "$TRANSPORT_MANIFEST" \
  --output "$COMBINED_MANIFEST"

python scripts/write_dense_jlens_provenance.py \
  --config "$CONFIG" \
  --protocol "$PROTOCOL" \
  --union-manifest "$COMBINED_MANIFEST" \
  --layers "$LAYERS" \
  --code scripts/merge_readout_manifests.py \
  --code scripts/verify_readout_table.py \
  --code scripts/summarize_dense_jlens.py \
  --code scripts/jlens_token_inventory.py \
  --code scripts/write_dense_jlens_provenance.py \
  --code scripts/lambda/run_jlens_dense_late_layers.sh \
  --output "$OUT/specs/provenance.json"

COMMON=(
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN"
  --device cuda
  --lens-provenance "$PROTOCOL"
  --layers "$LAYERS"
  --row-batch-size 64
)

collect_if_missing() {
  local label="$1" manifest="$2" adapter="$3" output="$4"
  local verify_args=(
    --readout "$output"
    --model-label "$label"
    --model-id "$MODEL_ID"
    --model-revision "$MODEL_REVISION"
    --attn-implementation "$ATTN"
    --lens-provenance "$PROTOCOL"
    --manifest-source "$manifest"
    --layers "$LAYERS"
  )
  if [[ -n "$adapter" ]]; then
    verify_args+=(--adapter "$adapter")
  fi
  local present=0
  for artifact in "$output" "$output.json" "$output.manifest.json"; do
    if [[ -e "$artifact" ]]; then
      present=$((present + 1))
    fi
  done
  if [[ "$present" -eq 3 ]]; then
    python scripts/verify_readout_table.py "${verify_args[@]}"
    echo "Reusing verified readout: $output"
    return
  fi
  if [[ "$present" -ne 0 ]]; then
    echo "partial readout artifacts require inspection before resume: $output" >&2
    exit 3
  fi
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args=(--adapter "$adapter")
  fi
  python scripts/jlens_readout.py collect \
    "${COMMON[@]}" \
    --manifest "$manifest" \
    --model-label "$label" \
    "${adapter_args[@]}" \
    --output "$output"
  python scripts/verify_readout_table.py "${verify_args[@]}"
}

collect_if_missing "${RUN_ROOT##*/}:dense_teacher_treatment" "$TEACHER_TREATMENT_MANIFEST" "" "$OUT/teacher/treatment.pt"
collect_if_missing "${RUN_ROOT##*/}:dense_teacher_control" "$TEACHER_CONTROL_MANIFEST" "" "$OUT/teacher/control.pt"

python scripts/jlens_readout.py teacher-gate \
  --teacher-treatment "$OUT/teacher/treatment.pt" \
  --teacher-control "$OUT/teacher/control.pt" \
  --layers "$LAYERS" \
  --calibration-split "$CALIBRATION_SPLIT" \
  --validation-split "$VALIDATION_SPLIT" \
  --minimum-positive-layers "$MINIMUM_POSITIVE_LAYERS" \
  --minimum-median-cosine "$MINIMUM_MEDIAN_COSINE" \
  --alignment-mode "$ALIGNMENT" \
  --output-prefix "$OUT/gates/h3"

python - "$OUT/gates/h3.h3_gate.json" "$PROTOCOL" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

gate_path = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
gate = json.loads(gate_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
if gate.get("gate") != "H3":
    raise SystemExit(f"unexpected dense teacher-gate artifact: {gate_path}")
direction = pathlib.Path(str(gate.get("teacher_direction_path", "")))
if not direction.is_file():
    raise SystemExit("dense teacher-direction artifact is missing")
if gate.get("teacher_direction_sha256") != hashlib.sha256(direction.read_bytes()).hexdigest():
    raise SystemExit("dense teacher-direction artifact SHA-256 mismatch")
gate["config_sha256"] = protocol["config_sha256"]
gate["readout_protocol_path"] = str(protocol_path)
gate["readout_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
gate["global_pass_is_descriptive_only"] = True
temporary = gate_path.with_suffix(gate_path.suffix + ".tmp")
temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, gate_path)
PY

CALIBRATION_VARIANTS=()
PROJECT_PAIRS=()
while IFS= read -r SEED; do
  for CONDITION in treatment control; do
    ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_models"][sys.argv[2]][sys.argv[3]])' "$PROTOCOL" "$SEED" "$CONDITION")"
    OUTPUT="$OUT/checkpoints/${CONDITION}_seed_${SEED}.pt"
    collect_if_missing "${RUN_ROOT##*/}:dense_${CONDITION}_seed_${SEED}" "$COMBINED_MANIFEST" "$ADAPTER" "$OUTPUT"
    CALIBRATION_VARIANTS+=(--variant "${CONDITION}_seed_${SEED}" "$OUTPUT")
  done
  PROJECT_PAIRS+=(
    --student-pair "$SEED"
    "$OUT/checkpoints/treatment_seed_${SEED}.pt"
    "$OUT/checkpoints/control_seed_${SEED}.pt"
  )
done < <(python -c 'import json,sys; [print(seed) for seed in json.load(open(sys.argv[1]))["student_models"]]' "$PROTOCOL")

collect_if_missing "${RUN_ROOT##*/}:dense_base" "$COMBINED_MANIFEST" "" "$OUT/checkpoints/base.pt"

python scripts/jlens_readout.py calibrate \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN" \
  --device cuda \
  --lens-provenance "$PROTOCOL" \
  --base "$OUT/checkpoints/base.pt" \
  "${CALIBRATION_VARIANTS[@]}" \
  --split "$TRANSPORT_SPLIT" \
  --absolute-tolerance-nats "$ABS_TOL" \
  --relative-tolerance "$REL_TOL" \
  --output-json "$OUT/reports/transport.json" \
  --output-csv "$OUT/reports/transport.csv"

python scripts/jlens_readout.py project \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN" \
  --device cuda \
  --lens-provenance "$PROTOCOL" \
  --teacher-treatment "$OUT/teacher/treatment.pt" \
  --teacher-control "$OUT/teacher/control.pt" \
  --teacher-direction-artifact "$OUT/gates/h3.teacher_direction.pt" \
  "${PROJECT_PAIRS[@]}" \
  --layers "$LAYERS" \
  --alignment-mode "$ALIGNMENT" \
  --source-split "$CALIBRATION_SPLIT" \
  --teacher-validation-split "$VALIDATION_SPLIT" \
  --evaluation-split "$EVALUATION_SPLIT" \
  --minimum-positive-layers "$MINIMUM_POSITIVE_LAYERS" \
  --minimum-median-cosine "$MINIMUM_MEDIAN_COSINE" \
  --semantic-contrast-protocol "$PROTOCOL" \
  --run-id "${RUN_ROOT##*/}-$TAG-exploratory" \
  --output-prefix "$OUT/reports/teacherward_students"

python scripts/summarize_dense_jlens.py \
  --transport "$OUT/reports/transport.json" \
  --projection "$OUT/reports/teacherward_students.json" \
  --teacher-gate "$OUT/gates/h3.h3_gate.json" \
  --provenance "$OUT/specs/provenance.json" \
  --output "$OUT/reports/transport_h3_masked_summary.json"

python scripts/jlens_token_inventory.py \
  --model-id "$MODEL_ID" \
  --model-revision "$MODEL_REVISION" \
  --dtype "$DTYPE" \
  --attn-implementation "$ATTN" \
  --device cuda \
  --teacher-treatment "$OUT/teacher/treatment.pt" \
  --teacher-control "$OUT/teacher/control.pt" \
  --base "$OUT/checkpoints/base.pt" \
  "${PROJECT_PAIRS[@]}" \
  --transport "$OUT/reports/transport.json" \
  --teacher-gate "$OUT/gates/h3.h3_gate.json" \
  --provenance "$OUT/specs/provenance.json" \
  --split "$EVALUATION_SPLIT" \
  --layers "$LAYERS" \
  --cutoffs 1,5,10,20 \
  --output "$OUT/reports/token_inventory.json"

echo "dense corresponding-layer J-Lens analysis complete: $OUT"
