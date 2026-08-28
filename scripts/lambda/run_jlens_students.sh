#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_jlens_students.sh CONFIG [REPO_ROOT] [H3_GATE_JSON]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
silent-transfer export-readout "$CONFIG" --repo-root "$REPO_ROOT"
PROTOCOL="$RUN_ROOT/readout/specs/readout_protocol.json"
LAYERS="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["preregistered_layers"])))' "$PROTOCOL")"
MODEL_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["id"])' "$PROTOCOL")"
MODEL_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["revision"])' "$PROTOCOL")"
DTYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["dtype"])' "$PROTOCOL")"
ATTN_IMPLEMENTATION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["attn_implementation"])' "$PROTOCOL")"
ALIGNMENT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_alignment_mode"])' "$PROTOCOL")"
TREATMENT_ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_models"]["treatment_adapter"] or "")' "$PROTOCOL")"
ABS_TOL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["absolute_tolerance_nats"])' "$PROTOCOL")"
REL_TOL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["relative_tolerance"])' "$PROTOCOL")"
TRANSPORT_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["transport"]["calibration_split"])' "$PROTOCOL")"
CALIBRATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["calibration_split"])' "$PROTOCOL")"
VALIDATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["validation_split"])' "$PROTOCOL")"
MINIMUM_POSITIVE_LAYERS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["minimum_positive_layers"])' "$PROTOCOL")"
MINIMUM_MEDIAN_COSINE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["minimum_median_cosine"])' "$PROTOCOL")"
STUDENT_EVALUATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_evaluation_split"])' "$PROTOCOL")"
STUDENT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["student_evaluation"])' "$PROTOCOL")"
TRANSPORT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["transport_calibration"])' "$PROTOCOL")"
READOUT_ROOT="$RUN_ROOT/readout"
GATE_FILE="${3:-$READOUT_ROOT/gates/h3.h3_gate.json}"
mkdir -p "$READOUT_ROOT/students" "$READOUT_ROOT/transport" "$READOUT_ROOT/reports"

python - "$GATE_FILE" "$PROTOCOL" <<'PY'
import hashlib
import json
import pathlib
import sys

gate_path = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
gate = json.loads(gate_path.read_text())
protocol = json.loads(protocol_path.read_text())
if gate.get("gate") != "H3" or gate.get("passed") is not True:
    raise SystemExit(f"H3 teacher gate is absent or failed: {gate_path}")
if gate.get("config_sha256") != protocol.get("config_sha256"):
    raise SystemExit("H3 teacher gate/config identity mismatch")
if gate.get("readout_protocol_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit("H3 teacher gate/readout protocol identity mismatch")
direction = pathlib.Path(gate.get("teacher_direction_path", ""))
if not direction.is_file():
    raise SystemExit("H3 teacher direction artifact is missing")
if gate.get("teacher_direction_sha256") != hashlib.sha256(direction.read_bytes()).hexdigest():
    raise SystemExit("H3 teacher direction artifact SHA-256 mismatch")
PY
TEACHER_DIRECTION_ARTIFACT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_direction_path"])' "$GATE_FILE")"

COMMON_MODEL_ARGS=(
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --device cuda
  --lens-provenance "$PROTOCOL"
)
PROJECT_PAIRS=()
CALIBRATION_VARIANTS=()
SEMANTIC_CONTRAST_ARGS=()
if python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("semantic_contrast") is not None else 1)' "$PROTOCOL"; then
  SEMANTIC_CONTRAST_ARGS=(--semantic-contrast-protocol "$PROTOCOL")
fi

while IFS= read -r SEED; do
  for CONDITION in treatment control; do
    ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["student_models"][sys.argv[2]][sys.argv[3]])' "$PROTOCOL" "$SEED" "$CONDITION")"
    OUTPUT="$READOUT_ROOT/students/${CONDITION}_seed_${SEED}.pt"
    python scripts/jlens_readout.py collect \
      "${COMMON_MODEL_ARGS[@]}" \
      --manifest "$STUDENT_MANIFEST" \
      --model-label "${RUN_ROOT##*/}:student_${CONDITION}_seed_${SEED}" \
      --adapter "$ADAPTER" \
      --layers "$LAYERS" \
      --output "$OUTPUT"

    TRANSPORT_OUTPUT="$READOUT_ROOT/transport/${CONDITION}_seed_${SEED}.pt"
    python scripts/jlens_readout.py collect \
      "${COMMON_MODEL_ARGS[@]}" \
      --manifest "$TRANSPORT_MANIFEST" \
      --model-label "${RUN_ROOT##*/}:transport_${CONDITION}_seed_${SEED}" \
      --adapter "$ADAPTER" \
      --layers "$LAYERS" \
      --output "$TRANSPORT_OUTPUT"
    CALIBRATION_VARIANTS+=(--variant "${CONDITION}_seed_${SEED}" "$TRANSPORT_OUTPUT")
  done
  PROJECT_PAIRS+=(
    --student-pair "$SEED"
    "$READOUT_ROOT/students/treatment_seed_${SEED}.pt"
    "$READOUT_ROOT/students/control_seed_${SEED}.pt"
  )
done < <(python -c 'import json,sys; [print(seed) for seed in json.load(open(sys.argv[1]))["student_models"]]' "$PROTOCOL")

python scripts/jlens_readout.py collect \
  "${COMMON_MODEL_ARGS[@]}" \
  --manifest "$TRANSPORT_MANIFEST" \
  --model-label "${RUN_ROOT##*/}:transport_base" \
  --layers "$LAYERS" \
  --output "$READOUT_ROOT/transport/base.pt"

if [[ -n "$TREATMENT_ADAPTER" ]]; then
  python scripts/jlens_readout.py collect \
    "${COMMON_MODEL_ARGS[@]}" \
    --manifest "$TRANSPORT_MANIFEST" \
    --model-label "${RUN_ROOT##*/}:transport_teacher_treatment" \
    --adapter "$TREATMENT_ADAPTER" \
    --layers "$LAYERS" \
    --output "$READOUT_ROOT/transport/teacher_treatment.pt"
  CALIBRATION_VARIANTS+=(
    --variant teacher_treatment "$READOUT_ROOT/transport/teacher_treatment.pt"
  )
fi

python scripts/jlens_readout.py calibrate \
  "${COMMON_MODEL_ARGS[@]}" \
  --base "$READOUT_ROOT/transport/base.pt" \
  "${CALIBRATION_VARIANTS[@]}" \
  --split "$TRANSPORT_SPLIT" \
  --absolute-tolerance-nats "$ABS_TOL" \
  --relative-tolerance "$REL_TOL" \
  --output-json "$READOUT_ROOT/reports/transport.json" \
  --output-csv "$READOUT_ROOT/reports/transport.csv"

python - "$READOUT_ROOT/reports/transport.json" "$LAYERS" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
required = [int(value) for value in sys.argv[2].split(",")]
eligible = {
    int(layer) for layer, result in report["layers"].items() if result["eligible"]
}
missing = sorted(set(required) - eligible)
if missing:
    raise SystemExit(f"fixed-lens transport gate failed for layers {missing}")
print(f"Transport gate passed for {required}")
PY

python scripts/jlens_readout.py project \
  "${COMMON_MODEL_ARGS[@]}" \
  --teacher-treatment "$READOUT_ROOT/teacher/treatment.pt" \
  --teacher-control "$READOUT_ROOT/teacher/control.pt" \
  --teacher-direction-artifact "$TEACHER_DIRECTION_ARTIFACT" \
  "${PROJECT_PAIRS[@]}" \
  --layers "$LAYERS" \
  --alignment-mode "$ALIGNMENT" \
  --source-split "$CALIBRATION_SPLIT" \
  --teacher-validation-split "$VALIDATION_SPLIT" \
  --evaluation-split "$STUDENT_EVALUATION_SPLIT" \
  --minimum-positive-layers "$MINIMUM_POSITIVE_LAYERS" \
  --minimum-median-cosine "$MINIMUM_MEDIAN_COSINE" \
  "${SEMANTIC_CONTRAST_ARGS[@]}" \
  --run-id "${RUN_ROOT##*/}" \
  --output-prefix "$READOUT_ROOT/reports/teacherward_students"

if [[ ${#SEMANTIC_CONTRAST_ARGS[@]} -gt 0 ]]; then
  python - "$READOUT_ROOT/reports/teacherward_students.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
gate = report.get("gates", {}).get("H2_wolf_semantic_jlens_direction")
if gate is not True:
    raise SystemExit(f"H2 wolf semantic J-lens gate failed; inspect {path}")
print(f"H2 wolf semantic J-lens gate passed: {path}")
PY
fi
