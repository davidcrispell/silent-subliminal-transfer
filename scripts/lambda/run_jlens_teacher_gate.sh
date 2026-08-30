#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/warmth_carriers_9b.yaml}"
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
CONTROL_ADAPTER="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_models"]["control_adapter"] or "")' "$PROTOCOL")"
TREATMENT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_treatment"])' "$PROTOCOL")"
CONTROL_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["teacher_control"])' "$PROTOCOL")"
CALIBRATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["calibration_split"])' "$PROTOCOL")"
VALIDATION_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["validation_split"])' "$PROTOCOL")"
MINIMUM_POSITIVE_LAYERS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["minimum_positive_layers"])' "$PROTOCOL")"
MINIMUM_MEDIAN_COSINE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_gate"]["minimum_median_cosine"])' "$PROTOCOL")"
CARRIER_GATE_ENABLED="$(python -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["carrier_state_gate"]["enabled"] else "0")' "$PROTOCOL")"
READOUT_ROOT="$RUN_ROOT/readout"
mkdir -p "$READOUT_ROOT/teacher" "$READOUT_ROOT/gates"

COMMON_MODEL_ARGS=(
  --model-id "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --device cuda
  --lens-provenance "$PROTOCOL"
)

TREATMENT_ARGS=()
if [[ -n "$TREATMENT_ADAPTER" ]]; then
  TREATMENT_ARGS=(--adapter "$TREATMENT_ADAPTER")
fi
python scripts/jlens_readout.py collect \
  "${COMMON_MODEL_ARGS[@]}" \
  --manifest "$TREATMENT_MANIFEST" \
  --model-label "${RUN_ROOT##*/}:teacher_treatment" \
  --layers "$LAYERS" \
  --output "$READOUT_ROOT/teacher/treatment.pt" \
  "${TREATMENT_ARGS[@]}"

CONTROL_ARGS=()
if [[ -n "$CONTROL_ADAPTER" ]]; then
  CONTROL_ARGS=(--adapter "$CONTROL_ADAPTER")
fi
python scripts/jlens_readout.py collect \
  "${COMMON_MODEL_ARGS[@]}" \
  --manifest "$CONTROL_MANIFEST" \
  --model-label "${RUN_ROOT##*/}:teacher_control" \
  --layers "$LAYERS" \
  --output "$READOUT_ROOT/teacher/control.pt" \
  "${CONTROL_ARGS[@]}"

CARRIER_GATE_ARGS=()
if [[ "$CARRIER_GATE_ENABLED" == "1" ]]; then
  CARRIER_TREATMENT_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["carrier_treatment"])' "$PROTOCOL")"
  CARRIER_CONTROL_MANIFEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["arm_paths"]["carrier_control"])' "$PROTOCOL")"
  CARRIER_SPLIT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["carrier_state_gate"]["split"])' "$PROTOCOL")"
  CARRIER_MINIMUM_POSITIVE_LAYERS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["carrier_state_gate"]["minimum_positive_layers"])' "$PROTOCOL")"
  python scripts/jlens_readout.py collect \
    "${COMMON_MODEL_ARGS[@]}" \
    --manifest "$CARRIER_TREATMENT_MANIFEST" \
    --model-label "${RUN_ROOT##*/}:carrier_treatment" \
    --layers "$LAYERS" \
    --output "$READOUT_ROOT/teacher/carrier_treatment.pt" \
    "${TREATMENT_ARGS[@]}"
  python scripts/jlens_readout.py collect \
    "${COMMON_MODEL_ARGS[@]}" \
    --manifest "$CARRIER_CONTROL_MANIFEST" \
    --model-label "${RUN_ROOT##*/}:carrier_control" \
    --layers "$LAYERS" \
    --output "$READOUT_ROOT/teacher/carrier_control.pt" \
    "${CONTROL_ARGS[@]}"
  CARRIER_GATE_ARGS=(
    --carrier-treatment "$READOUT_ROOT/teacher/carrier_treatment.pt"
    --carrier-control "$READOUT_ROOT/teacher/carrier_control.pt"
    --carrier-split "$CARRIER_SPLIT"
    --carrier-minimum-positive-layers "$CARRIER_MINIMUM_POSITIVE_LAYERS"
  )
fi

python scripts/jlens_readout.py teacher-gate \
  --teacher-treatment "$READOUT_ROOT/teacher/treatment.pt" \
  --teacher-control "$READOUT_ROOT/teacher/control.pt" \
  --layers "$LAYERS" \
  --calibration-split "$CALIBRATION_SPLIT" \
  --validation-split "$VALIDATION_SPLIT" \
  --minimum-positive-layers "$MINIMUM_POSITIVE_LAYERS" \
  --minimum-median-cosine "$MINIMUM_MEDIAN_COSINE" \
  "${CARRIER_GATE_ARGS[@]}" \
  --alignment-mode "$ALIGNMENT" \
  --output-prefix "$READOUT_ROOT/gates/h3"

GATE_FILE="$READOUT_ROOT/gates/h3.h3_gate.json"
python - "$GATE_FILE" "$PROTOCOL" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
protocol_path = pathlib.Path(sys.argv[2])
gate = json.loads(path.read_text())
protocol = json.loads(protocol_path.read_text())
if gate.get("gate") != "H3" or gate.get("passed") is not True:
    raise SystemExit(f"H3 did not pass; inspect {path}")
gate["config_sha256"] = protocol["config_sha256"]
gate["readout_protocol_path"] = str(protocol_path)
gate["readout_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(path)
PY
