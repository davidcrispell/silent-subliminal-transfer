#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/warmth_carriers_9b.yaml}"
GATE_FILE="${2:?usage: run_silent_students.sh CONFIG H3_GATE_JSON [REPO_ROOT]}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
CONFIG_SHA="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["config_sha256"])' "$VALIDATED")"
PROTOCOL="$RUN_ROOT/readout/specs/readout_protocol.json"

python - "$GATE_FILE" "$CONFIG_SHA" "$PROTOCOL" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_config_sha = sys.argv[2]
protocol_path = pathlib.Path(sys.argv[3])
gate = json.loads(path.read_text())
if gate.get("gate") != "H3" or gate.get("passed") is not True:
    raise SystemExit("H3 gate file must contain gate='H3' and passed=true")
if gate.get("config_sha256") != expected_config_sha:
    raise SystemExit("H3 gate was produced from a different frozen config")
if gate.get("readout_protocol_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit("H3 gate was produced from a different readout protocol")
direction = pathlib.Path(gate.get("teacher_direction_path", ""))
if not direction.is_file():
    raise SystemExit("H3 teacher direction artifact is missing")
if gate.get("teacher_direction_sha256") != hashlib.sha256(direction.read_bytes()).hexdigest():
    raise SystemExit("H3 gate file must bind the frozen teacher direction SHA-256")
print(f"Accepted H3 gate: {path}")
PY

silent-transfer pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer train-students "$CONFIG" --repo-root "$REPO_ROOT" --resume
scripts/lambda/run_jlens_students.sh "$CONFIG" "$REPO_ROOT" "$GATE_FILE"

echo "Students trained and the frozen-lens H4 projection is complete."
