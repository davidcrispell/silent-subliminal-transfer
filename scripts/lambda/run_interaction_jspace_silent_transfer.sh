#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_interaction_jspace_silent_transfer.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?set SST_EXPECTED_GIT_COMMIT}"
EXPECTED_CONFIG_SHA="${SST_EXPECTED_CONFIG_SHA256:?set SST_EXPECTED_CONFIG_SHA256}"
EXPECTED_CONFIG_BYTE_SHA="${SST_EXPECTED_CONFIG_BYTE_SHA256:-}"

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
  echo "another interaction J-space pipeline owns the run lock" >&2
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
run_st export-readout "$CONFIG" --repo-root "$REPO_ROOT"

GATE="$RUN_ROOT/readout/gates/h3.h3_gate.json"
if [[ ! -f "$GATE" ]]; then
  scripts/lambda/run_jlens_teacher_gate.sh "$CONFIG" "$REPO_ROOT"
fi
"$PYTHON" - "$GATE" "$RUN_ROOT/readout/specs/readout_protocol.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

gate_path = Path(sys.argv[1])
protocol_path = Path(sys.argv[2])
gate = json.loads(gate_path.read_text())
protocol = json.loads(protocol_path.read_text())
if gate.get("gate") != "H3" or gate.get("passed") is not True:
    raise SystemExit(f"teacher manipulation gate did not pass: {gate_path}")
if gate.get("config_sha256") != protocol.get("config_sha256"):
    raise SystemExit("teacher gate/config identity mismatch")
if gate.get("readout_protocol_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit("teacher gate/readout protocol identity mismatch")
direction = Path(gate.get("teacher_direction_path", ""))
if not direction.is_file():
    raise SystemExit("frozen teacher direction is missing")
if gate.get("teacher_direction_sha256") != hashlib.sha256(direction.read_bytes()).hexdigest():
    raise SystemExit("frozen teacher direction hash mismatch")
PY

run_st prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"
run_st generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
run_st generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition control
run_st pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"
run_st train-students "$CONFIG" --repo-root "$REPO_ROOT" --resume

REPORT="$RUN_ROOT/readout/reports/teacherward_students.json"
if [[ ! -f "$REPORT" ]]; then
  scripts/lambda/run_jlens_students.sh "$CONFIG" "$REPO_ROOT" "$GATE"
fi

mkdir -p "$RUN_ROOT/analysis"
"$PYTHON" - "$CONFIG" "$RUN_ROOT" "$REPORT" "$EXPECTED_COMMIT" "$EXPECTED_CONFIG_SHA" <<'PY'
import json
from pathlib import Path
import sys
from silent_transfer.config import load_config
from silent_transfer.provenance import sha256_file, write_json_atomic

config = load_config(sys.argv[1])
root = Path(sys.argv[2])
report_path = Path(sys.argv[3])
report = json.loads(report_path.read_text())
expected_seeds = [int(seed) for seed in config["seeds"]["students"]]
observed_seeds = sorted(
    int(row["seed"]) for row in report.get("student_projections", [])
)
if observed_seeds != sorted(expected_seeds):
    raise SystemExit(
        f"student projection seed mismatch: {observed_seeds} != {expected_seeds}"
    )
gates = report.get("gates", {})
if gates.get("H3_teacher_state_reproducible") is not True:
    raise SystemExit("final report lost the preregistered H3 teacher-state gate")
if not isinstance(gates.get("H4_teacherward_student_projection"), bool):
    raise SystemExit("final report is missing the boolean H4 result")
for seed in expected_seeds:
    for condition in ("treatment", "control"):
        completion = (
            root / "models" / "students" / condition / f"seed-{seed}" /
            "training_complete.json"
        )
        if not completion.is_file():
            raise SystemExit(f"missing completed training cell: {completion}")

summary = {
    "schema_version": 1,
    "experiment_id": config["experiment"]["id"],
    "estimand": config["experiment"]["estimand"],
    "git_commit": sys.argv[4],
    "config_sha256": sys.argv[5],
    "paired_seeds": expected_seeds,
    "h3_teacher_state_reproducible": gates["H3_teacher_state_reproducible"],
    "h4_teacherward_student_projection": gates["H4_teacherward_student_projection"],
    "paired_seed_summary": report["paired_seed_summary"],
    "teacher_state_gate": report["teacher_state_gate"],
    "report_path": str(report_path),
    "report_sha256": sha256_file(report_path),
    "interpretation_scope": (
        "history-conditioned latent-state signal inherited through numeric-only "
        "outputs; not literal copying of a hidden-state vector"
    ),
}
summary_path = root / "analysis" / "interaction_jspace_transfer_summary.json"
write_json_atomic(summary_path, summary)
write_json_atomic(
    root / "orchestration" / "pipeline_complete.json",
    {
        "schema_version": 1,
        "stage": "interaction_jspace_silent_transfer_complete",
        "git_commit": sys.argv[4],
        "config_sha256": sys.argv[5],
        "summary_sha256": sha256_file(summary_path),
        "h4_teacherward_student_projection": gates[
            "H4_teacherward_student_projection"
        ],
    },
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "stage=complete summary=$RUN_ROOT/analysis/interaction_jspace_transfer_summary.json"
