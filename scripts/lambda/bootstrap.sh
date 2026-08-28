#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
PY

python3 -m venv --system-site-packages .venv
. .venv/bin/activate

# Preserve Lambda's CUDA-enabled PyTorch instead of letting pip replace it with
# an incompatible wheel during dependency resolution.
python - <<'PY'
import torch

version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
if not (version >= (2, 7) and version < (3, 0)):
    raise SystemExit(
        f"Lambda image PyTorch {torch.__version__} is outside the supported range "
        "[2.7, 3.0); select a newer CUDA image"
    )
if torch.version.cuda is None:
    raise SystemExit("The Lambda image must provide a CUDA-enabled PyTorch build")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-build-isolation -e '.[dev]'

silent-transfer --help >/dev/null
silent-transfer validate configs/wolf_sl_9b.yaml --repo-root "$REPO_ROOT" >/dev/null

python - <<'PY'
import platform
import torch
import transformers
import peft

print({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
})
PY
