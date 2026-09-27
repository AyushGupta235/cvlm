#!/usr/bin/env bash
# On the pod: install CVLM's pinned dependencies next to the image's CUDA torch, and fetch CLM.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=$(command -v python3 || command -v python)
CLM_COMMIT=bb42c6c5bf914fd449bed2f6ca65be80602cb1f7

"$PY" -m pip install -q --upgrade uv
# torch==2.13.0 is already satisfied by the image's 2.13.0+cu128 build (a local version label),
# so uv keeps the CUDA wheel and only adds what is missing.
"$PY" -m uv pip install --system --break-system-packages -q -e ".[dev]"
bash scripts/fetch_clm.sh "$CLM_COMMIT"
"$PY" -m uv pip install --system --break-system-packages -q --no-deps -e third_party/CLM

"$PY" - <<'EOF'
import torch, transformers
assert torch.cuda.is_available(), "CUDA is not available on this pod"
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB | transformers {transformers.__version__}")
EOF
