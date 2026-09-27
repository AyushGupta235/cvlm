#!/usr/bin/env bash
# Local (Apple Silicon) environment: uv venv + pinned deps + CLM at the pinned commit.
set -euo pipefail
cd "$(dirname "$0")/.."

CLM_COMMIT=bb42c6c5bf914fd449bed2f6ca65be80602cb1f7

if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv/bin/python -e ".[dev]"

bash scripts/fetch_clm.sh "$CLM_COMMIT"
# --no-deps: the clm package lists vLLM as a hard dependency, which does not build on macOS.
# We only use clm.schema and clm.heads, which need numpy/torch/requests (already installed).
uv pip install --python .venv/bin/python --no-deps -e third_party/CLM

# Dedicated key for RunPod pods (passed to each pod as PUBLIC_KEY; nothing changes in your account).
if [ ! -f "$HOME/.ssh/runpod_ed25519" ]; then
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -t ed25519 -N "" -C "clm-vision-probe" -f "$HOME/.ssh/runpod_ed25519"
fi

.venv/bin/python -c "import torch, transformers, clm.schema; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| transformers', transformers.__version__)"
