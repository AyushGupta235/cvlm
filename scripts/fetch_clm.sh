#!/usr/bin/env bash
# Clone Contrastive-LM/CLM into third_party/CLM at a pinned commit (idempotent). Shared by local and pod setup.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="${1:?usage: fetch_clm.sh <commit>}"

if [ ! -d third_party/CLM/.git ]; then
  mkdir -p third_party
  git clone --quiet https://github.com/Contrastive-LM/CLM.git third_party/CLM
fi
if [ "$(git -C third_party/CLM rev-parse HEAD)" != "$COMMIT" ]; then
  git -C third_party/CLM fetch --quiet origin "$COMMIT" 2>/dev/null || git -C third_party/CLM fetch --quiet origin
  git -C third_party/CLM checkout --quiet "$COMMIT"
fi
echo "CLM at $(git -C third_party/CLM rev-parse --short HEAD)"
