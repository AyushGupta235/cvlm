#!/usr/bin/env bash
# On the pod: watchdog, setup, CUDA smoke test on tiny models, then the real run.
# Usage: remote_run.sh <config> <run name>. Writes runs/<run>/DONE or runs/<run>/FAILED.
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"
RUN="$2"
mkdir -p "runs/$RUN"
PY=$(command -v python3 || command -v python)
export PYTHONUNBUFFERED=1

# Container variables (pod id, pod-scoped API key, time cap) are set on PID 1 but not always in an
# SSH session's environment. Read only the ones we need; never echo them.
while IFS= read -r -d '' kv; do
  case "$kv" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*|CVLM_MAX_MINUTES=*) export "$kv" ;; esac
done < /proc/1/environ 2>/dev/null || true

# Dead-man's switch: delete this pod after the time cap even if the laptop running launch.py is gone.
# Best effort: it needs the pod-scoped key to be allowed to delete its own pod.
MAX_MIN="${CVLM_MAX_MINUTES:-90}"
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  # The header goes to curl on stdin (printf is a builtin), so the key never appears in a process listing.
  nohup bash -c "sleep $((MAX_MIN * 60)); printf 'header = \"Authorization: Bearer %s\"\n' \"\$RUNPOD_API_KEY\" \
    | curl -s -K - -o /dev/null -w 'watchdog delete: %{http_code}\n' -X DELETE \
      https://rest.runpod.io/v1/pods/\$RUNPOD_POD_ID" >> "runs/$RUN/watchdog.log" 2>&1 &
  echo "[remote] watchdog armed: pod deletes itself after ${MAX_MIN} min"
else
  echo "[remote] watchdog NOT armed (no pod-scoped key visible); launch.py remains responsible for teardown"
fi

fail() { echo "[remote] FAILED: $1"; echo "$1" > "runs/$RUN/FAILED"; exit 1; }

echo "[remote] setup"
bash runpod/remote_setup.sh || fail "setup"

# CUDA smoke test: the tiny config in bfloat16 on the GPU, reusing the tiny data prepared on the laptop.
if [ -f runs/e1-tiny/data/manifest.json ]; then
  echo "[remote] smoke test (tiny models, cuda, bfloat16)"
  rm -rf runs/_smoke && mkdir -p runs/_smoke && cp -r runs/e1-tiny/data runs/_smoke/data
  "$PY" -m cvlm run --config configs/e1/tiny.yaml --run-dir runs/_smoke --device cuda --set dtype=bfloat16 \
    || fail "smoke test"
fi

echo "[remote] run $CONFIG -> runs/$RUN"
"$PY" -m cvlm run --config "$CONFIG" --run-dir "runs/$RUN" || fail "run"
date -u +%FT%TZ > "runs/$RUN/DONE"
echo "[remote] DONE"
