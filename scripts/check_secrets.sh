#!/usr/bin/env bash
# Refuse to commit or push credentials. Wired up as .githooks/pre-commit (staged changes)
# and .githooks/pre-push (every tracked file). Run by hand: scripts/check_secrets.sh --all
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Token shapes: RunPod, Hugging Face, GitHub, Anthropic, OpenAI, AWS, and private-key headers.
PATTERN='(rpa_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
# Files that must never be tracked, whatever they contain.
BAD_NAMES='(^|/)(\.env(\..*)?|.*\.pem|id_[a-z0-9]+|runpod_ed25519|.*\.key)$'

if [ "${1:-}" = "--all" ]; then
  files=$(git ls-files)
  show() { cat -- "$1"; }
else
  files=$(git diff --cached --name-only --diff-filter=ACMR)
  show() { git show ":$1"; }          # the staged version, not the working tree
fi
# Report file:line only, never the matched value.
content_cmd() {
  printf '%s\n' "$files" | while read -r f; do
    [ -n "$f" ] || continue
    show "$f" 2>/dev/null | grep -nIE "$PATTERN" | cut -d: -f1 | sed "s|^|$f:|" || true
  done
}

fail=0
bad=$(printf '%s\n' "$files" | grep -E "$BAD_NAMES" | grep -v '\.env\.example$' || true)
if [ -n "$bad" ]; then
  echo "check_secrets: refusing credential-like files:" >&2
  printf '  %s\n' $bad >&2
  fail=1
fi
hits=$(content_cmd)
if [ -n "$hits" ]; then
  echo "check_secrets: possible credentials at (values hidden):" >&2
  printf '%s\n' "$hits" | sed 's/^/  /' >&2
  fail=1
fi
big=$(printf '%s\n' "$files" | while read -r f; do [ -f "$f" ] && [ "$(wc -c <"$f")" -gt 5000000 ] && echo "$f"; done || true)
if [ -n "$big" ]; then
  echo "check_secrets: files over 5 MB (embeddings and data belong in runs/, which is ignored):" >&2
  printf '  %s\n' $big >&2
  fail=1
fi
exit $fail
