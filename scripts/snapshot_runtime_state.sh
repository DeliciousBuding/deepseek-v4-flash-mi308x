#!/usr/bin/env bash
# Persist the validated local vLLM venv plus runtime-generated compiler caches.
# Run after a healthy GPU warm-up whenever the production runtime/JIT registry changes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VLLM_VENV:-/root/.venvs/vllm}"
PERSIST="${PERSIST_DIR:-/mnt/workspace/.venvs}"
OUT="$PERSIST/vllm.tar.gz"
TMP="${OUT}.tmp.$$"

[ -x "$VENV/bin/vllm" ] || { echo "missing production venv: $VENV" >&2; exit 1; }
mkdir -p "$PERSIST"
trap 'rm -f "$TMP"' EXIT

echo "snapshotting production venv: $VENV -> $OUT"
tar -cf "$TMP" -C "$(dirname "$VENV")" "$(basename "$VENV")"
tar -tf "$TMP" >/dev/null
mv -f "$TMP" "$OUT"
trap - EXIT
echo "OK   venv snapshot: $OUT ($(du -h "$OUT" | cut -f1))"

bash "$ROOT/scripts/snapshot_runtime_caches.sh"

echo "runtime state snapshot complete"
