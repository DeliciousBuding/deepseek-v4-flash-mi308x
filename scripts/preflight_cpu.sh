#!/usr/bin/env bash
# CPU-instance preflight for the DeepSeek-V4-Flash ROCm recipe.
#
# Goal: finish every persistent-storage / source / artifact check that does not
# require an AMD GPU. After this passes, switching to the GPU instance should
# only require runtime restore/audit, model launch, and measured A/Bs.
#
# Safe on CPU-only DSW instances: this script never starts vLLM and does not
# import GPU kernels.
#
# Usage:
#   bash scripts/preflight_cpu.sh
#
# Optional overrides:
#   MODEL_BASE=/mnt/workspace/models
#   WHEELS=/mnt/workspace/wheels
#   PATCH_REPO=/mnt/workspace/deepseek-v4-flash-mi300x
#   PREPARE_PATCH_REPO=1   # default: pin/fetch historical source revision
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
MODEL_PATH="$MODEL_BASE/deepseek-ai/DeepSeek-V4-Flash-0731"
WHEELS="${WHEELS:-/mnt/workspace/wheels}"
PATCH_REPO="${PATCH_REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"
PREPARE_PATCH_REPO="${PREPARE_PATCH_REPO:-1}"
PERSIST="/mnt/workspace/.venvs"

failures=0
warnings=0

ok()   { printf 'OK   %s\n' "$*"; }
warn() { printf 'WARN %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL %s\n' "$*"; failures=$((failures + 1)); }

section() {
  echo
  echo "================================================================"
  echo "$*"
  echo "================================================================"
}

section "1. Repository integrity"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  HEAD="$(git -C "$ROOT" rev-parse HEAD)"
  ok "recipe git HEAD $HEAD"
  if [ -n "$(git -C "$ROOT" status --porcelain)" ]; then
    warn "recipe checkout has local changes/untracked files"
    git -C "$ROOT" status --short
  else
    ok "recipe checkout clean"
  fi
else
  fail "$ROOT is not a git checkout"
fi

section "2. Static syntax checks"
while IFS= read -r -d '' shfile; do
  if bash -n "$shfile"; then
    ok "bash -n ${shfile#$ROOT/}"
  else
    fail "shell syntax: ${shfile#$ROOT/}"
  fi
done < <(find "$ROOT/scripts" -type f -name '*.sh' -print0)

# Compile source text only; PYTHONDONTWRITEBYTECODE avoids persistent noise.
export PYTHONDONTWRITEBYTECODE=1
while IFS= read -r -d '' pyfile; do
  if python3 - "$pyfile" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
compile(p.read_text(encoding="utf-8"), str(p), "exec")
PY
  then
    ok "python syntax ${pyfile#$ROOT/}"
  else
    fail "python syntax: ${pyfile#$ROOT/}"
  fi
done < <(find "$ROOT/scripts" -type f -name '*.py' -print0)

section "3. Pin historical upstream patch source"
if [ "$PREPARE_PATCH_REPO" = "1" ]; then
  if PATCH_REPO="$PATCH_REPO" bash "$ROOT/scripts/prepare_patch_repo.sh"; then
    ok "pinned patch source prepared"
  else
    fail "prepare_patch_repo.sh failed"
  fi
else
  warn "PREPARE_PATCH_REPO=0; patch source was not repaired/pinned"
fi

if [ -d "$PATCH_REPO/.git" ]; then
  PATCH_HEAD="$(git -C "$PATCH_REPO" rev-parse HEAD 2>/dev/null || true)"
  echo "patch HEAD: ${PATCH_HEAD:-unknown}"
fi

section "4. Model weights"
if [ -d "$MODEL_PATH" ]; then
  shard_count="$(find "$MODEL_PATH" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
  if [ "$shard_count" -eq 48 ]; then
    ok "DeepSeek-V4-Flash shards complete: 48/48"
  else
    fail "DeepSeek-V4-Flash shards: ${shard_count}/48"
  fi

  for required in config.json tokenizer_config.json model.safetensors.index.json; do
    if [ -f "$MODEL_PATH/$required" ]; then
      ok "model metadata $required"
    else
      warn "model metadata missing: $required"
    fi
  done
  du -sh "$MODEL_PATH" 2>/dev/null || true
else
  fail "model directory missing: $MODEL_PATH"
fi

section "5. Exact wheel inventory"
mkdir -p "$WHEELS"
find_one() {
  local label="$1" pattern="$2"
  local f
  f="$(find "$WHEELS" -maxdepth 1 -type f -name "$pattern" -print -quit)"
  if [ -n "$f" ]; then
    ok "$label: $(basename "$f")"
    python3 - "$f" <<'PY' || exit 5
import sys, zipfile
p=sys.argv[1]
with zipfile.ZipFile(p) as z:
    bad=z.testzip()
    if bad:
        raise SystemExit(f"corrupt wheel member: {bad}")
print("     wheel zip integrity OK")
PY
  else
    fail "$label wheel missing in $WHEELS (pattern: $pattern)"
  fi
}
find_one "vLLM dev306 ROCm" 'vllm-0.26.1rc1.dev306+*.whl'
find_one "AITER 0.1.19" 'amd_aiter-0.1.19-*.whl'
find_one "flydsl 0.2.4" 'flydsl-0.2.4-*.whl'

section "6. Persistent restart artifacts"
if [ -f "$PERSIST/vllm.tar.gz" ]; then
  size="$(du -h "$PERSIST/vllm.tar.gz" | cut -f1)"
  ok "vLLM venv snapshot exists ($size)"
  # tar -tf reads the complete archive without extracting. Also ensure the
  # package locations required after restart are actually represented.
  if tar -tf "$PERSIST/vllm.tar.gz" >/tmp/vllm-tar-list.$$ 2>/tmp/vllm-tar-error.$$; then
    ok "vLLM venv tar archive readable"
    for needle in \
      'vllm/bin/vllm' \
      'site-packages/vllm/' \
      'site-packages/aiter/' \
      'site-packages/flydsl'; do
      if grep -q "$needle" /tmp/vllm-tar-list.$$; then
        ok "venv snapshot contains $needle"
      else
        fail "venv snapshot does not contain $needle"
      fi
    done
  else
    fail "vLLM venv tar archive is unreadable"
    cat /tmp/vllm-tar-error.$$ 2>/dev/null || true
  fi
  rm -f /tmp/vllm-tar-list.$$ /tmp/vllm-tar-error.$$
else
  fail "persistent vLLM venv snapshot missing: $PERSIST/vllm.tar.gz"
fi

for cache in aiter_cache.tar.gz torch_ext_cache.tar.gz; do
  if [ -f "$PERSIST/$cache" ]; then
    if tar -tf "$PERSIST/$cache" >/dev/null 2>&1; then
      ok "$cache exists and is readable"
    else
      fail "$cache exists but archive is unreadable"
    fi
  else
    warn "$cache missing (performance warm-start cost only; not model weights)"
  fi
done

section "7. Persistent storage headroom"
df -h /mnt/workspace 2>/dev/null || true
free -h 2>/dev/null || true

# The CPU instance may not expose /dev/shm exactly like the GPU instance, so
# report rather than gate it here. The GPU-side runtime audit/serve checks the
# 16 GB sandbox-specific constraint again.
if [ -d /dev/shm ]; then
  df -h /dev/shm || true
fi

section "8. GPU-switch readiness"
echo "failures: $failures"
echo "warnings : $warnings"
if [ "$failures" -ne 0 ]; then
  echo
  echo "CPU PREFLIGHT FAILED — fix the items above before paying for GPU time."
  exit 1
fi

echo
cat <<'EOF'
CPU PREFLIGHT PASSED.

After switching to the GPU instance:
  1. Human: run the private infra bootstrap once (establish SSH/tunnels + restore caches).
  2. Agent: cd /mnt/workspace/vllm-rocm-dsv4-flash && git pull --ff-only
  3. Agent: python3 scripts/audit_runtime.py
  4. Agent: start the default dsflash service, then run docs/GPU_VALIDATION_PLAN.md.

Do not reinstall or migrate to current upstream main unless audit/recovery proves
the persistent stable venv is missing or invalid.
EOF
