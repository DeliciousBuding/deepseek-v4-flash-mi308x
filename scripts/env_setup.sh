#!/usr/bin/env bash
# =============================================================================
# env_setup.sh — isolated environments (venv, reusing system torch)
#
# Why venv instead of conda:
#   The platform image ships a custom ROCm torch (`2.11.0+gitd0c8b1f`) that is
#   exactly the torch vLLM ROCm nightly (wheels.vllm.ai @ cb8104839) was built
#   against. That build does not exist on conda channels, and installing a
#   conda env would pull a mismatched torch (+3 GB). Instead we create a
#   `python -m venv --system-site-packages` venv that reuses the system torch
#   and isolates only the version-sensitive packages (vLLM / AITER / flydsl).
#
# Layout:
#   system Python          = untouched (platform torch)
#   $VENV_BASE/vllm        = LLM stack (nightly vLLM + AITER 0.1.19 + patches)
#   $VENV_BASE/sglang      = SGLang fallback (source-built sgl-kernel, optional)
#
# Usage: bash env_setup.sh
#
# NOTE: create the venv on local disk, not on network storage. NFS-class
#   storage writes small files ~660x slower, which turns dependency installs
#   into multi-hour jobs. The install script snapshots the vllm venv to a
#   tarball on persistent storage for restart recovery.
# =============================================================================
set -euo pipefail

VENV_BASE="${VENV_BASE:-/root/.venvs}"

echo "===== [1/3] create vllm venv ====="
if [ -x "$VENV_BASE/vllm/bin/python" ]; then
  echo "exists, skipping"
else
  python3 -m venv --system-site-packages "$VENV_BASE/vllm"
  echo "vllm venv created"
fi

echo "===== [2/3] create sglang venv (optional fallback) ====="
if [ -x "$VENV_BASE/sglang/bin/python" ]; then
  echo "exists, skipping"
else
  python3 -m venv --system-site-packages "$VENV_BASE/sglang"
  echo "sglang venv created"
fi

echo "===== [3/3] optional pip mirror (set PIP_INDEX_URL to skip) ====="
if [ -n "${PIP_INDEX_URL:-}" ]; then
  for env in vllm sglang; do
    "$VENV_BASE/$env/bin/pip" config set global.index-url "$PIP_INDEX_URL" >/dev/null 2>&1 || true
  done
fi

echo ""
echo "===== done ====="
for env in vllm sglang; do
  echo "  $VENV_BASE/$env -> $("$VENV_BASE/$env/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo n/a)"
done
echo ""
echo "activate:"
echo "  source $VENV_BASE/vllm/bin/activate    # LLM stack"
echo "  source $VENV_BASE/sglang/bin/activate  # SGLang fallback"
