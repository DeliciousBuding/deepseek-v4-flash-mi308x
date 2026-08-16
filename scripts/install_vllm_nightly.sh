#!/usr/bin/env bash
# =============================================================================
# install_vllm_nightly.sh — install nightly vLLM + AITER 0.1.19 + patch stack
#
# Ports the ryanzhou/deepseek-v4-flash-mi300x overlay source pinned at
# 012b9945c1e61ec7a7c7de12da58e8c7cafd92ab onto this project's Docker-less
# dev306 runtime. As of 2026-08-16 that source SHA is also upstream main, but
# upstream production applies it to vLLM dev229. Same overlay source therefore
# does NOT imply a byte-identical runtime.
#
# Prerequisites (paths overridable via env):
#   1. wheels downloaded to $WHEELS (vLLM dev306, AITER, flydsl)
#   2. patch repo prepared by scripts/prepare_patch_repo.sh
#   3. vllm venv created by scripts/env_setup.sh
#
# Usage: bash install_vllm_nightly.sh
# Idempotent: patches are overwritten; wheels are installed into the venv and
# system Python / system torch are never replaced.
#
# Version pins:
#   vLLM  0.26.1rc1.dev306+gcb8104839.rocm723
#   AITER 0.1.19
#   flydsl 0.2.4
#   patch source 012b9945c1e61ec7a7c7de12da58e8c7cafd92ab
#   torch 2.11.0+gitd0c8b1f -> reused from system
#
# NOTE: keep the venv on local disk. Snapshot it to persistent storage after a
# validated install so a GPU restart does not require rebuilding the stack.
# =============================================================================
set -euo pipefail

VENV="${VENV:-/root/.venvs/vllm}"
WHEELS="${WHEELS:-/mnt/workspace/wheels}"
REPO="${PATCH_REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"
PATCH_REPO_REV="${PATCH_REPO_REV:-012b9945c1e61ec7a7c7de12da58e8c7cafd92ab}"

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
PYVER="$("$PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$VENV/lib/$PYVER/site-packages"

echo "=============================================="
echo "  vLLM dev306 + AITER 0.1.19 + pinned overlays"
echo "=============================================="
echo "venv:   $VENV"
echo "site:   $SITE"
echo "wheels: $WHEELS"
echo "patch:  $REPO"
echo "patch revision: $PATCH_REPO_REV"

echo ""
echo "===== [0/10] verify patch source revision ====="
if [ ! -d "$REPO/.git" ]; then
  echo "❌ patch repo missing or not a git checkout: $REPO"
  echo "   run first: bash scripts/prepare_patch_repo.sh"
  exit 1
fi
ACTUAL_PATCH_REV="$(git -C "$REPO" rev-parse HEAD)"
if [ "$ACTUAL_PATCH_REV" != "$PATCH_REPO_REV" ]; then
  echo "❌ patch repo revision drift"
  echo "   expected: $PATCH_REPO_REV"
  echo "   actual:   $ACTUAL_PATCH_REV"
  echo "   Refusing to mix an unpinned overlay source into the validated dev306 runtime."
  echo "   fix: bash scripts/prepare_patch_repo.sh"
  exit 1
fi
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "❌ patch repo has tracked local changes; refusing reproducible install"
  git -C "$REPO" status --short
  exit 1
fi
echo "  ✓ patch source pinned: $ACTUAL_PATCH_REV"

echo ""
echo "===== [1/10] verify system ROCm torch reuse ====="
"$PY" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip); assert torch.version.hip, "non-ROCm torch"'

echo ""
echo "===== [2/10] locate exact wheels ====="
AITER_WHL="$(ls "$WHEELS"/amd_aiter-0.1.19-*.whl 2>/dev/null | head -1 || true)"
VLLM_WHL="$(ls "$WHEELS"/vllm-0.26.1rc1.dev306+*.whl 2>/dev/null | head -1 || true)"
FLYDSL_WHL="$(ls "$WHEELS"/flydsl-0.2.4-*.whl 2>/dev/null | head -1 || true)"
[ -n "$AITER_WHL" ] || { echo "❌ missing AITER wheel in $WHEELS"; exit 1; }
[ -n "$VLLM_WHL" ]  || { echo "❌ missing vLLM dev306 wheel in $WHEELS"; exit 1; }
[ -n "$FLYDSL_WHL" ] || { echo "❌ missing flydsl 0.2.4 wheel in $WHEELS"; exit 1; }
echo "AITER:  $AITER_WHL"
echo "vLLM:   $VLLM_WHL"
echo "flydsl: $FLYDSL_WHL"

"$PY" -c "import zipfile,sys; [zipfile.ZipFile(f).testzip() and sys.exit(1) for f in sys.argv[1:]]" \
  "$AITER_WHL" "$VLLM_WHL" "$FLYDSL_WHL"
echo "wheel ZIP integrity OK"

echo ""
echo "===== [3/10] install wheels into isolated venv ====="
export PYTHONDONTWRITEBYTECODE=1
"$PIP" install --no-deps --ignore-installed --no-compile "$AITER_WHL" "$VLLM_WHL"
# AITER 0.1.19 validates flydsl>=0.2.4 only when the long-prefix FlyDSL path is
# imported. Leaving the platform's 0.2.0 visible makes short prompts work and
# then kills EngineCore on a long prefill, so the venv must explicitly override it.
"$PIP" install --no-deps --no-compile "$FLYDSL_WHL"

echo ""
echo "===== [4/10] top-k binary artifact + sha256 ====="
SO_ARTIFACT="$REPO/patches/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so"
EXPECT_SHA="a2912b897911c75d77611dcd42e4b0e0126bb8535f069045b32efc5f8f105610"
if [ -f "$SO_ARTIFACT" ] && echo "$EXPECT_SHA  $SO_ARTIFACT" | sha256sum -c - >/dev/null 2>&1; then
  echo "  ✓ patched .so already present and verified"
elif command -v zstd >/dev/null 2>&1; then
  zstd -d -f "$REPO/artifacts/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so.zst" -o "$SO_ARTIFACT"
  echo "$EXPECT_SHA  $SO_ARTIFACT" | sha256sum -c -
else
  echo "❌ patched .so missing and zstd unavailable"
  exit 1
fi

echo ""
echo "===== [5/10] apply Python overlays ====="
PATCHES=(
  "gpt_oss_triton_kernels_moe.row-i8asym-candidate.py|vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
  "mxfp4.fused-silu.py|vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
  "activation.rocm-exact-swiglu.py|vllm/model_executor/layers/activation.py"
  "block_table.active-width-copy.py|vllm/v1/worker/block_table.py"
  "deepseek_v4_amd_model.router-bf16.py|vllm/models/deepseek_v4/amd/model.py"
  "triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py|vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py"
  "fused_compress_quant_cache.fnuz-shuffle.py|vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
  "cache_utils.gather2048.py|vllm/models/deepseek_v4/common/ops/cache_utils.py"
  "aiter_pa_mqa_logits.i64.py|aiter/ops/triton/gluon/pa_mqa_logits.py"
  "rocm_aiter_mla_sparse.decode-h32-k16.py|vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
  "deepseek_v4_attention.wqb-bpreshuffle.py|vllm/models/deepseek_v4/attention.py"
  "deepseek_v4_rocm.wqb-bpreshuffle.py|vllm/models/deepseek_v4/amd/rocm.py"
  "rocm_aiter_mla.dspark-causal.py|vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
  "dspark-speculator.independent-draft-gumbel.py|vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
  "spec-decode-utils.independent-draft-gumbel.py|vllm/v1/worker/gpu/spec_decode/utils.py"
  "kv_offload_cpu_gpu_worker.load-war.py|vllm/v1/kv_offload/cpu/gpu_worker.py"
  "scheduler.contention-aware.py|vllm/v1/core/sched/scheduler.py"
  "shared_offload_region.madvise-tolerant.py|vllm/v1/kv_offload/cpu/shared_offload_region.py"
)
OWN_PATCHES="${OWN_PATCHES:-$(dirname "$(readlink -f "$0")")/../patches}"
for entry in "${PATCHES[@]}"; do
  src="${entry%%|*}"
  dst="${entry##*|}"
  if [ -f "$OWN_PATCHES/$src" ]; then
    patch_src="$OWN_PATCHES/$src"
  else
    patch_src="$REPO/patches/$src"
  fi
  [ -f "$patch_src" ] || { echo "❌ missing overlay source: $patch_src"; exit 1; }
  mkdir -p "$SITE/$(dirname "$dst")"
  cp -f "$patch_src" "$SITE/$dst"
  echo "  ✓ $dst  ($patch_src)"
done

cp -f "$SO_ARTIFACT" "$SITE/vllm/_C_stable_libtorch.abi3.so"
echo "  ✓ vllm/_C_stable_libtorch.abi3.so"

echo ""
echo "===== [6/10] sparse-prefill module -> aiter/jit ====="
mkdir -p "$SITE/aiter/jit"
cp -f "$REPO/kernel-dev/hip-a8w4/opus942/module_pa_sparse_prefill_opus942.so" "$SITE/aiter/jit/"
echo "  ✓ module_pa_sparse_prefill_opus942.so"

echo ""
echo "===== [7/10] AITER tuning table ====="
mkdir -p "$SITE/aiter/configs/model_configs"
cp -f "$REPO/tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv" \
  "$SITE/aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv"
echo "  ✓ dsv4_a8w8_blockscale_tuned_gemm.csv"

echo ""
echo "===== [8/10] JIT kernel source -> /opt/cj-moe ====="
mkdir -p /opt/cj-moe
cp -rf "$REPO/kernel-dev/hip-a8w4/." /opt/cj-moe/
echo "  ✓ /opt/cj-moe"

echo ""
echo "===== [9/10] dev306 mxfp4 signature compatibility ====="
# The upstream overlay omits an activation argument that dev306's caller still
# passes. Add an ignored optional parameter; this is a runtime-port compatibility
# edit, not a model-math change.
MFXP4="$SITE/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
if grep -q "def mxfp4_round_up_hidden_size_and_intermediate_size" "$MFXP4" && \
   ! grep -q "activation=None" "$MFXP4"; then
  sed -i 's|    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int|    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int,\n    activation=None,|' "$MFXP4"
  echo "  ✓ activation=None compatibility parameter added"
else
  echo "  ✓ compatibility already present"
fi

echo ""
echo "===== [10/10] validate + persist restart snapshots ====="
"$PY" -c 'import vllm; print("vllm", vllm.__version__)'
"$PY" -c 'import importlib.metadata as m; import aiter; print("AITER", m.version("amd-aiter"))'
"$PY" -c 'import flydsl; from aiter.ops.flydsl import is_flydsl_available; assert flydsl.__version__.startswith("0.2.4"), flydsl.__version__; assert is_flydsl_available(); print("flydsl", flydsl.__version__, "OK")'

echo "snapshotting install-time runtime state ..."
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/snapshot_runtime_state.sh"

echo ""
echo "=============================================="
echo "  ✅ installation complete"
echo "=============================================="
echo "verify runtime:"
echo "  python3 scripts/audit_runtime.py"
echo "start serving DS0731:"
echo "  source $VENV/bin/activate"
echo "  bash scripts/02_serve_vllm.sh dsflash"
