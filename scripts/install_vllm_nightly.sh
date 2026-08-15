#!/usr/bin/env bash
# =============================================================================
# install_vllm_nightly.sh — install nightly vLLM + AITER 0.1.19 + the patch stack
#
# Ports the ryanzhou/deepseek-v4-flash-mi300x Docker production stack onto a
# Docker-less host (venv + local wheels + byte-for-byte patch overlay).
#
# Prerequisites (paths overridable via env):
#   1. wheels downloaded to $WHEELS (vLLM nightly, AITER, flydsl)
#   2. patch repo cloned to $REPO (deepseek-v4-flash-mi300x)
#   3. vllm venv created (scripts/env_setup.sh)
#
# Usage: bash install_vllm_nightly.sh
# Idempotent: safe to re-run (patches are overwritten; wheels installed with
# --ignore-installed and never touch system packages).
#
# Version pins:
#   vLLM  0.26.0+rocm723 (preinstalled) -> 0.26.1rc1.dev306+gcb8104839.rocm723 (nightly)
#   AITER 0.1.16 (preinstalled)         -> 0.1.19
#   torch 2.11.0+gitd0c8b1f             -> reused from system (identical to the
#                                          torch the nightly wheel was built with)
#
# NOTE: create the venv on local disk (network storage writes small files
#   ~660x slower). VENV defaults to /root/.venvs/vllm; snapshot it to a tarball
#   on persistent storage for restart recovery (see infra/bootstrap.sh).
# =============================================================================
set -euo pipefail

VENV="${VENV:-/root/.venvs/vllm}"
WHEELS="${WHEELS:-/mnt/workspace/wheels}"
REPO="${REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
# venv 的 site-packages(补丁目标根目录)。用 pythonX.Y 显式拼, 避免 sysconfig
# 在 --system-site-packages 下返回系统路径。
PYVER="$("$PY" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$VENV/lib/$PYVER/site-packages"

echo "=============================================="
echo "  vLLM nightly 安装 + ryanhou 补丁套用"
echo "=============================================="
echo "venv:   $VENV"
echo "site:   $SITE"
echo "wheels: $WHEELS"
echo "patch:  $REPO"

echo ""
echo "===== [1/8] 确认系统 torch 复用(须 2.11.0+gitd0c8b1f + ROCm) ====="
"$PY" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip); assert torch.version.hip, "非 ROCm torch"'

echo ""
echo "===== [2/8] 定位 wheel ====="
AITER_WHL="$(ls "$WHEELS"/amd_aiter-0.1.19-*.whl 2>/dev/null | head -1 || true)"
VLLM_WHL="$(ls "$WHEELS"/vllm-0.26.1rc1.dev306+*.whl 2>/dev/null | head -1 || true)"
FLYDSL_WHL="$(ls "$WHEELS"/flydsl-0.2.4-*.whl 2>/dev/null | head -1 || true)"
[ -n "$AITER_WHL" ] || { echo "❌ 缺 AITER wheel, 请先下载到 $WHEELS"; exit 1; }
[ -n "$VLLM_WHL" ]  || { echo "❌ 缺 vLLM wheel, 请先下载到 $WHEELS"; exit 1; }
[ -n "$FLYDSL_WHL" ] || { echo "❌ 缺 flydsl 0.2.4 wheel(AITER 0.1.19 运行时要求 flydsl>=0.2.4), 请先下载到 $WHEELS"; exit 1; }
echo "AITER: $AITER_WHL"
echo "vLLM:  $VLLM_WHL"
echo "flydsl: $FLYDSL_WHL"

# 完整性 sanity check(能当 zip 打开)
"$PY" -c "import zipfile,sys; [zipfile.ZipFile(f).testzip() and sys.exit(1) for f in sys.argv[1:]]" "$AITER_WHL" "$VLLM_WHL" "$FLYDSL_WHL"
echo "wheel 完整性 OK(zip 可读)"

echo ""
echo "===== [3/8] 安装 wheel(--ignore-installed 不碰系统包, --no-deps 复用 torch) ====="
# --no-compile 跳过 .pyc 字节码生成: venv 在 NFS 上, 给几千个 .py 生成 .pyc
# 的小文件写延迟极高(实测卡 30+ 分钟), 跳过它只损失首次 import 的内存编译时间。
export PYTHONDONTWRITEBYTECODE=1
"$PIP" install --no-deps --ignore-installed --no-compile "$AITER_WHL" "$VLLM_WHL"
# flydsl: AITER 0.1.19 运行时要求 flydsl>=0.2.4(长前缀触发 flydsl 内核路径, 系统 0.2.0 会让 Engine 死)。
# 系统 0.2.0 在 dist-packages, 这里正常装到 venv site-packages 覆盖之(不加 --ignore-installed, 让 pip 写到 venv)。
"$PIP" install --no-deps --no-compile "$FLYDSL_WHL"

echo ""
echo "===== [4/8] .so artifact 就位 + 校验 sha256 ====="
SO_ARTIFACT="$REPO/patches/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so"
EXPECT_SHA="a2912b897911c75d77611dcd42e4b0e0126bb8535f069045b32efc5f8f105610"
# 若已解压且 sha256 匹配, 直接跳过(容忍 AMD 实例无 zstd); 否则现场解压
if [ -f "$SO_ARTIFACT" ] && echo "$EXPECT_SHA  $SO_ARTIFACT" | sha256sum -c - >/dev/null 2>&1; then
  echo "  ✓ .so 已就位且 sha256 匹配, 跳过解压"
elif command -v zstd >/dev/null 2>&1; then
  zstd -d -f "$REPO/artifacts/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so.zst" -o "$SO_ARTIFACT"
  echo "$EXPECT_SHA  $SO_ARTIFACT" | sha256sum -c -
else
  echo "❌ .so 缺失且无 zstd 工具, 请先在 CPU 实例预解压"
  exit 1
fi

echo ""
echo "===== [5/8] 套用 ryanhou Python 补丁(byte-for-byte 覆盖) ====="
# 映射表: "源文件名|目标相对 site-packages 路径"
# 源查找顺序: 本仓库 patches/ 优先(自有环境补丁), 其次 ryanzhou 补丁仓库。
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
  mkdir -p "$SITE/$(dirname "$dst")"
  cp -f "$patch_src" "$SITE/$dst"
  echo "  ✓ $dst  ($patch_src)"
done

# 二进制补丁: topk-tiebreak 修复(崩溃根因)
cp -f "$REPO/patches/_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so" "$SITE/vllm/_C_stable_libtorch.abi3.so"
echo "  ✓ vllm/_C_stable_libtorch.abi3.so (topk 空集崩溃修复)"

echo ""
echo "===== [6/8] 预编译 hip-a8w4 内核 .so → aiter/jit/ ====="
mkdir -p "$SITE/aiter/jit"
cp -f "$REPO/kernel-dev/hip-a8w4/opus942/module_pa_sparse_prefill_opus942.so" "$SITE/aiter/jit/"
echo "  ✓ aiter/jit/module_pa_sparse_prefill_opus942.so"

echo ""
echo "===== [7/8] tuning CSV → AITER config ====="
mkdir -p "$SITE/aiter/configs/model_configs"
cp -f "$REPO/tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv" \
  "$SITE/aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv"
echo "  ✓ aiter/configs/model_configs/dsv4_a8w8_blockscale_tuned_gemm.csv"

echo ""
echo "===== [8/10] 内核源码 → /opt/cj-moe(JIT 编译用) ====="
mkdir -p /opt/cj-moe
cp -rf "$REPO/kernel-dev/hip-a8w4/." /opt/cj-moe/
echo "  ✓ /opt/cj-moe (swiglu_clamp/schedule/quanti8asym/fusedi8asym64/w2 + opus942)"

echo ""
echo "===== [9/10] mxfp4 activation 兼容修复 ====="
# ryanhou 的 mxfp4.fused-silu.py 来自旧生产版(124154a88), 删掉了函数的
# activation 参数; 但 cb8104839 的调用方 quantization/mxfp4.py 仍在传 activation=。
# 加回可选参数(忽略即可), 否则起服务报 "unexpected keyword argument 'activation'"。
MFXP4="$SITE/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"
if grep -q "def mxfp4_round_up_hidden_size_and_intermediate_size" "$MFXP4" && \
   ! grep -q "activation=None" "$MFXP4"; then
  sed -i 's|    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int|    backend: Mxfp4MoeBackend, hidden_size: int, intermediate_size: int,\n    activation=None,|' "$MFXP4"
  echo "  ✓ 已加回 activation 可选参数"
else
  echo "  ✓ 已兼容(跳过)"
fi

echo ""
echo "===== [10/10] 验证 + 打 tarball 备份到 NFS ====="
"$PY" -c 'import vllm; print("vllm", vllm.__version__)'
"$PY" -c 'import aiter; print("aiter import OK")'
"$PY" -c 'import flydsl; from aiter.ops.flydsl import is_flydsl_available; assert flydsl.__version__.startswith("0.2.4"), flydsl.__version__; assert is_flydsl_available(); print("flydsl", flydsl.__version__, "OK")'

echo "打 tarball 备份到 NFS(大文件写 NFS 快, 重启后 bootstrap.sh 恢复) ..."
tar -cf /mnt/workspace/.venvs/vllm.tar.gz -C "$(dirname "$VENV")" "$(basename "$VENV")"
echo "  ✓ /mnt/workspace/.venvs/vllm.tar.gz ($(du -sh /mnt/workspace/.venvs/vllm.tar.gz | cut -f1))"

echo ""
echo "=============================================="
echo "  ✅ installation complete"
echo "=============================================="
echo "start serving DS0731:"
echo "  source $VENV/bin/activate"
echo "  bash scripts/02_serve_vllm.sh dsflash"
