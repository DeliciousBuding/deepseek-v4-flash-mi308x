#!/usr/bin/env bash
# Prepare the exact upstream patch source used by the validated MI308X profile.
#
# As of 2026-08-16 this revision is also ryanzhou upstream main. We still pin the
# full SHA so a future branch move cannot silently change a reproducible install.
# Note that our runtime base is dev306 while ryanzhou's production image is
# dev229, so identical overlay source does NOT mean byte-identical runtime.
set -euo pipefail

REPO_URL="${PATCH_REPO_URL:-https://github.com/ryanzhou/deepseek-v4-flash-mi300x.git}"
DEST="${PATCH_REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"
REV="${PATCH_REPO_REV:-012b9945c1e61ec7a7c7de12da58e8c7cafd92ab}"

if [ -e "$DEST" ] && [ ! -d "$DEST/.git" ]; then
  echo "ERROR: $DEST exists but is not a git checkout" >&2
  exit 1
fi

if [ ! -d "$DEST/.git" ]; then
  echo "[patch-repo] cloning $REPO_URL -> $DEST"
  git clone --no-checkout "$REPO_URL" "$DEST"
fi

if [ -n "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]; then
  echo "ERROR: $DEST has tracked local changes; refusing to switch revisions" >&2
  git -C "$DEST" status --short
  exit 1
fi

# Fetch the exact object explicitly so a fresh host does not depend on a branch
# name, tag, or the future state of upstream main.
echo "[patch-repo] fetching pinned revision $REV"
git -C "$DEST" fetch --no-tags origin "$REV"
git -C "$DEST" checkout --detach "$REV"

ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
if [ "$ACTUAL" != "$REV" ]; then
  echo "ERROR: expected $REV, got $ACTUAL" >&2
  exit 1
fi

required=(
  patches/gpt_oss_triton_kernels_moe.row-i8asym-candidate.py
  patches/mxfp4.fused-silu.py
  patches/activation.rocm-exact-swiglu.py
  patches/block_table.active-width-copy.py
  patches/deepseek_v4_amd_model.router-bf16.py
  patches/triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py
  patches/fused_compress_quant_cache.fnuz-shuffle.py
  patches/cache_utils.gather2048.py
  patches/aiter_pa_mqa_logits.i64.py
  patches/rocm_aiter_mla_sparse.decode-h32-k16.py
  patches/deepseek_v4_attention.wqb-bpreshuffle.py
  patches/deepseek_v4_rocm.wqb-bpreshuffle.py
  patches/rocm_aiter_mla.dspark-causal.py
  patches/dspark-speculator.independent-draft-gumbel.py
  patches/spec-decode-utils.independent-draft-gumbel.py
  patches/kv_offload_cpu_gpu_worker.load-war.py
  patches/scheduler.contention-aware.py
  kernel-dev/hip-a8w4/opus942/module_pa_sparse_prefill_opus942.so
  tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.batch4096.csv
  tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv
)

missing=0
for rel in "${required[@]}"; do
  if [ ! -e "$DEST/$rel" ]; then
    echo "MISSING: $rel" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 1

echo "[patch-repo] OK: $ACTUAL"
echo "[patch-repo] stable overlay source is ready at $DEST"
