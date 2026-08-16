#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 with native vLLM on ROCm.
#
# This public recipe is intentionally single-model and secret-neutral. It does
# not generate/persist API keys or own ingress/SSH/bootstrap configuration.
set -euo pipefail

# Backward-compatible with the historical `... dsflash` invocation while making
# the repository's single-model scope explicit.
if [ "$#" -gt 0 ] && [ "$1" != "dsflash" ]; then
  echo "ERROR: this launcher only serves DeepSeek-V4-Flash-0731; remove model selector '$1'." >&2
  exit 2
fi

MODEL_BASE_EXPLICIT="${MODEL_BASE+x}"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"

PATCH_REPO_DIR="${PATCH_REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"
if [ ! -d /opt/cj-moe ] && [ -d "$PATCH_REPO_DIR/kernel-dev/hip-a8w4" ]; then
  mkdir -p /opt/cj-moe
  cp -r "$PATCH_REPO_DIR/kernel-dev/hip-a8w4/." /opt/cj-moe/
fi

VENV_DIR="${VLLM_VENV:-/root/.venvs/vllm}"
if [ -z "${USE_SYSTEM_VLLM:-}" ] && [ -x "$VENV_DIR/bin/vllm" ]; then
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  echo "[vllm] using patched nightly venv: $VENV_DIR"
else
  echo "[vllm] using system vLLM (unpatched control path)"
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: vllm executable not found" >&2
  exit 1
fi

export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

count_model_shards() {
  local path="$1"
  if [ ! -d "$path" ]; then
    printf '0\n'
    return 0
  fi
  find "$path" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l
}

# Prefer a complete ephemeral local-disk hot copy when available. An explicit
# MODEL_BASE remains authoritative; otherwise fall back to persistent storage.
HOT_MODEL_BASE="${HOT_MODEL_BASE:-/root/models}"
if [ -z "$MODEL_BASE_EXPLICIT" ]; then
  hot_path="$HOT_MODEL_BASE/deepseek-ai/DeepSeek-V4-Flash-0731"
  hot_shards=$(count_model_shards "$hot_path")
  if [ "$hot_shards" -eq 48 ] && [ -f "$hot_path/model.safetensors.index.json" ]; then
    MODEL_BASE="$HOT_MODEL_BASE"
    echo "[model] using local hot copy: $hot_path"
  else
    echo "[model] local hot copy incomplete (${hot_shards}/48); using persistent base: $MODEL_BASE"
  fi
fi
MODEL_PATH="${MODEL_PATH:-$MODEL_BASE/deepseek-ai/DeepSeek-V4-Flash-0731}"

SHARD_COUNT=$(count_model_shards "$MODEL_PATH")
if [ "$SHARD_COUNT" -lt 48 ]; then
  echo "ERROR: checkpoint incomplete: $SHARD_COUNT/48 shards" >&2
  echo "Run: bash scripts/01_download_model.sh" >&2
  exit 1
fi

# A dead EngineCore cannot unlink its mmap-backed native CPU-KV files.
find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete 2>/dev/null || true

export VLLM_ROCM_OPUS_PREFILL="${VLLM_ROCM_OPUS_PREFILL:-1}"
export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"

# gfx942 is not one tuning domain: this MI308X reports 80 CUs while inherited
# MI300X production tables are keyed for 304 CUs. AITER includes cu_num in its
# lookup key, so select the repository's measured 80-CU tables when applicable.
if [ -z "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-}" ] || \
   [ -z "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-}" ]; then
  AITER_GFX="unknown"
  AITER_CU="unknown"
  aiter_hw=$(python3 - <<'PY' 2>/dev/null || true
from aiter.jit.utils.chip_info import get_gfx, get_cu_num
print(get_gfx(), get_cu_num())
PY
)
  if [ -n "$aiter_hw" ]; then
    read -r AITER_GFX AITER_CU <<<"$aiter_hw"
  fi

  MI308X_BP="$RECIPE_ROOT/tuning/dsv4-mi308x-80cu-a8w8-blockscale-bpreshuffle.csv"
  MI308X_STD="$RECIPE_ROOT/tuning/dsv4-mi308x-80cu-a8w8-blockscale.csv"
  if [ "${AITER_GFX:-}" = "gfx942" ] && [ "${AITER_CU:-}" = "80" ] && \
     [ -f "$MI308X_BP" ] && [ -f "$MI308X_STD" ]; then
    : "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:=$MI308X_BP}"
    : "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:=$MI308X_STD}"
    echo "[aiter] gfx942/80-CU detected; using MI308X measured tuning tables"
  else
    : "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:=$PATCH_REPO_DIR/tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.batch4096.csv}"
    : "${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:=$PATCH_REPO_DIR/tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv}"
    echo "[aiter] 80-CU table not selected (gfx=${AITER_GFX:-unknown} cu=${AITER_CU:-unknown}); using pinned upstream fallback"
  fi
fi
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE

# Validated MI308X control profile. Performance-sensitive scheduling knobs stay
# environment-driven so A/B experiments never require editing the launcher.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-3072}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MOE_BACKEND="${MOE_BACKEND:-triton}"
KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-12}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-16000000000}"

EXTRA_ARGS=()
if [ "${KV_OFFLOAD_GB:-0}" -gt 0 ] 2>/dev/null; then
  EXTRA_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_BYTES")
  EXTRA_ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GB")
  EXTRA_ARGS+=(--kv-offloading-backend native)
  echo "[kv-offload] CPU tier ${KV_OFFLOAD_GB} GB + GPU pool $((KV_CACHE_BYTES / 1000000000)) GB"
else
  echo "[kv-offload] disabled (GPU-only)"
fi

# CUDA graph capture remains opt-in. Both validated dev306 capture-table
# experiments regressed fresh prefill and increased HBM/startup cost.
if [ "${CUDAGRAPH:-0}" = "1" ]; then
  EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,1664,2048,3072,3712,3840,4096],"max_cudagraph_capture_size":4096}')
  echo "[cudagraph] experimental FULL_AND_PIECEWISE enabled"
fi

DSPARK_ENABLED="${DSPARK_ENABLED:-1}"
DSPARK_K="${DSPARK_K:-7}"
SPEC_ARGS=()
if [ "$DSPARK_ENABLED" = "1" ]; then
  SPEC_ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}")
  echo "[dspark] enabled K=$DSPARK_K (probabilistic + block rejection)"
else
  echo "[dspark] disabled (native decode baseline)"
fi

# Authentication is deployment policy, not repository state. If a key is
# injected, ask vLLM to enforce it. Otherwise the caller must keep this backend
# private or protect it at a gateway/ingress layer.
AUTH_ARGS=()
API_KEY_VALUE="${VLLM_API_KEY:-}"
if [ -z "$API_KEY_VALUE" ] && [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  if [ ! -r "$VLLM_API_KEY_FILE" ]; then
    echo "[auth] VLLM_API_KEY_FILE is not readable: $VLLM_API_KEY_FILE" >&2
    exit 1
  fi
  API_KEY_VALUE="$(<"$VLLM_API_KEY_FILE")"
fi

if [ -n "$API_KEY_VALUE" ]; then
  AUTH_ARGS+=(--api-key "$API_KEY_VALUE")
  echo "[auth] API key injected"
elif [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  echo "[auth] VLLM_API_KEY_FILE is empty: $VLLM_API_KEY_FILE" >&2
  exit 1
else
  echo "[auth] no API key configured"
fi

echo "[model] $MODEL_PATH -> $SERVED_MODEL_NAME"
echo "[scheduler] max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_batched_tokens=$MAX_BATCHED_TOKENS long_prefill_cap=$LONG_PREFILL_TOKEN_THRESHOLD"
echo "[runtime] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION moe_backend=$MOE_BACKEND host=$HOST port=$PORT"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --generation-config vllm \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
  --moe-backend "$MOE_BACKEND" \
  --linear-backend auto \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --enable-prompt-tokens-details \
  "${SPEC_ARGS[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${EXTRA_ARGS[@]}" \
  "${AUTH_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
