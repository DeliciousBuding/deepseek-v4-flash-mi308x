#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 with SGLang on ROCm (gfx942 / MI308X / MI300X).
#
# SGLang alternative to the vLLM launcher (02_serve_vllm.sh). Uses SGLang's
# RadixAttention and the AMD-specific deepseek_v4_backend_hip_radix backend.
# The SGLang wheel (0.5.17) contains:
#   - sglang/srt/configs/deepseek_v4.py
#   - sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py (AMD)
#   - sglang/srt/function_call/deepseekv4_detector.py (tool parser)
#   - sglang/kernels/ops/speculative/dspark/ (DSpark support)
#
# This script is secret-neutral. It does not generate/persist API keys.
set -euo pipefail

# Backward-compatible with the historical `dsflash` invocation.
if [ "$#" -gt 0 ] && [ "$1" != "dsflash" ] && [ "$1" != "deepseek-v4-flash" ]; then
  echo "ERROR: this launcher only serves DeepSeek-V4-Flash-0731; remove model selector '$1'." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_BASE_EXPLICIT="${MODEL_BASE+x}"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4-flash}"

# Shared SGLang venv (same as Qwen3.8 — both use the same SGLang wheel + deps)
VENV_DIR="${SGLANG_VENV:-/root/.venvs/sglang}"
if [ -z "${USE_SYSTEM_SGLANG:-}" ] && [ -x "$VENV_DIR/bin/python" ]; then
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  echo "[sglang] using shared sglang venv: $VENV_DIR"
else
  echo "[sglang] using system SGLang (unverified control path)"
fi

if ! command -v sglang >/dev/null 2>&1; then
  echo "ERROR: sglang not found" >&2
  echo "Run: bash /mnt/workspace/qwen3-8-27b-mi308x/scripts/install_sglang.sh" >&2
  exit 1
fi

# AITER integration
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------
count_model_shards() {
  local path="$1"
  find "$path" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l
}

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
  exit 1
fi

# ---------------------------------------------------------------------------
# Scheduler / memory profile (DS0731: MLA + MoE, no GDN)
# ---------------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"

# DSpark speculative decoding (DeepSeek's native draft model)
DSPARK_ENABLED="${DSPARK_ENABLED:-1}"
SPEC_ARGS=()
if [ "$DSPARK_ENABLED" = "1" ]; then
  # DSpark in SGLang uses --speculative-algorithm DSPARK with a draft checkpoint.
  # The draft model path defaults to a sibling directory; override with DSPARK_DRAFT_PATH.
  DSPARK_DRAFT_PATH="${DSPARK_DRAFT_PATH:-}"
  if [ -n "$DSPARK_DRAFT_PATH" ]; then
    SPEC_ARGS+=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path "$DSPARK_DRAFT_PATH"
    )
    echo "[dspark] enabled with draft model: $DSPARK_DRAFT_PATH"
  else
    echo "[dspark] DSPARK_DRAFT_PATH not set; DSpark disabled (native decode)"
  fi
else
  echo "[dspark] disabled (native decode baseline)"
fi

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
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
else
  echo "[auth] no API key configured"
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "[model] $MODEL_PATH -> $SERVED_MODEL_NAME (48 shards)"
echo "[scheduler] max_model_len=$MAX_MODEL_LEN mem_fraction=$MEM_FRACTION_STATIC chunked_prefill=$CHUNKED_PREFILL_SIZE"
echo "[runtime] host=$HOST port=$PORT"

# DS0731 uses MLA (not GDN), so no --mamba-* flags.
# --attention-backend aiter uses the AMD-specific deepseek_v4_backend_hip_radix path.
# --reasoning-parser and --tool-call-parser use the deepseek_v4 detector.
exec sglang serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend aiter \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --max-model-len "$MAX_MODEL_LEN" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --kv-cache-dtype fp8 \
  --enable-radix-cache \
  --enable-expert-parallel \
  "${SPEC_ARGS[@]}" \
  "${AUTH_ARGS[@]}"
