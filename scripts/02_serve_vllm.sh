#!/usr/bin/env bash
# 02_serve_vllm.sh — Serve OpenAI-compatible /v1/chat/completions with native vLLM (no Docker)
# Usage: bash 02_serve_vllm.sh [qwen38|qwen36|dsflash]
#   qwen38  default, Qwen3.8-27B BF16 official (56 GB)
#   qwen36  Qwen3.6-35B-A3B BF16 (35B/3B MoE, agentic coding, day-0 support)
#   dsflash DeepSeek-V4-Flash-0731, primary model, tuned (prefix cache + DSpark)
#
# Uses the nightly vLLM venv (with the kernel patch stack) by default;
# set USE_SYSTEM_VLLM=1 to fall back to the system vLLM 0.26.0 (no patches).
set -euo pipefail

MODEL_KEY="${1:-qwen38}"
MODEL_BASE="${MODEL_BASE:-/mnt/workspace/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ---- restore /opt/cj-moe kernel sources (local disk, restored from patch repo) ----
PATCH_REPO_DIR="${PATCH_REPO:-/mnt/workspace/deepseek-v4-flash-mi300x}"
if [ ! -d /opt/cj-moe ] && [ -d "$PATCH_REPO_DIR/kernel-dev/hip-a8w4" ]; then
  mkdir -p /opt/cj-moe
  cp -r "$PATCH_REPO_DIR/kernel-dev/hip-a8w4/." /opt/cj-moe/
fi

# ---- vLLM stack selection: nightly venv (patched) by default ----
VENV_DIR="${VLLM_VENV:-/root/.venvs/vllm}"
if [ -z "${USE_SYSTEM_VLLM:-}" ] && [ -x "$VENV_DIR/bin/vllm" ]; then
  export PATH="$VENV_DIR/bin:$PATH"
  export VIRTUAL_ENV="$VENV_DIR"
  echo "[vllm] using patched nightly venv: $VENV_DIR"
else
  echo "[vllm] using system vLLM (no patches)"
fi

# Native vllm serve on ROCm; AITER kernels enabled via VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"

# API key (strong sk- key): env var > persisted key file > generate and persist
if [ -z "${VLLM_API_KEY:-}" ]; then
  KEY_FILE="${VLLM_API_KEY_FILE:-/mnt/workspace/.bootstrap/vllm_api_key}"
  if [ -f "$KEY_FILE" ]; then
    VLLM_API_KEY=$(cat "$KEY_FILE")
  else
    VLLM_API_KEY=$(python3 -c "import secrets; print('sk-' + secrets.token_urlsafe(32))")
    mkdir -p "$(dirname "$KEY_FILE")" && printf '%s' "$VLLM_API_KEY" > "$KEY_FILE"
    echo "generated API key and persisted it to $KEY_FILE"
  fi
fi

if [ "$MODEL_KEY" = "qwen38" ]; then
  MODEL_PATH="$MODEL_BASE/Qwen/Qwen3.8-27B"
  echo "启动 Qwen3.8-27B BF16 官方版 (单卡 192GB, 262K 上下文, fp8 KV, MTP 投机, 前缀缓存)"
  exec vllm serve "$MODEL_PATH" \
    --served-model-name qwen3.8-27b \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    --gpu-memory-utilization 0.92 \
    --api-key "$VLLM_API_KEY" \
    --host "$HOST" --port "$PORT"

elif [ "$MODEL_KEY" = "qwen36" ]; then
  MODEL_PATH="$MODEL_BASE/Qwen/Qwen3.6-35B-A3B"
  echo "启动 Qwen3.6-35B-A3B (35B/3B MoE, agentic coding 专精, AMD Day-0 无崩溃)"
  # 注意: GDN(Gated DeltaNet)层的 Mamba cache block 有限, max-num-seqs 设 512(默认 1024 会超)
  exec vllm serve "$MODEL_PATH" \
    --served-model-name qwen3.6-35b-a3b \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --max-num-seqs 512 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --gpu-memory-utilization 0.92 \
    --api-key "$VLLM_API_KEY" \
    --host "$HOST" --port "$PORT"

elif [ "$MODEL_KEY" = "dsflash" ]; then
  MODEL_PATH="$MODEL_BASE/deepseek-ai/DeepSeek-V4-Flash-0731"
  # 分片完整性预检: 持久化只留 30/48 shard, 缺的分片须先 bash 01_download_model.sh dsflash 补下
  SHARD_COUNT=$(ls "$MODEL_PATH"/model-*.safetensors 2>/dev/null | wc -l)
  if [ "$SHARD_COUNT" -lt 48 ]; then
    echo "❌ 权重不完整: $SHARD_COUNT/48 shard。请先补下缺失分片:"
    echo "   bash 01_download_model.sh dsflash   # 只补缺失的 shard 31-48(~58G), 跳过已有的 1-30"
    exit 1
  fi

  # 死掉的 EngineCore 无法 unlink 自己的 CPU-KV mmap; 起服务前清掉残留
  # (ryanzhou vllm-entrypoint.sh 同款处理, CPU offload 重启稳定性的前提)
  find /dev/shm -maxdepth 1 -type f -name 'vllm_offload_*.mmap' -delete 2>/dev/null || true

  echo "启动 DeepSeek-V4-Flash-0731 (patch stack: prefix cache + DSpark-7 + sparse MLA)"
  # ryanzhou 生产栈环境变量(对应 compose.yaml):
  #   OPUS_PREFILL=1 → 用预编译的稀疏 prefill 内核(module_pa_sparse_prefill_opus942.so)
  #   SKINNY_GEMM=0  → 关 skinny GEMM(该 shape 在 gfx942 有精度问题)
  #   AITER_CONFIG    → A8W8 blockscale GEMM 调优表
  export VLLM_ROCM_OPUS_PREFILL="${VLLM_ROCM_OPUS_PREFILL:-1}"
  export VLLM_ROCM_USE_SKINNY_GEMM="${VLLM_ROCM_USE_SKINNY_GEMM:-0}"
  export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE:-$PATCH_REPO_DIR/tuning/dsv4-mi300x-a8w8-blockscale-bpreshuffle-ck.batch4096.csv}"
  export AITER_CONFIG_GEMM_A8W8_BLOCKSCALE="${AITER_CONFIG_GEMM_A8W8_BLOCKSCALE:-$PATCH_REPO_DIR/tuning/dsv4-a8w8-blockscale-tuned-gemm.mi300x.decode-candidate.csv}"

  # Production defaults. Every performance-sensitive scheduling knob is an env
  # override so benchmarks can A/B without editing this file. Defaults preserve
  # the MI308X v2 baseline measured on this repository.
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
  MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
  LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-1024}"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
  MOE_BACKEND="${MOE_BACKEND:-triton}"

  # CPU KV offload: GPU pool is explicitly pinned; excess KV spills to the
  # native CPU tier. Offload without a pinned GPU pool previously shrank the
  # GPU pool to ~8.1 GB and could not admit a 512K request.
  # This sandbox exposes only 16 GB /dev/shm, so default CPU tier is 12 GB.
  # A/B: KV_OFFLOAD_GB=0 disables the CPU tier while leaving other defaults.
  KV_OFFLOAD_GB="${KV_OFFLOAD_GB:-12}"
  KV_CACHE_BYTES="${KV_CACHE_BYTES:-16000000000}"
  EXTRA_ARGS=()
  if [ "${KV_OFFLOAD_GB:-0}" -gt 0 ] 2>/dev/null; then
    EXTRA_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_BYTES")
    EXTRA_ARGS+=(--kv-offloading-size "$KV_OFFLOAD_GB")
    EXTRA_ARGS+=(--kv-offloading-backend native)
    echo "[kv-offload] CPU KV layer ${KV_OFFLOAD_GB}GB + GPU pool pinned $(($KV_CACHE_BYTES/1000000000))GB"
  else
    echo "[kv-offload] disabled (GPU-only)"
  fi

  # CUDA graph capture is intentionally OFF by default. It was A/B tested with
  # the upstream capture table and again with 3840/4096 added; both variants
  # reduced cold-prefill throughput on the pinned dev306 stack. Keep this gate
  # only for future runtime comparisons.
  if [ "${CUDAGRAPH:-0}" = "1" ]; then
    EXTRA_ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,1664,2048,3072,3712,3840,4096],"max_cudagraph_capture_size":4096}')
    echo "[cudagraph] FULL_AND_PIECEWISE (experimental; capture to M=4096)"
  fi

  # DSpark is a latency optimization for the low-concurrency coding-agent path.
  # DSPARK_ENABLED=0 provides a clean no-spec baseline without editing the script.
  DSPARK_ENABLED="${DSPARK_ENABLED:-1}"
  DSPARK_K="${DSPARK_K:-7}"
  SPEC_ARGS=()
  if [ "$DSPARK_ENABLED" = "1" ]; then
    SPEC_ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${DSPARK_K},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\"}")
    echo "[dspark] enabled K=${DSPARK_K} (probabilistic + block rejection)"
  else
    echo "[dspark] disabled (native decode baseline)"
  fi

  echo "[scheduler] max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} max_batched_tokens=${MAX_BATCHED_TOKENS} long_prefill_cap=${LONG_PREFILL_TOKEN_THRESHOLD}"
  echo "[runtime] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} moe_backend=${MOE_BACKEND}"

  exec vllm serve "$MODEL_PATH" \
    --served-model-name deepseek-v4-flash \
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
    --api-key "$VLLM_API_KEY" \
    --host "$HOST" --port "$PORT"

  # Validated production defaults (2026-08-16):
  #   - max-model-len=524288: 50K→500K ladder passes; short requests allocate
  #     only the blocks they actually use, while the upper bound remains 512K.
  #   - GPU KV 16 GB + native CPU KV 12 GB: stable on the 16 GB /dev/shm sandbox.
  #   - DSpark K=7, probabilistic drafting + block rejection: best tested C1 latency.
  #   - 4096 scheduler budget + 1024 long-prefill cap: current local Pareto point;
  #     use MAX_BATCHED_TOKENS to compare 2048/3072/4096/8192 on real agent traces.
  #   - CUDAGRAPH=0: both capture-table experiments were slower on dev306.
  #   - Dynamic speculative schedules are intentionally not enabled here until
  #     compatibility with this pinned DSpark runtime is validated on GPU.

else
  echo "未知模型: $MODEL_KEY (可选 qwen38|qwen36|dsflash)"; exit 1
fi
