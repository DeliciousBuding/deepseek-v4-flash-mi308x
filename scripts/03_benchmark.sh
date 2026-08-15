#!/usr/bin/env bash
# 03_benchmark.sh — concurrency sweep (native vllm bench): find the sweet spot
# Prerequisite: server is up (bash 02_serve_vllm.sh <model>), listening on :8000
# Usage: bash 03_benchmark.sh [qwen3.8-27b|deepseek-v4-flash] [max_concurrency]
set -euo pipefail

MODEL_NAME="${1:-qwen3.8-27b}"
MAX_CONCURRENCY="${2:-8}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "concurrency sweep: 1..$MAX_CONCURRENCY, 32 prompts per level, in 512 / out 256 tokens"
echo "metrics: aggregate tok/s, per-stream tok/s, TTFT, TPOT"

for n in 1 2 4 8 16 32 64; do
  [ "$n" -gt "$MAX_CONCURRENCY" ] && break
  echo ""
  echo "==================== concurrency = $n ===================="
  vllm bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --num-prompts $((n * 8)) \
    --request-rate inf \
    --input-len 512 \
    --output-len 256 2>/dev/null \
    || echo "(concurrency $n failed/timed out — check the server and model name)"
done

echo ""
echo "Reading: aggregate tok/s rises with concurrency until the plateau; per-stream"
echo "tok/s below ~20 means you are near the compute bound. TTFT/TPOT spikes mean"
echo "KV cache or scheduler pressure. Pick the sweet-spot concurrency from this."
