#!/usr/bin/env bash
# Small OpenAI-compatible concurrency sweep against the DeepSeek serving endpoint.
# Use scripts/bench/bench_full.py and docs/GPU_VALIDATION_PLAN.md for promotion.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
MAX_CONCURRENCY="${1:-8}"
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "model=$MODEL_NAME base_url=$BASE_URL concurrency=1..$MAX_CONCURRENCY"
echo "fixture: 8 requests per slot, 512 input / 256 output tokens"

for n in 1 2 4 8 16 32 64; do
  [ "$n" -gt "$MAX_CONCURRENCY" ] && break
  echo
  echo "================ concurrency=$n ================"
  vllm bench serve \
    --backend openai \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --num-prompts $((n * 8)) \
    --request-rate inf \
    --input-len 512 \
    --output-len 256 \
    || echo "benchmark at concurrency=$n failed; verify server health/auth and retry"
done

echo
echo "This is a smoke/sweep only. Promotion decisions must also pass the coding-agent,"
echo "long-context, prefix-cache, true-cold isolation and tool-call correctness gates."
