#!/usr/bin/env bash
# Download/resume the official DeepSeek-V4-Flash-0731 checkpoint from ModelScope.
# This repository is intentionally DeepSeek-only; other model launchers belong
# in separate serving projects.
set -euo pipefail

MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MODEL_ROOT="${MODEL_ROOT:-${MODEL_DEST:-/mnt/workspace/models}}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/$MODEL_ID}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-48}"
MAX_WORKERS="${MAX_WORKERS:-16}"

mkdir -p "$MODEL_DIR"

echo "Model      : $MODEL_ID"
echo "Destination: $MODEL_DIR"
echo "Workers    : $MAX_WORKERS"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ERROR: modelscope CLI is required. Install it in the active environment first:" >&2
  echo "  python -m pip install modelscope" >&2
  exit 1
fi

# ModelScope download is resumable/idempotent: existing files are retained and
# missing shards are fetched on a subsequent run. Authentication, if required,
# must be injected by the deployment environment; this script persists no token.
modelscope download \
  --model "$MODEL_ID" \
  --local_dir "$MODEL_DIR" \
  --max-workers "$MAX_WORKERS"

actual=$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')
size=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}')
echo "Downloaded size: ${size:-unknown}"

if [ "$actual" -ne "$EXPECTED_SHARDS" ]; then
  echo "ERROR: checkpoint incomplete: $actual/$EXPECTED_SHARDS safetensors shards" >&2
  echo "Re-run this script to resume the missing files." >&2
  exit 1
fi

echo "Checkpoint verified: $actual/$EXPECTED_SHARDS shards"
