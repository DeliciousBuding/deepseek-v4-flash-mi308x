#!/usr/bin/env bash
# 01_download_model.sh — download/resume LLM weights from ModelScope (idempotent)
# Principle: official full checkpoints, no community quantization (QAT official
# quantized releases excepted) — the largest official release that fits 192 GB.
# Usage: bash 01_download_model.sh [qwen38|qwen36|dsflash]
#   qwen38  default, Qwen3.8-27B BF16 official (56 GB)
#   qwen36  Qwen3.6-35B-A3B BF16 (35B/3B MoE, agentic coding, ~70 GB)
#   dsflash DeepSeek-V4-Flash-0731 (~156 GB, official FP4/FP8 — the only
#          official release that fits a 192 GB GPU)
#
# Shard persistence: if the persistent volume is smaller than the full
# checkpoint, keep the first 30 shards and re-run this script after a restart
# — `modelscope download` skips existing files and only fetches the missing
# shards 31-48 (~58 GB).
set -euo pipefail

MODEL_KEY="${1:-qwen38}"

# Destination on the persistent volume; override with MODEL_DEST.
DEST="${MODEL_DEST:-/mnt/workspace/models}"

case "$MODEL_KEY" in
  qwen38-fp8)
    MODEL_ID="Qwen/Qwen3.8-27B-FP8"
    EXPECT_SHARDS="-"          # 无需校验 shard 数
    ;;
  qwen38)
    MODEL_ID="Qwen/Qwen3.8-27B"
    EXPECT_SHARDS="-"
    ;;
  qwen36)
    MODEL_ID="Qwen/Qwen3.6-35B-A3B"
    EXPECT_SHARDS="-"
    ;;
  dsflash)
    MODEL_ID="deepseek-ai/DeepSeek-V4-Flash-0731"
    EXPECT_SHARDS="48"         # 完整 48 个 safetensors shard
    ;;
  *)
    echo "未知模型: $MODEL_KEY (可选 qwen38-fp8|qwen38|dsflash)"; exit 1 ;;
esac

MODEL_DIR="$DEST/$MODEL_ID"
echo "下载/补全模型: $MODEL_ID  ->  $MODEL_DIR"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "modelscope CLI 不存在, 尝试 pip install modelscope ..."
  pip install -q modelscope
fi

# modelscope download 幂等: 已存在的文件跳过, 只补缺失(shard 续传的核心机制)
# --max-workers 并发下载: ModelScope OSS 单连接限速 ~5-15MB/s, 16 并发聚合 ~60-100MB/s
#   (实测: 单线程下 58G 约 1.5h; 16 并发约 10-15min)
MAX_WORKERS="${MAX_WORKERS:-16}"
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR" --max-workers "$MAX_WORKERS"

echo "完成。磁盘占用: $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"

if [ "$EXPECT_SHARDS" != "-" ]; then
  ACTUAL=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
  if [ "$ACTUAL" -eq "$EXPECT_SHARDS" ]; then
    echo "✅ shard 校验通过: $ACTUAL/$EXPECT_SHARDS, 可起服务"
  else
    echo "⚠️  shard 不完整: $ACTUAL/$EXPECT_SHARDS, 请重跑本脚本补下缺失分片后再起服务"
  fi
fi
