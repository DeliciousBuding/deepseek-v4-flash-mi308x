#!/usr/bin/env bash
# stage_model_local.sh — copy the persistent DeepSeek-V4-Flash model to local SSD.
# The local copy is ephemeral and only used when all 48 shards + index validate.
set -euo pipefail

MODEL_KEY="${1:-dsflash}"
SRC_BASE="${MODEL_BASE_PERSIST:-/mnt/workspace/models}"
DST_BASE="${HOT_MODEL_BASE:-/root/models}"

if [ "$MODEL_KEY" != "dsflash" ]; then
  echo "unsupported model key: $MODEL_KEY (only dsflash is staged today)" >&2
  exit 2
fi

rel="deepseek-ai/DeepSeek-V4-Flash-0731"
src="$SRC_BASE/$rel"
dst="$DST_BASE/$rel"
tmp="${dst}.tmp.$$"

count_shards() {
  if [ ! -d "$1" ]; then
    printf '0\n'
    return 0
  fi
  find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' 2>/dev/null | wc -l
}

src_shards=$(count_shards "$src")
if [ "$src_shards" -ne 48 ] || [ ! -f "$src/model.safetensors.index.json" ]; then
  echo "source model incomplete: $src_shards/48 shards at $src" >&2
  exit 1
fi

if [ "$(count_shards "$dst")" -eq 48 ] && [ -f "$dst/model.safetensors.index.json" ]; then
  echo "local hot copy already complete: $dst"
  exit 0
fi

mkdir -p "$(dirname "$dst")"
rm -rf "$tmp"
mkdir -p "$tmp"
echo "staging $src -> $dst"
cp -a "$src/." "$tmp/"

find "$src" -maxdepth 1 -type f -printf '%f %s\n' | sort > /tmp/dsflash-src-inventory.$$
find "$tmp" -maxdepth 1 -type f -printf '%f %s\n' | sort > /tmp/dsflash-dst-inventory.$$
trap 'rm -f /tmp/dsflash-src-inventory.$$ /tmp/dsflash-dst-inventory.$$' EXIT
diff -u /tmp/dsflash-src-inventory.$$ /tmp/dsflash-dst-inventory.$$
[ "$(count_shards "$tmp")" -eq 48 ]
for f in config.json tokenizer_config.json model.safetensors.index.json; do
  cmp -s "$src/$f" "$tmp/$f"
done

rm -rf "$dst"
mv "$tmp" "$dst"
echo "local hot copy ready: $dst ($(du -sh "$dst" | cut -f1))"