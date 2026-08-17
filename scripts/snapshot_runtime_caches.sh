#!/usr/bin/env bash
# Persist runtime-generated JIT/tuning caches after a successful GPU warm-up.
# Safe to run repeatedly. Missing caches are warnings, not correctness failures.
set -euo pipefail

PERSIST="${PERSIST_DIR:-/mnt/workspace/.venvs}"
mkdir -p "$PERSIST"

snapshot() {
  local src="$1" out="$2" label="$3" mode="$4"
  local tmp="${out}.tmp.$$"

  if [ ! -d "$src" ] || [ -z "$(find "$src" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "WARN $label cache not present yet: $src"
    return 0
  fi

  trap 'rm -f "$tmp"' RETURN
  if [ "$mode" = "contents" ]; then
    tar -cf "$tmp" -C "$src" .
  else
    tar -cf "$tmp" -C "$(dirname "$src")" "$(basename "$src")"
  fi
  tar -tf "$tmp" >/dev/null
  mv -f "$tmp" "$out"
  trap - RETURN
  echo "OK   $label cache snapshot: $out ($(du -h "$out" | cut -f1))"
}

# bootstrap.sh restores this archive into $HOME/.aiter, so archive the directory
# contents rather than an outer .aiter directory.
snapshot "$HOME/.aiter" "$PERSIST/aiter_cache.tar.gz" "AITER" contents

# restore_runtime.sh restores these archives into their parent directories.
# torch_extensions contains custom HIP/C++ modules; COMGR holds ROCm compiler
# code objects; .triton contains kernels that may otherwise JIT on the first
# real request even after vLLM's built-in warm-up has completed.
snapshot "$HOME/.cache/torch_extensions" "$PERSIST/torch_ext_cache.tar.gz" "torch_extensions" directory
snapshot "$HOME/.cache/comgr" "$PERSIST/comgr_cache.tar.gz" "ROCm COMGR" directory
snapshot "$HOME/.triton" "$PERSIST/triton_cache.tar.gz" "Triton" directory
