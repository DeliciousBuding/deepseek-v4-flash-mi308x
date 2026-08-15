#!/usr/bin/env bash
# 00_check_env.sh — environment probe: verify the host satisfies native vLLM on ROCm
# Reference baseline: ROCm 7.2.3 / vLLM 0.26.0+rocm723 / AITER 0.1.16 / MI308X 192GB
# Usage: bash 00_check_env.sh
set -uo pipefail

echo "==================== 1. Hardware and system ===================="
echo "--- CPU cores ---"; nproc
echo "--- Memory ---"; free -h
echo "--- Disks (model storage + root) ---"; df -h

echo ""
echo "==================== 2. AMD GPU and ROCm ===================="
echo "--- ROCm version (expect 7.2.x) ---"
cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/version 2>/dev/null || echo "no /opt/rocm"
echo "--- GPU model (expect MI300X/MI308X, gfx942) ---"
rocm-smi --showproductname 2>/dev/null || amd-smi static --asic 2>/dev/null || echo "no rocm-smi"
echo "--- VRAM (expect ~192GB) ---"; rocm-smi --showmeminfo vram 2>/dev/null || echo "n/a"

echo ""
echo "==================== 3. Python / torch / HIP ===================="
python3 --version
python3 - <<'PY' 2>&1 | head -8
import torch
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", "None"))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM: {gb:.1f} GB")
PY

echo ""
echo "==================== 4. vLLM and AITER (native, no Docker) ===================="
python3 -c "import vllm; print('vllm:', vllm.__version__)" 2>&1 | tail -1
python3 -c "import aiter; print('aiter:', getattr(aiter,'__version__','installed'))" 2>&1 | tail -1
command -v vllm >/dev/null 2>&1 && echo "vllm CLI: $(command -v vllm)" || echo "vllm CLI not in PATH"
command -v modelscope >/dev/null 2>&1 && echo "modelscope CLI: $(command -v modelscope)" || echo "modelscope CLI missing"

echo ""
echo "==================== 5. torch/vLLM compatibility warning check (critical) ===================="
# If this prints "Skipping import of cpp extensions ... incompatible torch version",
# vLLM's C++ kernels are ABI-mismatched with the platform's custom torch build
# and may affect DeepSeek V4 (FP8 FNUZ paths).
python3 -c "import vllm; from vllm.model_executor.models import registry" 2>&1 \
  | grep -iE "skipping|incompatible" && echo ">>> cpp-extension skip warning found; see README" || echo "no cpp-extension warning (OK)"

echo ""
echo "==================== 6. Model storage ===================="
for d in "${MODEL_BASE:-/mnt/workspace/models}" /mnt/workspace/models; do
  [ -d "$d" ] && echo "$d exists: $(du -sh "$d" 2>/dev/null | cut -f1)" || echo "$d missing"
done

echo ""
echo "==================== Conclusion ===================="
echo "Native vLLM (no Docker) + weights under MODEL_BASE."
echo "Watch item 5 (cpp-extension warning) and whether weights are complete."
