#!/usr/bin/env bash
# Environment probe for the validated native ROCm serving stack.
# Reference control: ROCm 7.2.3 / vLLM dev306 / AITER 0.1.19 / gfx942 192 GB.
set -uo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/workspace/models}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/$MODEL_ID}"

echo "==================== 1. Hardware and system ===================="
echo "--- CPU cores ---"; nproc
echo "--- Memory ---"; free -h
echo "--- Disks ---"; df -h

echo
echo "==================== 2. AMD GPU and ROCm ===================="
echo "--- ROCm version (validated: 7.2.3) ---"
cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/version 2>/dev/null || echo "no /opt/rocm"
echo "--- GPU model (validated: MI308X/gfx942, 192 GB class) ---"
rocm-smi --showproductname 2>/dev/null || amd-smi static --asic 2>/dev/null || echo "no rocm-smi/amd-smi"
echo "--- VRAM ---"
rocm-smi --showmeminfo vram 2>/dev/null || echo "n/a"

echo
echo "==================== 3. Python / torch / HIP ===================="
python3 --version
python3 - <<'PY' 2>&1 | head -10
import torch
print("torch:", torch.__version__)
print("hip:", getattr(torch.version, "hip", "None"))
print("accelerator available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM: {gb:.1f} GiB")
PY

echo
echo "==================== 4. vLLM / AITER / flydsl ===================="
python3 -c "import vllm; print('vllm:', vllm.__version__)" 2>&1 | tail -1
python3 -c "import aiter; print('aiter:', getattr(aiter, '__version__', 'installed'))" 2>&1 | tail -1
python3 -c "import flydsl; print('flydsl:', getattr(flydsl, '__version__', 'installed'))" 2>&1 | tail -1
command -v vllm >/dev/null 2>&1 && echo "vllm CLI: $(command -v vllm)" || echo "vllm CLI not in PATH"
command -v modelscope >/dev/null 2>&1 && echo "modelscope CLI: $(command -v modelscope)" || echo "modelscope CLI missing"

echo
echo "==================== 5. Extension compatibility ===================="
python3 -c "import vllm; from vllm.model_executor.models import registry" 2>&1 \
  | grep -iE "skipping|incompatible" \
  && echo ">>> extension compatibility warning found; do not benchmark until audited" \
  || echo "no extension compatibility warning detected"

echo
echo "==================== 6. Model checkpoint ===================="
if [ -d "$MODEL_PATH" ]; then
  shards=$(find "$MODEL_PATH" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')
  echo "$MODEL_PATH exists: $(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}')"
  echo "safetensors shards: $shards/48"
else
  echo "$MODEL_PATH missing"
fi

echo
echo "==================== Conclusion ===================="
echo "This probe is informational. Run scripts/audit_runtime.py before accepting"
echo "performance numbers or changing the validated serving profile."
