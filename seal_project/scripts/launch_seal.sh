#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SEAL Baseline Reproduction — Launch Script
# For 2× RTX Pro GPUs with HuggingFace login
# ─────────────────────────────────────────────────────────────

set -e

echo "============================================"
echo "  SEAL Baseline Reproduction"
echo "  arXiv 2506.10943, Section 4.2"
echo "============================================"

# ── Check HuggingFace auth ──────────────────────────────────
if [ -z "$HF_TOKEN" ]; then
    echo ""
    read -p "Enter your HuggingFace Token: " HF_KEY
    export HF_TOKEN="$HF_KEY"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is required to download Qwen2.5-7B."
    exit 1
fi

# ── Check GPUs ──────────────────────────────────────────────
echo ""
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    cuda:{i}  {name}  ({mem:.0f} GB)')
if n == 0:
    print('[ERROR] No GPUs found!')
    exit(1)
"

# ── Create output dirs ──────────────────────────────────────
mkdir -p results checkpoints

# ── Run ─────────────────────────────────────────────────────
echo ""
echo "  Starting SEAL baselines..."
echo "  Output: results/seal_baselines.json"
echo ""

cd "$(dirname "$0")/.."

# Full run: 100 passages, ~1-2 hours total
python src/run_seal_baselines.py \
    --max-passages 100 \
    --output results/seal_baselines.json \
    --seed 42

echo ""
echo "============================================"
echo "  DONE. Results in: results/seal_baselines.json"
echo "============================================"
