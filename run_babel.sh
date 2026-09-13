#!/bin/bash
#SBATCH --job-name=conti-safety-p1p2
#SBATCH --output=conti_%j.out
#SBATCH --error=conti_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL

# -----------------------------------------------------------
# CONTI SAFETY — Full Phase 1 + Phase 2 SLURM launcher
# CMU LTI Babel (L40 GPU)
#
# Usage (PI runs this ONE command):
#   git clone https://github.com/Aadisharma1/Conti.git && cd Conti && WANDB_API_KEY="key" sbatch run_babel.sh
# -----------------------------------------------------------

set -euo pipefail

echo "=============================================="
echo " CONTI SAFETY — FULL EXPERIMENT (P1 + P2)"
echo " Job ID:  $SLURM_JOB_ID"
echo " Node:    $(hostname)"
echo " GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo " Start:   $(date)"
echo "=============================================="

# -------------------------------------------------------
# 1. ENVIRONMENT NUKE & PAVE
# -------------------------------------------------------
echo "[STEP 1/5] Nuking old environment and creating fresh one..."

# Load Anaconda — try common CMU Babel module names
module purge 2>/dev/null || true
module load anaconda3 2>/dev/null \
    || module load anaconda 2>/dev/null \
    || module load conda 2>/dev/null \
    || { echo "[WARN] No anaconda module found, trying system paths..."; export PATH="/opt/anaconda3/bin:/opt/conda/bin:$PATH"; }

# Nuke old env (ignore errors if it doesn't exist)
conda deactivate 2>/dev/null || true
conda env remove -n conti_env -y 2>/dev/null || true
echo "[OK] Old environment removed."

# Create fresh Python 3.10 env
conda create -n conti_env python=3.10 -y --quiet
eval "$(conda shell.bash hook)"
conda activate conti_env

echo "[OK] Python: $(python --version)"

# -------------------------------------------------------
# 2. INSTALL DEPENDENCIES
# -------------------------------------------------------
echo "[STEP 2/5] Installing dependencies..."

# PyTorch with CUDA 12.1 (L40 / A100 compatible)
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project requirements (excludes flash-attn — needs special install)
pip install --quiet -r requirements.txt

# flash-attn: install after torch. NON-FATAL — fallback to eager attention if build fails
pip install flash-attn --no-build-isolation 2>/dev/null && echo "[OK] flash-attn installed." || echo "[WARN] flash-attn failed to build — will use eager attention (slower but safe)."

# Install project as editable package
pip install --quiet -e .

echo "[OK] torch: $(python -c 'import torch; print(torch.__version__)')"
echo "[OK] CUDA:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[OK] GPU:   $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# -------------------------------------------------------
# 3. W&B SETUP
# -------------------------------------------------------
echo "[STEP 3/5] Configuring W&B..."

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[WARN] No WANDB_API_KEY — W&B logging disabled. Results still print to stdout."
else
    export WANDB_API_KEY="${WANDB_API_KEY}"
    wandb login --relogin "${WANDB_API_KEY}" 2>/dev/null || true
    echo "[OK] W&B authenticated."
fi

export WANDB_PROJECT="conti-safety"
export PYTHONPATH="."
export HF_HOME="${SLURM_TMPDIR:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export CONTI_GSM8K_CACHE="${SLURM_TMPDIR:-$PWD/.cache}"
mkdir -p "${HF_HOME}" "${CONTI_GSM8K_CACHE}"

# Verify flash-attn and patch configs if not available
python - <<'PYEOF'
import subprocess, sys
try:
    import flash_attn
    print("[OK] flash-attn available — using flash_attention_2")
except ImportError:
    print("[WARN] flash-attn not available — patching configs to use eager attention")
    import re
    for cfg in ["configs/phase1_10rounds_babel.yaml", "configs/phase2_10rounds_babel.yaml"]:
        txt = open(cfg).read()
        txt = txt.replace("attn_implementation: flash_attention_2", "attn_implementation: null")
        open(cfg, "w").write(txt)
    print("[OK] Configs patched to attn_implementation: null")
PYEOF

# -------------------------------------------------------
# 4. PHASE 1 — Verifier Only (No Replay Buffer)
# -------------------------------------------------------
echo ""
echo "=============================================="
echo " PHASE 1 — Verifier Only (10 rounds, no replay)"
echo " Config: configs/phase1_10rounds_babel.yaml"
echo " Output: ./outputs/babel_phase1_10rounds"
echo " Start:  $(date)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/phase1_10rounds_babel.yaml \
    --output-dir ./outputs/babel_phase1_10rounds

echo ""
echo "[OK] Phase 1 complete at $(date)"

# Save Phase 1 to W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    python - <<'PYEOF'
import wandb, glob, os
try:
    run = wandb.init(project="conti-safety", name="babel_phase1_artifacts", reinit=True)
    for f in glob.glob("outputs/babel_phase1_10rounds/**/*", recursive=True):
        if os.path.isfile(f) and not any(f.endswith(x) for x in [".bin", ".safetensors"]):
            wandb.save(f, policy="now")
    wandb.finish()
    print("[W&B] Phase 1 artifacts uploaded.")
except Exception as e:
    print(f"[WARN] W&B Phase 1 upload failed: {e}")
PYEOF
fi

# -------------------------------------------------------
# 5. PHASE 2 — Verifier + Replay Buffer
# -------------------------------------------------------
echo ""
echo "=============================================="
echo " PHASE 2 — Verifier + Replay Buffer (10 rounds)"
echo " Config: configs/phase2_10rounds_babel.yaml"
echo " Output: ./outputs/babel_phase2_10rounds"
echo " Start:  $(date)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/phase2_10rounds_babel.yaml \
    --output-dir ./outputs/babel_phase2_10rounds

echo ""
echo "[OK] Phase 2 complete at $(date)"

# Save Phase 2 to W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    python - <<'PYEOF'
import wandb, glob, os
try:
    run = wandb.init(project="conti-safety", name="babel_phase2_artifacts", reinit=True)
    for f in glob.glob("outputs/babel_phase2_10rounds/**/*", recursive=True):
        if os.path.isfile(f) and not any(f.endswith(x) for x in [".bin", ".safetensors"]):
            wandb.save(f, policy="now")
    wandb.finish()
    print("[W&B] Phase 2 artifacts uploaded.")
except Exception as e:
    print(f"[WARN] W&B Phase 2 upload failed: {e}")
PYEOF
fi

# -------------------------------------------------------
# DONE
# -------------------------------------------------------
echo ""
echo "=============================================="
echo " ALL EXPERIMENTS COMPLETE"
echo " End: $(date)"
echo " Results in: outputs/babel_phase1_10rounds/"
echo "             outputs/babel_phase2_10rounds/"
echo " SLURM log:  conti_${SLURM_JOB_ID}.out"
echo "=============================================="
