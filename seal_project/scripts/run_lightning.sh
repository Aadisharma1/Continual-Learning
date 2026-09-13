#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SEAL Continual Learning — Full Pipeline for Lightning.ai A100
# Split by MODEL, not by dataset. Load once, run everything.
# ─────────────────────────────────────────────────────────────

set -e

echo "============================================"
echo "  SEAL Pipeline — API Key Setup"
echo "============================================"
echo ""
read -p "Enter your GROQ API Key: " GROQ_KEY
read -p "Enter your HuggingFace Token: " HF_KEY
echo ""

export GROQ_API_KEY="$GROQ_KEY"
export HF_TOKEN="$HF_KEY"

if [ -z "$GROQ_API_KEY" ] || [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] Both keys are required. Exiting."
    exit 1
fi

# ─── CONFIG ─────────────────────────────────────────────────
MODEL_1="Qwen/Qwen2.5-7B-Instruct"
MODEL_2="meta-llama/Meta-Llama-3.1-8B-Instruct"
RESULTS_DIR="results"
CKPT_DIR="checkpoints"
DATA_DIR="data/self_edits"
SAFETY_DATA="$DATA_DIR/safety_anchor.jsonl"

# ─── PRE-FLIGHT CHECKS ──────────────────────────────────────
echo "============================================"
echo "  Pre-Flight: Checking Llama 3.1 License Access"
echo "============================================"
python -c "
import sys
try:
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained('$MODEL_2')
    print('  [OK] Successfully authenticated and accessed $MODEL_2')
except Exception as e:
    print('\n[ERROR] Failed to access $MODEL_2!')
    print('This is a gated model. You must do two things before running this script:')
    print('  1. Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct and accept the usage license.')
    print('  2. Make sure your HF_TOKEN is valid and not expired.')
    print(f'  Error: {e}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

SQUAD_SAMPLES=200
GSM8K_SAMPLES=200
ADVBENCH_SAMPLES=100
XSTEST_SAMPLES=200
FISHER_SAMPLES=100
TRAIN_EPOCHS=3
LORA_RANK=16
EWC_LAMBDA=0.5
BATCH_SIZE=4
LR=2e-5

mkdir -p $RESULTS_DIR $CKPT_DIR $DATA_DIR

# ─── STEP 0: Generate data ─────────────────────────────────
echo "============================================"
echo "  Step 0a: Building safety anchor dataset"
echo "============================================"

python src/safety_anchor.py \
    --output $SAFETY_DATA \
    --n-refusals 50 \
    --n-helpful 50

echo "============================================"
echo "  Step 0b: Generating self-edit trajectories"
echo "============================================"

python src/groq_generator.py \
    --output-dir $DATA_DIR \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --dataset both

# ─── FUNCTION: Run full pipeline for one model ─────────────
run_model_pipeline() {
    local MODEL=$1
    local MODEL_SHORT=$2

    echo ""
    echo "============================================"
    echo "  Running pipeline for: $MODEL_SHORT"
    echo "============================================"

    # ── Arm 1: Baseline Frozen ──────────────────────────────
    echo "--- Arm 1: baseline_frozen ---"
    python src/eval.py \
        --base-model $MODEL \
        --arm-name "${MODEL_SHORT}_baseline_frozen" \
        --output $RESULTS_DIR/eval_results.json \
        --squad-samples $SQUAD_SAMPLES \
        --gsm8k-samples $GSM8K_SAMPLES \
        --advbench-samples $ADVBENCH_SAMPLES \
        --xstest-samples $XSTEST_SAMPLES

    # ── Arm 2: Single SFT on SQuAD (no EWC, no replay) ─────
    echo "--- Arm 2: baseline_single_sft (SQuAD) ---"
    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/squad_edits.jsonl \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_squad_naive \
        --epochs 1 \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda 0.0 \
        --lora-rank $LORA_RANK \
        --verify

    python src/eval.py \
        --base-model $MODEL \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_squad_naive \
        --arm-name "${MODEL_SHORT}_single_sft_squad" \
        --output $RESULTS_DIR/eval_results.json \
        --squad-samples $SQUAD_SAMPLES \
        --gsm8k-samples $GSM8K_SAMPLES \
        --advbench-samples $ADVBENCH_SAMPLES \
        --xstest-samples $XSTEST_SAMPLES

    # ── Arm 3: Naive Continual (SQuAD + GSM8K, no EWC) ─────
    echo "--- Arm 3: naive_continual ---"
    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/squad_edits.jsonl \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_continual_step1 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda 0.0 \
        --lora-rank $LORA_RANK

    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/gsm8k_edits.jsonl \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_continual_step1 \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_continual_step2 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda 0.0 \
        --lora-rank $LORA_RANK

    python src/eval.py \
        --base-model $MODEL \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_continual_step2 \
        --arm-name "${MODEL_SHORT}_naive_continual" \
        --output $RESULTS_DIR/eval_results.json \
        --squad-samples $SQUAD_SAMPLES \
        --gsm8k-samples $GSM8K_SAMPLES \
        --advbench-samples $ADVBENCH_SAMPLES \
        --xstest-samples $XSTEST_SAMPLES

    # ── Arm 4: EWC-only (Fisher on SAFETY data, not SQuAD) ─
    #    Fisher anchors the weights that matter for safety behavior
    #    No replay buffer — just the penalty term
    echo "--- Arm 4: ewc_protected ---"
    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/squad_edits.jsonl \
        --fisher-data $SAFETY_DATA \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_ewc_step1 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda $EWC_LAMBDA \
        --fisher-samples $FISHER_SAMPLES \
        --lora-rank $LORA_RANK \
        --verify

    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/gsm8k_edits.jsonl \
        --fisher-data $SAFETY_DATA \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_ewc_step1 \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_ewc_step2 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda $EWC_LAMBDA \
        --fisher-samples $FISHER_SAMPLES \
        --lora-rank $LORA_RANK \
        --verify

    python src/eval.py \
        --base-model $MODEL \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_ewc_step2 \
        --arm-name "${MODEL_SHORT}_ewc_protected" \
        --output $RESULTS_DIR/eval_results.json \
        --squad-samples $SQUAD_SAMPLES \
        --gsm8k-samples $GSM8K_SAMPLES \
        --advbench-samples $ADVBENCH_SAMPLES \
        --xstest-samples $XSTEST_SAMPLES

    # ── Arm 5: EWC + Replay Buffer (Ours) ───────────────────
    #    Fisher on safety data (same as Arm 4)
    #    PLUS: safety_anchor.jsonl mixed INTO the training batches
    #    via --replay-data so the model sees refusal examples every epoch
    echo "--- Arm 5: ewc_plus_replay (Ours) ---"
    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/squad_edits.jsonl \
        --fisher-data $SAFETY_DATA \
        --replay-data $SAFETY_DATA \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_replay_step1 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda 1.0 \
        --fisher-samples $FISHER_SAMPLES \
        --lora-rank $LORA_RANK \
        --verify

    python src/train_ewc.py \
        --base-model $MODEL \
        --train-data $DATA_DIR/gsm8k_edits.jsonl \
        --fisher-data $SAFETY_DATA \
        --replay-data $SAFETY_DATA \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_replay_step1 \
        --output-dir $CKPT_DIR/${MODEL_SHORT}_replay_step2 \
        --epochs $TRAIN_EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --ewc-lambda 1.0 \
        --fisher-samples $FISHER_SAMPLES \
        --lora-rank $LORA_RANK \
        --verify

    python src/eval.py \
        --base-model $MODEL \
        --adapter-path $CKPT_DIR/${MODEL_SHORT}_replay_step2 \
        --arm-name "${MODEL_SHORT}_ewc_plus_replay" \
        --output $RESULTS_DIR/eval_results.json \
        --squad-samples $SQUAD_SAMPLES \
        --gsm8k-samples $GSM8K_SAMPLES \
        --advbench-samples $ADVBENCH_SAMPLES \
        --xstest-samples $XSTEST_SAMPLES

    echo "  [DONE] $MODEL_SHORT pipeline complete"
}

# ─── EXECUTE ────────────────────────────────────────────────

# Phase 1: Qwen 7B (stay loaded the whole time)
run_model_pipeline "$MODEL_1" "qwen7b"

# Phase 2: Llama 8B
run_model_pipeline "$MODEL_2" "llama8b"

echo ""
echo "============================================"
echo "  ALL DONE. Results in: $RESULTS_DIR/"
echo "============================================"

python -c "
import json
with open('$RESULTS_DIR/eval_results.json') as f:
    results = json.load(f)
print(f\"{'Arm':<35} {'SQuAD EM':>10} {'GSM8K':>10} {'AdvBench':>10} {'XSTEST':>10}\")
print('-'*75)
for r in results:
    print(f\"{r['arm']:<35} {r['squad_em']:>10.4f} {r['gsm8k_pass1']:>10.4f} {r['advbench_asr']:>10.4f} {r['xstest_refusal_pct']:>10.4f}\")
"
