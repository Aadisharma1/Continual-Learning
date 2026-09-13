#!/bin/bash
# ─────────────────────────────────────────────────────────────
# SEAL Continual Learning — Full 5-Arm Pipeline
# Single RTX 6000 Blackwell (96GB VRAM)
# Model: Qwen/Qwen2.5-7B (BASE, not Instruct)
# ─────────────────────────────────────────────────────────────

set -e

echo "============================================"
echo "  SEAL Pipeline — Single GPU (RTX 6000 96GB)"
echo "  Model: Qwen/Qwen2.5-7B (Base)"
echo "============================================"
echo ""

# ─── API KEY SETUP ──────────────────────────────────────────
read -p "Enter your HuggingFace Token: " HF_KEY
echo ""
export HF_TOKEN="$HF_KEY"

if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN is required. Exiting."
    exit 1
fi

# ─── GPU CHECK ──────────────────────────────────────────────
python3 - <<'PYEOF'
import torch
n = torch.cuda.device_count()
print(f'  GPUs detected: {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'    cuda:{i}  {name}  ({mem:.0f} GB)')
if n == 0:
    print('[ERROR] No GPUs found!')
    exit(1)
PYEOF
echo ""

# ─── CONFIG ─────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B"
RESULTS_DIR="results"
CKPT_DIR="checkpoints"
DATA_DIR="data/self_edits"
SAFETY_DATA="$DATA_DIR/safety_anchor.jsonl"

SQUAD_SAMPLES=200
GSM8K_SAMPLES=200
ADVBENCH_SAMPLES=100
XSTEST_SAMPLES=200
FISHER_SAMPLES=100
TRAIN_EPOCHS=3
LORA_RANK=32
LORA_ALPHA=64
EWC_LAMBDA=0.5
BATCH_SIZE=4
LR=2e-5

mkdir -p $RESULTS_DIR $CKPT_DIR $DATA_DIR

# ─── STEP 0a: Build safety anchor dataset ──────────────────
echo "============================================"
echo "  Step 0a: Building safety anchor dataset"
echo "============================================"

python src/safety_anchor.py \
    --output $SAFETY_DATA \
    --n-refusals 50 \
    --n-helpful 50

# ─── STEP 0b: Generate self-edit data LOCALLY ──────────────
echo "============================================"
echo "  Step 0b: Generating self-edit data (LOCAL)"
echo "  Using frozen Qwen 7B Base — NO external API"
echo "============================================"

python src/local_generator.py \
    --output-dir $DATA_DIR \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --batch-size 8 \
    --model $MODEL \
    --dataset both

# ─── ARM 1: Baseline Frozen ────────────────────────────────
echo ""
echo "============================================"
echo "  Arm 1: baseline_frozen"
echo "============================================"

python src/eval.py \
    --base-model $MODEL \
    --arm-name "baseline_frozen" \
    --output $RESULTS_DIR/eval_results.json \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --advbench-samples $ADVBENCH_SAMPLES \
    --xstest-samples $XSTEST_SAMPLES

# ─── ARM 2: Single SFT on SQuAD ───────────────────────────
echo ""
echo "============================================"
echo "  Arm 2: baseline_single_sft (SQuAD)"
echo "============================================"

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --output-dir $CKPT_DIR/squad_naive \
    --epochs 1 \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda 0.0 \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --verify

python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/squad_naive \
    --arm-name "single_sft_squad" \
    --output $RESULTS_DIR/eval_results.json \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --advbench-samples $ADVBENCH_SAMPLES \
    --xstest-samples $XSTEST_SAMPLES

# ─── ARM 3: Naive Continual (SQuAD → GSM8K, no EWC) ──────
echo ""
echo "============================================"
echo "  Arm 3: naive_continual"
echo "============================================"

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --output-dir $CKPT_DIR/continual_step1 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda 0.0 \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/gsm8k_edits.jsonl \
    --adapter-path $CKPT_DIR/continual_step1 \
    --output-dir $CKPT_DIR/continual_step2 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda 0.0 \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA

python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/continual_step2 \
    --arm-name "naive_continual" \
    --output $RESULTS_DIR/eval_results.json \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --advbench-samples $ADVBENCH_SAMPLES \
    --xstest-samples $XSTEST_SAMPLES

# ─── ARM 4: EWC Protected ─────────────────────────────────
echo ""
echo "============================================"
echo "  Arm 4: ewc_protected"
echo "============================================"

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --output-dir $CKPT_DIR/ewc_step1 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda $EWC_LAMBDA \
    --fisher-samples $FISHER_SAMPLES \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --verify

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/gsm8k_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --adapter-path $CKPT_DIR/ewc_step1 \
    --output-dir $CKPT_DIR/ewc_step2 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda $EWC_LAMBDA \
    --fisher-samples $FISHER_SAMPLES \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --verify

python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/ewc_step2 \
    --arm-name "ewc_protected" \
    --output $RESULTS_DIR/eval_results.json \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --advbench-samples $ADVBENCH_SAMPLES \
    --xstest-samples $XSTEST_SAMPLES

# ─── ARM 5: EWC + Replay Buffer (Ours) ────────────────────
echo ""
echo "============================================"
echo "  Arm 5: ewc_plus_replay (Ours)"
echo "============================================"

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --replay-data $SAFETY_DATA \
    --output-dir $CKPT_DIR/replay_step1 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda 1.0 \
    --fisher-samples $FISHER_SAMPLES \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --verify

python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/gsm8k_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --replay-data $SAFETY_DATA \
    --adapter-path $CKPT_DIR/replay_step1 \
    --output-dir $CKPT_DIR/replay_step2 \
    --epochs $TRAIN_EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --ewc-lambda 1.0 \
    --fisher-samples $FISHER_SAMPLES \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --verify

python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/replay_step2 \
    --arm-name "ewc_plus_replay" \
    --output $RESULTS_DIR/eval_results.json \
    --squad-samples $SQUAD_SAMPLES \
    --gsm8k-samples $GSM8K_SAMPLES \
    --advbench-samples $ADVBENCH_SAMPLES \
    --xstest-samples $XSTEST_SAMPLES

# ─── SUMMARY ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ALL DONE. Results: $RESULTS_DIR/eval_results.json"
echo "============================================"

RESULTS_FILE="$RESULTS_DIR/eval_results.json"
python3 - "$RESULTS_FILE" <<'PYEOF'
import sys, json
with open(sys.argv[1]) as f:
    results = json.load(f)
print(f"{'Arm':<30} {'SQuAD EM':>10} {'GSM8K':>10} {'AdvBench':>10} {'XSTEST':>10}")
print('-' * 70)
for r in results:
    print(f"{r['arm']:<30} {r.get('squad_em',0):>10.4f} {r.get('gsm8k_pass1',0):>10.4f} {r.get('advbench_asr',0):>10.4f} {r.get('xstest_refusal_pct',0):>10.4f}")
PYEOF
