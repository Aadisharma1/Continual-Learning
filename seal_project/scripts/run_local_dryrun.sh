#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Local dry run on Qwen2.5-0.5B-Instruct (tiny data)
# Tests the entire pipeline incl. safety anchor, verifier, replay
# ─────────────────────────────────────────────────────────────

set -e

export GROQ_API_KEY="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"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_DIR="results/local_test"
CKPT_DIR="checkpoints/local_test"
DATA_DIR="data/local_test"
SAFETY_DATA="$DATA_DIR/safety_anchor.jsonl"

mkdir -p $RESULTS_DIR $CKPT_DIR $DATA_DIR

python='.\\.venv\\Scripts\\python.exe'

echo "=== Step 0a: Build safety anchor (5 refusals + 5 helpful) ==="
$python src/safety_anchor.py \
    --output $SAFETY_DATA \
    --n-refusals 5 \
    --n-helpful 5

echo "=== Step 0b: Generate self-edits (10 samples each) ==="
$python src/groq_generator.py \
    --output-dir $DATA_DIR \
    --squad-samples 10 \
    --gsm8k-samples 10 \
    --dataset both

echo "=== Step 1: Baseline eval ==="
$python src/eval.py \
    --base-model $MODEL \
    --arm-name "local_baseline" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 5 \
    --gsm8k-samples 5 \
    --advbench-samples 5 \
    --xstest-samples 5

echo "=== Step 2: Naive SFT (no EWC, no replay) ==="
$python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --output-dir $CKPT_DIR/naive \
    --epochs 1 \
    --batch-size 2 \
    --lr 2e-5 \
    --ewc-lambda 0.0 \
    --lora-rank 8 \
    --max-length 512

$python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/naive \
    --arm-name "local_naive_sft" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 5 \
    --gsm8k-samples 5 \
    --advbench-samples 5 \
    --xstest-samples 5

echo "=== Step 3: EWC-only (Fisher on safety data) ==="
$python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --output-dir $CKPT_DIR/ewc_only \
    --epochs 1 \
    --batch-size 2 \
    --lr 2e-5 \
    --ewc-lambda 0.5 \
    --fisher-samples 3 \
    --lora-rank 8 \
    --max-length 512 \
    --verify

$python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/ewc_only \
    --arm-name "local_ewc_only" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 5 \
    --gsm8k-samples 5 \
    --advbench-samples 5 \
    --xstest-samples 5

echo "=== Step 4: EWC + Replay (Fisher + safety mixed into training) ==="
$python src/train_ewc.py \
    --base-model $MODEL \
    --train-data $DATA_DIR/squad_edits.jsonl \
    --fisher-data $SAFETY_DATA \
    --replay-data $SAFETY_DATA \
    --output-dir $CKPT_DIR/ewc_replay \
    --epochs 1 \
    --batch-size 2 \
    --lr 2e-5 \
    --ewc-lambda 1.0 \
    --fisher-samples 3 \
    --lora-rank 8 \
    --max-length 512 \
    --verify

$python src/eval.py \
    --base-model $MODEL \
    --adapter-path $CKPT_DIR/ewc_replay \
    --arm-name "local_ewc_replay" \
    --output $RESULTS_DIR/eval.json \
    --squad-samples 5 \
    --gsm8k-samples 5 \
    --advbench-samples 5 \
    --xstest-samples 5

echo ""
echo "=== LOCAL DRY RUN COMPLETE ==="
echo "Results in: $RESULTS_DIR/eval.json"
cat $RESULTS_DIR/eval.json | $python -m json.tool
