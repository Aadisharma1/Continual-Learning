#!/usr/bin/env python3
"""
SEAL Baseline Reproduction — Master Orchestrator
=================================================
arXiv 2506.10943, Section 4.2 — Knowledge Incorporation

Runs the three SEAL baselines sequentially:
  Stage 1:  Base model zero-shot closed-book eval  (target ~32.7% EM)
  Stage 2:  LoRA fine-tune on raw passage only      (target ~33.5% EM)
  Stage 3:  LoRA fine-tune on passage + synthetic QA (target ~39.7% EM)

Usage (from seal_project/):
    python src/run_seal_baselines.py
    python src/run_seal_baselines.py --dry-run          # quick 3-passage test
    python src/run_seal_baselines.py --max-passages 50  # half budget
"""

import getpass
import json
import os
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model


def ensure_hf_token():
    """Prompt for HuggingFace token at startup if not already set."""
    if os.environ.get("HF_TOKEN"):
        print("  HF_TOKEN: found in environment ✓")
        return
    print("\n  HuggingFace token required to download Qwen/Qwen2.5-7B.")
    print("  (Get yours at https://huggingface.co/settings/tokens)\n")
    token = getpass.getpass("  Enter HF_TOKEN: ").strip()
    if not token:
        print("[ERROR] No token provided. Exiting.")
        sys.exit(1)
    os.environ["HF_TOKEN"] = token
    print("  HF_TOKEN: set ✓\n")

from config import SEALConfig
from data_loader import (
    load_squad_passages,
    get_all_questions,
    PassageTrainDataset,
    CombinedTrainDataset,
)
from eval_closedbook import eval_closedbook_squad
from generate_synthetic import generate_synthetic_qa
from train_lora import create_lora_config, train_lora_on_dataset


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_base_model(cfg: SEALConfig):
    """
    Load Qwen2.5-7B base model in bfloat16 with device_map="auto"
    so it shards across all available GPUs.
    """
    print(f"\n  loading {cfg.model_id} in {cfg.torch_dtype}...")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[cfg.torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left-padding is required for correct batched generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )

    n_gpus = torch.cuda.device_count()
    print(f"  model loaded on {n_gpus} GPU(s)")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"    cuda:{i}  {name}  ({mem:.0f} GB)")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — Zero-shot baseline
# ═══════════════════════════════════════════════════════════════════

def run_stage1(model, tokenizer, all_questions, cfg: SEALConfig) -> dict:
    """Evaluate the frozen base model on the full SQuAD validation set."""
    print("\n" + "=" * 60)
    print("  STAGE 1: Base Model Zero-Shot Evaluation")
    print(f"  {len(all_questions)} questions (full validation set)")
    print("=" * 60)

    t0 = time.time()
    results = eval_closedbook_squad(
        model,
        tokenizer,
        all_questions,
        batch_size=cfg.eval_batch_size,
        max_new_tokens=cfg.eval_max_new_tokens,
        desc="stage1 zero-shot",
    )
    elapsed = time.time() - t0

    results["time_seconds"] = round(elapsed, 1)
    print(f"\n  Stage 1 EM: {results['em']:.4f}  "
          f"({results['correct']}/{results['total']})  "
          f"[{elapsed:.0f}s]")
    return results


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — Passage-only training
# ═══════════════════════════════════════════════════════════════════

def run_stage2(model, tokenizer, passages, cfg: SEALConfig) -> dict:
    """
    Per-passage inner loop: for each passage, train a fresh LoRA on
    just the passage text, then evaluate closed-book on that passage's
    questions.  EM is averaged across all passages.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: Train on Passage Only")
    print(f"  {len(passages)} passages × {cfg.epochs_per_passage} epochs each")
    print("=" * 60)

    t0 = time.time()
    total_correct = 0
    total_questions = 0
    per_passage = []

    for pi, pdata in enumerate(passages):
        ctx = pdata["context"]
        questions = pdata["questions"]

        # build tiny training dataset (1 sample = the passage)
        train_ds = PassageTrainDataset([ctx], tokenizer, cfg.max_seq_length)

        # train fresh LoRA
        stats = train_lora_on_dataset(model, tokenizer, train_ds, cfg)

        # eval closed-book on this passage's questions
        result = eval_closedbook_squad(
            model, tokenizer, questions,
            batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.eval_max_new_tokens,
        )

        total_correct += result["correct"]
        total_questions += result["total"]

        per_passage.append({
            "idx": pi,
            "n_questions": len(questions),
            "em": result["em"],
            "correct": result["correct"],
            "train_loss": round(stats["avg_loss"], 4),
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            running = total_correct / max(total_questions, 1)
            print(f"  [{pi+1:3d}/{len(passages)}]  running_EM={running:.4f}  "
                  f"this_EM={result['em']:.4f}  loss={stats['avg_loss']:.4f}")

        # free intermediate CUDA cache every 20 passages
        if (pi + 1) % 20 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    overall_em = total_correct / max(total_questions, 1)

    out = {
        "em": overall_em,
        "correct": total_correct,
        "total": total_questions,
        "num_passages": len(passages),
        "time_seconds": round(elapsed, 1),
        "per_passage": per_passage,
    }

    print(f"\n  Stage 2 EM: {overall_em:.4f}  "
          f"({total_correct}/{total_questions})  "
          f"[{elapsed:.0f}s]")
    return out


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — Passage + Base-model synthetic Q&A
# ═══════════════════════════════════════════════════════════════════

def run_stage3(model, tokenizer, passages, cfg: SEALConfig) -> dict:
    """
    1. Generate synthetic Q&A from the FROZEN base model (adapter off).
    2. For each passage, train a fresh LoRA on passage + synthetic Q&A.
    3. Evaluate closed-book.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: Train on Passage + Synthetic Q&A")
    print(f"  {len(passages)} passages × {cfg.num_synthetic_pairs} pairs each")
    print("=" * 60)

    t0 = time.time()

    # ── Step 1: generate ALL synthetic Q&A up front (batched) ─────
    # Disable LoRA adapters so the frozen base model produces the
    # synthetic data — this matches the paper's setup where the
    # "frozen copy" generates self-edits.
    print("\n  [gen] disabling LoRA adapters for generation...")
    model.disable_adapter_layers()

    t_gen = time.time()
    all_contexts = [p["context"] for p in passages]
    synthetic_map = generate_synthetic_qa(
        model,
        tokenizer,
        all_contexts,
        num_pairs=cfg.num_synthetic_pairs,
        batch_size=cfg.gen_batch_size,
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        top_p=cfg.gen_top_p,
    )
    gen_time = time.time() - t_gen

    model.enable_adapter_layers()
    print(f"  [gen] done in {gen_time:.0f}s, LoRA re-enabled")

    total_generated = sum(len(v) for v in synthetic_map.values())
    empty_count = sum(1 for v in synthetic_map.values() if len(v) == 0)
    print(f"  [gen] {total_generated} Q&A pairs total  "
          f"({empty_count} passages had 0 parseable pairs)")

    torch.cuda.empty_cache()

    # ── Step 2: per-passage train + eval ──────────────────────────
    print("\n  [train] starting per-passage inner loop...")
    total_correct = 0
    total_questions = 0
    per_passage = []

    for pi, pdata in enumerate(passages):
        ctx = pdata["context"]
        questions = pdata["questions"]
        qa_pairs = synthetic_map.get(ctx, [])

        # combined dataset: passage text + synthetic Q&A
        train_ds = CombinedTrainDataset(
            [ctx], qa_pairs, tokenizer, cfg.max_seq_length
        )

        # train fresh LoRA on combined data
        stats = train_lora_on_dataset(model, tokenizer, train_ds, cfg)

        # eval closed-book
        result = eval_closedbook_squad(
            model, tokenizer, questions,
            batch_size=cfg.eval_batch_size,
            max_new_tokens=cfg.eval_max_new_tokens,
        )

        total_correct += result["correct"]
        total_questions += result["total"]

        per_passage.append({
            "idx": pi,
            "n_questions": len(questions),
            "n_synthetic_qa": len(qa_pairs),
            "em": result["em"],
            "correct": result["correct"],
            "train_loss": round(stats["avg_loss"], 4),
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            running = total_correct / max(total_questions, 1)
            print(f"  [{pi+1:3d}/{len(passages)}]  running_EM={running:.4f}  "
                  f"this_EM={result['em']:.4f}  synth={len(qa_pairs)}  "
                  f"loss={stats['avg_loss']:.4f}")

        if (pi + 1) % 20 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    train_time = elapsed - gen_time
    overall_em = total_correct / max(total_questions, 1)

    out = {
        "em": overall_em,
        "correct": total_correct,
        "total": total_questions,
        "num_passages": len(passages),
        "total_synthetic_qa": total_generated,
        "generation_time_seconds": round(gen_time, 1),
        "training_time_seconds": round(train_time, 1),
        "time_seconds": round(elapsed, 1),
        "per_passage": per_passage,
    }

    print(f"\n  Stage 3 EM: {overall_em:.4f}  "
          f"({total_correct}/{total_questions})  "
          f"[{elapsed:.0f}s = {gen_time:.0f}s gen + {train_time:.0f}s train]")
    return out


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SEAL Baseline Reproduction (Stages 1-3)"
    )
    parser.add_argument(
        "--max-passages", type=int, default=100,
        help="Number of passages for Stages 2 & 3 (default: 100)",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=0,
        help="Cap Stage 1 eval samples (0 = full validation set)",
    )
    parser.add_argument(
        "--output", type=str, default="results/seal_baselines.json",
        help="Path for the results JSON",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick sanity check: 3 passages, 2 epochs, 2 synth pairs",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── API Key prompt ────────────────────────────────────────────
    ensure_hf_token()

    # ── Build config ──────────────────────────────────────────────
    cfg = SEALConfig(
        max_passages_stage2_3=3 if args.dry_run else args.max_passages,
        max_eval_samples_stage1=20 if args.dry_run else args.max_eval_samples,
        seed=args.seed,
    )
    if args.dry_run:
        cfg.epochs_per_passage = 2
        cfg.num_synthetic_pairs = 2
        cfg.eval_batch_size = 4
        cfg.gen_batch_size = 2

    # ── Banner ────────────────────────────────────────────────────
    print("=" * 60)
    print("  SEAL Paper Baseline Reproduction")
    print("  arXiv 2506.10943 — Section 4.2")
    print("=" * 60)
    print(f"  Model:           {cfg.model_id}")
    print(f"  LoRA:            r={cfg.lora_r}  α={cfg.lora_alpha}  "
          f"dropout={cfg.lora_dropout}")
    print(f"  LR:              {cfg.lr}")
    print(f"  Epochs/passage:  {cfg.epochs_per_passage}")
    print(f"  Passages (2/3):  {cfg.max_passages_stage2_3}")
    print(f"  Synthetic pairs: {cfg.num_synthetic_pairs}")
    print(f"  Eval batch:      {cfg.eval_batch_size}")
    print(f"  Gen batch:       {cfg.gen_batch_size}")
    print(f"  Seed:            {cfg.seed}")
    print(f"  Mode:            {'DRY RUN' if args.dry_run else 'FULL'}")
    print(f"  GPUs:            {torch.cuda.device_count()}")

    t_total = time.time()

    # ── Load model ────────────────────────────────────────────────
    base_model, tokenizer = load_base_model(cfg)

    # ── Load SQuAD data ───────────────────────────────────────────
    print("\n  loading SQuAD v2 validation data...")
    all_passages = load_squad_passages(
        split=cfg.squad_split, seed=cfg.seed
    )
    print(f"  unique passages with answerable Qs: {len(all_passages)}")

    all_questions = get_all_questions(all_passages)
    print(f"  total questions: {len(all_questions)}")

    if cfg.max_eval_samples_stage1 > 0:
        all_questions_stage1 = all_questions[:cfg.max_eval_samples_stage1]
        print(f"  Stage 1 eval capped at: {len(all_questions_stage1)}")
    else:
        all_questions_stage1 = all_questions

    subset_passages = all_passages[:cfg.max_passages_stage2_3]
    subset_qs = get_all_questions(subset_passages)
    print(f"  Stage 2/3: {len(subset_passages)} passages "
          f"({len(subset_qs)} questions)")

    results = {}

    # ── Stage 1: Zero-shot ────────────────────────────────────────
    results["stage1"] = run_stage1(
        base_model, tokenizer, all_questions_stage1, cfg
    )

    # ── Wrap with LoRA (stays in memory for Stages 2 & 3) ────────
    lora_config = create_lora_config(cfg)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # ── Stage 2: Passage only ─────────────────────────────────────
    results["stage2"] = run_stage2(model, tokenizer, subset_passages, cfg)

    # ── Stage 3: Passage + Synthetic Q&A ──────────────────────────
    results["stage3"] = run_stage3(model, tokenizer, subset_passages, cfg)

    # ── Save results ──────────────────────────────────────────────
    total_time = time.time() - t_total
    results["config"] = {
        "model_id": cfg.model_id,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lr": cfg.lr,
        "epochs_per_passage": cfg.epochs_per_passage,
        "max_passages": cfg.max_passages_stage2_3,
        "num_synthetic_pairs": cfg.num_synthetic_pairs,
        "seed": cfg.seed,
        "total_time_seconds": round(total_time, 1),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Stage':<35} {'EM':>8}  {'Target':>8}")
    print(f"  {'-' * 53}")
    print(f"  {'1: Zero-Shot (no adaptation)':<35} "
          f"{results['stage1']['em']:>8.4f}  {'~0.327':>8}")
    print(f"  {'2: Passage Only':<35} "
          f"{results['stage2']['em']:>8.4f}  {'~0.335':>8}")
    print(f"  {'3: Passage + Synthetic Q&A':<35} "
          f"{results['stage3']['em']:>8.4f}  {'~0.397':>8}")
    print(f"  {'-' * 53}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Results: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
