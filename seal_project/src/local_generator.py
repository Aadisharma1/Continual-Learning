"""
Local synthetic Q&A data generator — replaces groq_generator.py.

Uses the frozen Qwen/Qwen2.5-7B base model loaded on a single 96GB GPU
to generate self-edit trajectories (synthetic Q&A pairs) for SQuAD and
GSM8K datasets, matching the format expected by train_ewc.py.

Output: JSONL files with {"messages": [...]} format, same as groq_generator.py
produced, so train_ewc.py's SelfEditDataset can consume them unchanged.
"""

import argparse
import getpass
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════
# SQuAD loader — bypasses datasets library entirely
# ═══════════════════════════════════════════════════════════════════

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"


def load_squad_raw(max_samples: int = 500) -> List[Dict]:
    """
    Download SQuAD v1.1 validation JSON directly — no datasets library needed.
    Returns list of {context, question, answer} dicts.
    """
    cache_path = Path("/tmp/squad_dev.json")
    if not cache_path.exists():
        print(f"  downloading SQuAD from {SQUAD_URL}...")
        urllib.request.urlretrieve(SQUAD_URL, cache_path)
        print("  SQuAD downloaded OK")

    with open(cache_path) as f:
        data = json.load(f)

    items = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                items.append({
                    "context": ctx,
                    "question": qa["question"],
                    "answer": answers[0]["text"],
                    "all_answers": [a["text"] for a in answers],
                })
                if len(items) >= max_samples:
                    return items
    return items


# ═══════════════════════════════════════════════════════════════════
# GSM8K loader — also bypass datasets if broken
# ═══════════════════════════════════════════════════════════════════

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def load_gsm8k_raw(max_samples: int = 500) -> List[Dict]:
    """
    Download GSM8K test JSONL directly — no datasets library needed.
    Returns list of {question, answer} dicts.
    """
    cache_path = Path("/tmp/gsm8k_test.jsonl")
    if not cache_path.exists():
        print(f"  downloading GSM8K from {GSM8K_URL}...")
        urllib.request.urlretrieve(GSM8K_URL, cache_path)
        print("  GSM8K downloaded OK")

    items = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append({
                "question": row["question"],
                "answer": row["answer"],
            })
            if len(items) >= max_samples:
                break
    return items


# ═══════════════════════════════════════════════════════════════════
# Generation prompts
# ═══════════════════════════════════════════════════════════════════

SQUAD_GEN_PROMPT = (
    "Passage: {context}\n\n"
    "Based on the passage above, generate a question and its short factual answer.\n\n"
    "Question:"
)

GSM8K_GEN_PROMPT = (
    "Problem: {question}\n\n"
    "Solve this math problem step by step and give the final numerical answer.\n\n"
    "Solution:"
)


# ═══════════════════════════════════════════════════════════════════
# Model loading (single GPU, bfloat16)
# ═══════════════════════════════════════════════════════════════════

def load_generator_model(model_id: str = "Qwen/Qwen2.5-7B"):
    """Load the frozen base model for synthetic data generation."""
    print(f"  loading generator: {model_id} (bfloat16, single GPU)...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  model loaded on: {gpu_name}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# Batched generation
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate completions for a batch of prompts. OOM-safe."""
    device = model.get_input_embeddings().weight.device

    # OOM guard: if we get a CUDA OOM, halve the batch and retry
    try:
        encodings = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        outputs = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(prompts) == 1:
            print("  [warn] OOM even on single sample, skipping")
            return ["" for _ in prompts]
        mid = len(prompts) // 2
        print(f"  [warn] OOM on batch of {len(prompts)}, splitting to {mid}+{len(prompts)-mid}")
        left = generate_batch(model, tokenizer, prompts[:mid], max_new_tokens, temperature, top_p)
        right = generate_batch(model, tokenizer, prompts[mid:], max_new_tokens, temperature, top_p)
        return left + right

    results = []
    for i, out_ids in enumerate(outputs):
        input_len = int(encodings["attention_mask"][i].sum().item())
        new_tokens = out_ids[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)

    return results


# ═══════════════════════════════════════════════════════════════════
# SQuAD self-edit generation
# ═══════════════════════════════════════════════════════════════════

def generate_squad_edits(
    model,
    tokenizer,
    max_samples: int = 200,
    batch_size: int = 8,
) -> List[Dict]:
    """
    Generate self-edit trajectories from SQuAD passages.

    For each passage, the model generates a Q&A pair.
    Output format matches what groq_generator.py produced:
    {"messages": [{"role": "user", "content": "Question: ..."}, {"role": "assistant", "content": "answer"}]}
    """
    print(f"  generating self-edits for SQuAD (max {max_samples})...")
    rows = load_squad_raw(max_samples=max_samples)
    print(f"  loaded {len(rows)} SQuAD Q&A pairs")

    edits = []
    for i in tqdm(range(0, len(rows), batch_size), desc="squad gen"):
        batch = rows[i : i + batch_size]
        prompts = [SQUAD_GEN_PROMPT.format(context=c["context"][:1500]) for c in batch]

        completions = generate_batch(model, tokenizer, prompts, max_new_tokens=128)

        for ctx_data, completion in zip(batch, completions):
            gen_q = completion.split("\n")[0].strip()
            lines = completion.split("\n")
            gen_a = ""
            for line in lines[1:]:
                if line.strip().lower().startswith("answer"):
                    gen_a = re.sub(r"^[Aa]nswer\s*:\s*", "", line.strip())
                    break
            if not gen_a:
                # fallback to gold
                gen_q = ctx_data["question"]
                gen_a = ctx_data["answer"]

            edits.append({
                "messages": [
                    {"role": "user", "content": f"Answer the following question.\nQuestion: {gen_q}\nAnswer:"},
                    {"role": "assistant", "content": gen_a},
                ],
                "source": "local_qwen_squad",
            })

    print(f"  generated {len(edits)} SQuAD self-edits")
    return edits


# ═══════════════════════════════════════════════════════════════════
# GSM8K self-edit generation
# ═══════════════════════════════════════════════════════════════════

def generate_gsm8k_edits(
    model,
    tokenizer,
    max_samples: int = 200,
    batch_size: int = 8,
) -> List[Dict]:
    """Generate self-edit trajectories from GSM8K."""
    print(f"  generating self-edits for GSM8K (max {max_samples})...")
    try:
        items = load_gsm8k_raw(max_samples=max_samples)
    except Exception as e:
        print(f"  [warn] GSM8K raw download failed ({e}), trying datasets library")
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        items = [{"question": r["question"], "answer": r["answer"]} for r in ds][:max_samples]
    print(f"  loaded {len(items)} GSM8K problems")

    edits = []
    for i in tqdm(range(0, len(items), batch_size), desc="gsm8k gen"):
        batch = items[i : i + batch_size]
        prompts = [GSM8K_GEN_PROMPT.format(question=b["question"]) for b in batch]

        completions = generate_batch(model, tokenizer, prompts, max_new_tokens=512)

        for item, completion in zip(batch, completions):
            edit = {
                "messages": [
                    {"role": "user", "content": f"Solve the following math problem step by step.\nProblem: {item['question']}\nSolution:"},
                    {"role": "assistant", "content": completion if completion.strip() else item["answer"]},
                ],
                "source": "local_qwen_gsm8k",
            }
            edits.append(edit)

    print(f"  generated {len(edits)} GSM8K self-edits")
    return edits


# ═══════════════════════════════════════════════════════════════════
# File I/O
# ═══════════════════════════════════════════════════════════════════

def save_jsonl(data: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  saved {len(data)} items to {path}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data locally using Qwen 7B Base"
    )
    parser.add_argument("--output-dir", type=str, default="data/self_edits",
                        help="Directory for output JSONL files")
    parser.add_argument("--squad-samples", type=int, default=200)
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (tune for GPU memory)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Generator model ID")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["squad", "gsm8k", "both"],
                        help="Which dataset to generate for")
    args = parser.parse_args()

    # HF token prompt
    if not os.environ.get("HF_TOKEN"):
        token = getpass.getpass("Enter HF_TOKEN: ").strip()
        if token:
            os.environ["HF_TOKEN"] = token

    model, tokenizer = load_generator_model(args.model)

    if args.dataset in ("squad", "both"):
        squad_edits = generate_squad_edits(
            model, tokenizer,
            max_samples=args.squad_samples,
            batch_size=args.batch_size,
        )
        save_jsonl(squad_edits, f"{args.output_dir}/squad_edits.jsonl")

    if args.dataset in ("gsm8k", "both"):
        gsm8k_edits = generate_gsm8k_edits(
            model, tokenizer,
            max_samples=args.gsm8k_samples,
            batch_size=args.batch_size,
        )
        save_jsonl(gsm8k_edits, f"{args.output_dir}/gsm8k_edits.jsonl")

    print("\n  [DONE] All synthetic data generated locally.")


if __name__ == "__main__":
    main()
