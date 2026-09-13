"""
Closed-book SQuAD evaluation — batched generation across GPUs.

The model NEVER sees the context paragraph at eval time.
It must answer strictly from memory (from weights updated during training).

Metric: Exact Match (EM) with standard SQuAD normalisation.
"""

import re
from typing import Dict, List

import torch
from tqdm import tqdm

from config import PROMPT_CLOSEDBOOK_EVAL


# ═══════════════════════════════════════════════════════════════════
# Answer normalisation (standard SQuAD)
# ═══════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles / punctuation / excess whitespace."""
    s = s.lower().strip()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, gold_answers: List[str]) -> bool:
    """True if normalised prediction exactly matches ANY gold answer."""
    pred_norm = normalize_answer(prediction)
    return any(normalize_answer(g) == pred_norm for g in gold_answers)


# ═══════════════════════════════════════════════════════════════════
# Batched generation
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 15,  # Reduced to 15 to prevent Base model babbling
) -> List[str]:
    """
    Generate short answers for a batch of prompts.
    Uses greedy decoding (do_sample=False) for deterministic eval.
    """
    device = model.get_input_embeddings().weight.device

    # Ensure padding side is explicitly left for batched generation
    tokenizer.padding_side = "left"

    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    # Fallback to eos_token_id if pad_token_id isn't set (common in Qwen base)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    outputs = model.generate(
        **encodings,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
    )

    # THE FIX: Slice based on the exact 2D shape of the padded input tensor, 
    # not the attention mask sum, which breaks on left-padded sequences.
    prompt_length = encodings["input_ids"].shape[1]
    
    responses = []
    for output_ids in outputs:
        # Extract ONLY the newly generated tokens
        new_tokens = output_ids[prompt_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Stop the base model from rambling (split at newline or period)
        text = text.split("\n")[0].split(".")[0].strip()
        responses.append(text)

    return responses


# ═══════════════════════════════════════════════════════════════════
# Main evaluation driver
# ═══════════════════════════════════════════════════════════════════

def eval_closedbook_squad(
    model,
    tokenizer,
    questions: List[Dict],
    batch_size: int = 32,
    max_new_tokens: int = 15, # Hardcoded short constraint
    desc: str = "eval",
) -> Dict:
    """
    Evaluate closed-book QA on a list of questions.

    Args:
        questions: List of dicts, each with:
                   "question" (str) and "answers" (List[str])
        batch_size: GPU batch size for generation

    Returns:
        {"em": float, "correct": int, "total": int}
    """
    model.eval()

    prompts = []
    all_golds = []
    for q in questions:
        prompts.append(PROMPT_CLOSEDBOOK_EVAL.format(question=q["question"]))
        all_golds.append(q["answers"])

    correct = 0
    total = len(prompts)

    for i in tqdm(
        range(0, total, batch_size),
        desc=desc,
        disable=(total <= batch_size),
    ):
        batch_prompts = prompts[i : i + batch_size]
        batch_golds = all_golds[i : i + batch_size]

        predictions = generate_batch(
            model, tokenizer, batch_prompts, max_new_tokens
        )

        for pred, golds in zip(predictions, batch_golds):
            if exact_match(pred, golds):
                correct += 1

    em = correct / max(total, 1)
    return {"em": em, "correct": correct, "total": total}