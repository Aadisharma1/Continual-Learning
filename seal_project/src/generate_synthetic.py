

import re
from typing import Dict, List

import torch
from tqdm import tqdm

from config import PROMPT_GENERATE_QA


# ═══════════════════════════════════════════════════════════════════
# Q&A pair parsing
# ═══════════════════════════════════════════════════════════════════

def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """
    Parse model output into structured (question, answer) pairs.

    Handles formats like:
        1. Question: What is X?
        Answer: X is Y.
        2. Question: How does Z work?
        Answer: Z works by...

    Also handles:
        Q: What is X?
        A: X is Y.
    """
    pairs = []

    # ── Pattern 1: "N. Question: ... Answer: ..." ──────────────────
    pattern_numbered = (
        r"(?:\d+[\.\)]\s*)?(?:Question|Q)\s*:\s*(.*?)\s*"
        r"(?:\n\s*)?(?:Answer|A)\s*:\s*(.*?)"
        r"(?=\n\s*\d+[\.\)]\s*(?:Question|Q)\s*:|\Z)"
    )
    matches = re.findall(pattern_numbered, text, re.DOTALL | re.IGNORECASE)

    for q, a in matches:
        q = q.strip()
        a = a.strip()
        # clean up trailing whitespace / newlines in the answer
        a = a.split("\n")[0].strip()

        # basic quality filter
        if len(q) < 5 or len(a) < 2:
            continue
        if q.lower().startswith("answer") or a.lower().startswith("question"):
            continue  # misparse

        pairs.append({"question": q, "answer": a})

    # ── Fallback: simple "Q: ... A: ..." on separate lines ────────
    if not pairs:
        lines = text.strip().split("\n")
        i = 0
        while i < len(lines) - 1:
            qline = lines[i].strip()
            aline = lines[i + 1].strip() if i + 1 < len(lines) else ""
            q_match = re.match(r"(?:\d+[\.\)]\s*)?(?:Q(?:uestion)?)\s*:\s*(.*)", qline, re.IGNORECASE)
            a_match = re.match(r"(?:A(?:nswer)?)\s*:\s*(.*)", aline, re.IGNORECASE)
            if q_match and a_match:
                q = q_match.group(1).strip()
                a = a_match.group(1).strip()
                if len(q) >= 5 and len(a) >= 2:
                    pairs.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1

    return pairs


# ═══════════════════════════════════════════════════════════════════
# Batched synthetic generation
# ═══════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate_synthetic_qa(
    model,
    tokenizer,
    passages: List[str],
    num_pairs: int = 5,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generate synthetic Q&A pairs for multiple passages, batched.

    Args:
        model:          Frozen base model (device_map="auto" across GPUs).
                        If a PEFT model, caller should disable_adapter_layers()
                        before calling this function.
        passages:       List of passage strings.
        num_pairs:      Target number of Q&A pairs per passage.
        batch_size:     Number of passages processed per GPU batch.

    Returns:
        Dict mapping passage text → list of {"question": ..., "answer": ...}
    """
    model.eval()
    device = model.get_input_embeddings().weight.device

    results: Dict[str, List[Dict[str, str]]] = {}
    total_generated = 0
    total_passages = len(passages)

    for i in tqdm(
        range(0, total_passages, batch_size),
        desc="generating synthetic Q&A",
    ):
        batch_passages = passages[i : i + batch_size]

        prompts = [
            PROMPT_GENERATE_QA.format(context=p, n=num_pairs)
            for p in batch_passages
        ]

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

        for j, output_ids in enumerate(outputs):
            input_len = int(encodings["attention_mask"][j].sum().item())
            new_tokens = output_ids[input_len:]
            generated_text = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            # the generation continues from "1. Question:" in the prompt,
            # so prepend that prefix for the parser
            full_text = "1. Question:" + generated_text

            pairs = parse_qa_pairs(full_text)
            pairs = pairs[:num_pairs]  # cap at requested count

            passage_key = batch_passages[j]
            results[passage_key] = pairs
            total_generated += len(pairs)

    avg = total_generated / max(total_passages, 1)
    print(f"  generated {total_generated} Q&A pairs "
          f"({avg:.1f} avg per passage, {total_passages} passages)")

    return results
