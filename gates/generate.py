"""Batched greedy generation with steering hooks attached.

Left-padding is enforced here, not by the caller. The ``last`` steering mode
and the ``h[:, -1, :]`` readout both assume the final column is the final real
token, which is only true when padding is on the left. Mixing padding sides
between the unsteered baseline and the steered arm produces a difference that
looks like a steering effect and is not one.

Generation is greedy and deterministic on purpose: an alpha sweep with
sampling adds a second noise source to a dose-response curve that already has
to resolve small displacements.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["generate", "token_ids_for"]


@torch.no_grad()
def generate(
    model,
    tok,
    prompts: Sequence[str],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    device: str = "cuda:0",
    left_pad: bool = True,
    max_len: int = 1024,
    progress: bool = False,
) -> list[str]:
    """Greedy batched decode. Returns completions only (prompt stripped)."""
    tok.padding_side = "left" if left_pad else "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    out: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = list(prompts[i : i + batch_size])
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        new = gen[:, enc["input_ids"].size(1) :]
        out.extend(tok.batch_decode(new, skip_special_tokens=True))

        if progress:
            done = min(i + batch_size, len(prompts))
            print(f"  gen {done}/{len(prompts)}", flush=True)
    return out


def token_ids_for(tok, strings: Sequence[str], first_only: bool = False) -> list[int]:
    """Map strings to token ids in the model's own vocabulary.

    ``first_only`` takes only the FIRST token of each string, which is the right
    choice for "does the response begin with a refusal" and avoids a multi-token
    initialization spanning several tokens. Never hardcode ids: they are
    tokenizer-dependent, so ids valid for Llama-3 are wrong for Qwen2.5.
    """
    ids: list[int] = []
    for s in strings:
        toks = tok(s, add_special_tokens=False)["input_ids"]
        if not toks:
            raise ValueError(f"string {s!r} produced no tokens")
        ids.append(int(toks[0]) if first_only else int(toks[0]))
    return ids