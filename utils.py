# utils.py -- dumping all the helper stuff here bc importing from 5 different submodules
# was getting ridiculous. loaders + format merged into one place.
# if this file gets too long ill split it again but for now whatever

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from datasets import load_dataset


# -----------------------------------------------------------------------
# DATA LOADERS
# -----------------------------------------------------------------------

def _read_packaged_jsonl(name: str) -> str:
    # cluster was breaking relative paths so added hardcoded fallback
    try:
        return resources.files("conti.data").joinpath(name).read_text(encoding="utf-8")
    except (TypeError, FileNotFoundError, OSError, ValueError):
        base = Path(__file__).resolve().parent / "conti" / "data"
        return (base / name).read_text(encoding="utf-8")


def _parse_jsonl_prompts(raw_text: str, key: str = "prompt", max_n: int | None = None) -> list[str]:
    prompts: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        p = obj.get(key)
        if p:
            prompts.append(str(p))
        if max_n and len(prompts) >= max_n:
            break
    return prompts


def load_math_prompts(dataset_name: str, split: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    ds = None
    if dataset_name.lower() == "gsm8k":
        try:
            ds = load_dataset("gsm8k", "main", split=split)
        except Exception as e:
            print(f"  [warn] gsm8k load_dataset failed ({e}), downloading raw JSONL...")
            ds = _load_gsm8k_raw_fallback(split, max_samples)
    elif dataset_name.lower() == "math":
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        q = row.get("question") or row.get("problem") or row.get("text")
        if q is None:
            continue
        answer = row.get("answer") or row.get("solution")
        item = {"id": str(row.get("id", i)), "question": str(q).strip()}
        if answer is not None:
            item["answer"] = str(answer).strip()
        rows.append(item)
    return rows


def _load_gsm8k_raw_fallback(split: str, max_samples: int | None = None) -> list[dict]:
    """Download GSM8K directly as JSONL — bypass broken datasets lib."""
    import urllib.request
    urls = {
        "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    }
    url = urls.get(split, urls["train"])
    import os
    cache_dir = Path(os.environ.get("CONTI_GSM8K_CACHE", os.environ.get("SLURM_TMPDIR", "/tmp")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"gsm8k_{split}.jsonl"
    if not cache.exists():
        print(f"  downloading GSM8K {split} from {url}...")
        urllib.request.urlretrieve(url, cache)
    items = []
    with open(cache) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(row)
            if max_samples and len(items) >= max_samples:
                break
    print(f"  loaded {len(items)} GSM8K items via raw fallback")
    return items


_BUNDLED_SAFETY_SETS = {
    "advbench_subset": "advbench_subset.jsonl",
    "xstest_subset": "xstest_subset.jsonl",
    "donotanswer_subset": "donotanswer_subset.jsonl",
}


def load_safety_prompts(name: str, max_samples: int, package_dir: Path | None = None) -> list[str]:
    if name in _BUNDLED_SAFETY_SETS:
        raw = _read_packaged_jsonl(_BUNDLED_SAFETY_SETS[name])
        return _parse_jsonl_prompts(raw, key="prompt", max_n=max_samples)

    # not bundled, try huggingface -- this is slow, only use if you have to
    ds = load_dataset(name, split="train")
    out: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        p = row.get("prompt") or row.get("text") or row.get("instruction")
        if p:
            out.append(str(p))
    return out


def load_all_safety_prompts(names: list[str], max_per_benchmark: int) -> dict[str, list[str]]:
    # sabhi benchmarks ek saath load karo
    result: dict[str, list[str]] = {}
    for name in names:
        try:
            result[name] = load_safety_prompts(name, max_samples=max_per_benchmark)
        except Exception as e:
            print(f"[warn] couldnt load '{name}': {e}")
            result[name] = []
    return result


# -----------------------------------------------------------------------
# CHAT FORMATTING
# -----------------------------------------------------------------------

def messages_to_text(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # gpt2 and other old models dont have chat templates, do it manually
    parts: list[str] = []
    for m in messages:
        role, content = m.get("role", ""), m.get("content", "")
        if role == "user":
            parts.append(f"User: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "".join(parts).strip()


def build_supervised_example(tokenizer, user: str, assistant: str) -> str:
    msgs: list[dict[str, str]] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return messages_to_text(tokenizer, msgs, add_generation_prompt=False)


def user_prompt_only(tokenizer, user: str) -> str:
    msgs: list[dict[str, str]] = [{"role": "user", "content": user}]
    return messages_to_text(tokenizer, msgs, add_generation_prompt=True)


def replay_item_to_text(tokenizer, item: dict[str, Any]) -> str:
    msgs = item.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("replay item needs messages list")
    return messages_to_text(tokenizer, msgs, add_generation_prompt=False)
