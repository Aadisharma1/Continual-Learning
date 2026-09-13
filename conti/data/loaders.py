from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _read_packaged_jsonl(name: str) -> str:
    # editable installs sometimes break importlib.resources so fallback to relative
    try:
        return resources.files("conti.data").joinpath(name).read_text(encoding="utf-8")
    except (TypeError, FileNotFoundError, OSError, ValueError):
        base = Path(__file__).resolve().parent
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


def load_math_prompts(
    dataset_name: str,
    split: str,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    if dataset_name.lower() == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
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


_FULL_SAFETY_MAP = {
    "advbench_full": ("walledai/AdvBench", "train"),
    "xstest_full": ("walledai/XSTest", "train"),
    "donotanswer_full": ("LibrAI/do-not-answer", "train"),
}

_BUNDLED_SAFETY_SETS = {
    "advbench_subset": "advbench_subset.jsonl",
    "xstest_subset": "xstest_subset.jsonl",
    "donotanswer_subset": "donotanswer_subset.jsonl",
}


def load_safety_prompts(
    name: str,
    max_samples: int,
    package_dir: Path | None = None,
) -> list[str]:
    if name in _FULL_SAFETY_MAP:
        repo, split = _FULL_SAFETY_MAP[name]
        ds = load_dataset(repo, split=split)
        out: list[str] = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            p = row.get("prompt") or row.get("text") or row.get("instruction") or row.get("question")
            if p:
                out.append(str(p))
        return out

    if name in _BUNDLED_SAFETY_SETS:
        raw = _read_packaged_jsonl(_BUNDLED_SAFETY_SETS[name])
        return _parse_jsonl_prompts(raw, key="prompt", max_n=max_samples)

    # fallback for other HF dataset names (e.g. legacy aliases)
    ds = load_dataset(name, split="train")
    out = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        p = row.get("prompt") or row.get("text") or row.get("instruction") or row.get("question")
        if p:
            out.append(str(p))
    return out


def load_all_safety_prompts(
    names: list[str],
    max_per_benchmark: int,
) -> dict[str, list[str]]:
    # sabhi benchmarks ek saath
    result: dict[str, list[str]] = {}
    for name in names:
        try:
            result[name] = load_safety_prompts(name, max_samples=max_per_benchmark)
        except Exception as e:
            print(f"[warn] couldnt load '{name}': {e}")
            result[name] = []
    return result
