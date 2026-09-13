#!/usr/bin/env python3
"""Frozen Round-0 baseline evaluation for one model (greedy, fixed seed).

Produces the single authoritative ``round0_baseline.json`` per model that every
run directory copies and every paper table reuses. Running this once per model
and reusing the artifact is what guarantees the "round-0 identical across all
tables" invariant from the evaluation protocol.

Usage (DGX):
    python scripts/eval_round0.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --output outputs/round0/llama3-8b-instruct

Usage (local smoke, tiny slices):
    python scripts/eval_round0.py --model gpt2 --smoke --output outputs/round0/gpt2_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from conti.eval.metrics import eval_math_pass1, eval_safety_disentangled  # noqa: E402
from conti.reproducibility import set_global_seed  # noqa: E402

DEFAULT_BENCHMARKS = ("advbench", "xstest", "donotanswer")


def load_model_and_tokenizer(model_id: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="Directory for round0_baseline.json")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=list(DEFAULT_BENCHMARKS),
                        help="Benchmark names (advbench, xstest, donotanswer)")
    parser.add_argument("--math-dataset", type=str, default="gsm8k")
    parser.add_argument("--math-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:0 / cpu (default: auto)")
    parser.add_argument("--smoke", action="store_true",
                        help="Probe-mode slices for local smoke tests ONLY "
                             "(math max_samples=64, safety max=50). Never for paper numbers.")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Optional Llama-Guard model id to score judged ASR "
                             "(default: proxy non-refusal ASR).")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    set_global_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype, device)

    judge_fn = None
    if args.judge_model is not None:
        from conti.eval.judge import LlamaGuardJudge

        judge_fn = LlamaGuardJudge(
            args.judge_model, device=device, batch_size=max(1, args.batch_size // 2)
        ).judge_fn()

    if args.smoke:
        math_result = eval_math_pass1(
            model, tokenizer,
            dataset_name=args.math_dataset, split=args.math_split,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
            max_samples=64,
        )
        safety_result = eval_safety_disentangled(
            model, tokenizer,
            benchmark_names=tuple(args.benchmarks),
            batch_size=args.batch_size, max_new_tokens=256,
            judge_fn=judge_fn, max_samples_per_benchmark=50,
        )
        eval_mode = "smoke"
    else:
        math_result = eval_math_pass1(
            model, tokenizer,
            dataset_name=args.math_dataset, split=args.math_split,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
        safety_result = eval_safety_disentangled(
            model, tokenizer,
            benchmark_names=tuple(args.benchmarks),
            batch_size=args.batch_size, max_new_tokens=256,
            judge_fn=judge_fn,
        )
        eval_mode = "full"

    payload = {
        "model": args.model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "eval_mode": eval_mode,
        "judge": args.judge_model or "none_proxy_asr",
        "math_pass1": math_result,
        "safety": safety_result,
    }

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    destination = out_path / "round0_baseline.json"
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"round-0 baseline written: {destination}")
    print(f"  math accuracy: {math_result['accuracy']:.4f} (n={math_result['n']})")
    for name, benchmark in safety_result.items():
        if name == "aggregate_unsafe":
            print(f"  aggregate ASR: {benchmark['asr']:.4f} (n={benchmark['n']})")
        elif "unsafe" in benchmark:
            print(f"  {name} ASR: {benchmark['unsafe']['asr']:.4f} (n={benchmark['unsafe']['n']})")
        if "safe" in benchmark:
            print(f"  {name} false-refusal rate: {benchmark['safe']['false_refusal_rate']:.4f} "
                  f"(n={benchmark['safe']['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
