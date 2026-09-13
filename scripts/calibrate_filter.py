#!/usr/bin/env python3
"""Calibrate the Gate R gradient-alignment threshold tau_drift.

Protocol (docs/IDEATION_subliminal_pivot.md section 7.4):

1. Compute the safety gradient anchor over refusal examples.
2. Score a sample of human-gold GSM8K solutions (capability floor): their
   cosines with the anchor form the "must-pass" distribution.
3. tau = the (1 - min_pass_rate) quantile of the gold cosines, so at least
   ``min_pass_rate`` (default 0.90) of human-gold traces pass the gate.
4. Optionally score a "bad" corpus (e.g. misaligned-teacher traces) and emit an
   ROC/CSV for the paper figure.

Writes ``calibration_report.json`` with tau, pass rates, cosine distributions
and (optionally) the ROC. Run on the model+LoRA state that will actually be
gated; a tau calibrated on different weights is meaningless.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from conti.config_schema import load_config  # noqa: E402
from conti.data.loaders import load_math_prompts  # noqa: E402
from conti.loop.run import (  # noqa: E402
    _format_supervised_text,
    _move_batch_to,
    _resolve_safety_jsonl,
)
from conti.loop.subliminal_filter import SubliminalFilter  # noqa: E402
from conti.reproducibility import set_global_seed  # noqa: E402
from conti.training.ewc import (  # noqa: E402
    SafetyTextDataset,
    make_fisher_collate,
    make_safety_fisher_dataloader,
)
from conti.training.sft import SFTTrainer  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


def choose_threshold(cosines: list[float], min_pass_rate: float) -> float:
    """Largest tau such that at least ``min_pass_rate`` of cosines are >= tau.

    For an empty list raises ValueError; pass rates outside (0, 1] raise too.
    """
    if not cosines:
        raise ValueError("choose_threshold needs at least one cosine value.")
    if not 0.0 < min_pass_rate <= 1.0:
        raise ValueError(f"min_pass_rate must be in (0, 1], got {min_pass_rate}.")
    ordered = sorted(cosines)
    # index of the value below which (1 - pass_rate) of the mass sits
    target_index = int(round((1.0 - min_pass_rate) * len(ordered)))
    target_index = min(max(target_index, 0), len(ordered) - 1)
    return float(ordered[target_index])


def roc_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney U) of positive vs negative cosine scores."""
    if not positive_scores or not negative_scores:
        raise ValueError("ROC AUC needs both a positive and a negative score set.")
    combined = [(score, 1) for score in positive_scores] + [(score, 0) for score in negative_scores]
    combined.sort(key=lambda item: item[0])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(combined):
        tie_end = index
        while tie_end + 1 < len(combined) and combined[tie_end + 1][0] == combined[index][0]:
            tie_end += 1
        average_rank = (index + tie_end) / 2.0 + 1.0
        for position in range(index, tie_end + 1):
            ranks[position] = average_rank
        index = tie_end + 1
    rank_sum_positive = 0.0
    for position, (score, label) in enumerate(combined):
        if label == 1:
            rank_sum_positive += ranks[position]
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    u = rank_sum_positive - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


class _TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        return {"text": self._texts[index]}


def _corpus_batches(
    texts: list[str],
    tokenizer,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    loader = DataLoader(
        _TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_fisher_collate(tokenizer, max_len=max_len),
        num_workers=0,
    )
    return [
        {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
        for batch in loader
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="A run YAML (e.g. from configs/generated) so calibration uses the "
                             "exact model/LoRA/subliminal settings that will be gated.")
    parser.add_argument("--output-dir", type=str, default="outputs/calibration")
    parser.add_argument("--gold-samples", type=int, default=256)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--bad-corpus", type=str, default=None,
                        help="Optional JSONL of drift-inducing traces for the ROC "
                             "(rows with prompt/response or text).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from accelerate import Accelerator

    accelerator = Accelerator()
    device = accelerator.device
    trainer = SFTTrainer(cfg=cfg, accelerator=accelerator)
    model, tokenizer = trainer.load_model_tokenizer()
    model_for_scoring = accelerator.unwrap_model(model)

    safety_path = _resolve_safety_jsonl(cfg)
    anchor_loader = make_safety_fisher_dataloader(
        safety_path, tokenizer,
        batch_size=cfg.ewc.fisher_batch_size, max_len=cfg.ewc.fisher_max_len,
    )
    anchor_batches = [
        _move_batch_to(batch, device) for batch in list(anchor_loader)[: max(1, cfg.subliminal.anchor_samples)]
    ]

    filtr = SubliminalFilter(
        model_for_scoring,
        layer_indices=cfg.subliminal.layer_indices,
        tau_drift=cfg.subliminal.tau_drift,
    )
    anchor = filtr.capture_safety_anchor(
        anchor_batches, grad_clip=cfg.subliminal.grad_clip, max_samples=cfg.subliminal.anchor_samples
    )

    gold_items = load_math_prompts(
        cfg.data.math_dataset, cfg.data.math_split, max_samples=args.gold_samples
    )
    if not gold_items:
        raise RuntimeError("No gold math items loaded; cannot calibrate.")
    gold_texts = [
        _format_supervised_text(tokenizer, item["question"], str(item.get("answer", "")))
        for item in gold_items
    ]
    gold_batches = _corpus_batches(
        gold_texts, tokenizer, cfg.model.max_seq_length, cfg.ewc.fisher_batch_size, device
    )
    gold_results = filtr.gradient_alignment_test(
        gold_batches, safety_anchor=anchor, tau_drift=-1.0, grad_clip=cfg.subliminal.grad_clip
    )
    gold_cosines = [result["cosine"] for result in gold_results if result["cosine"] == result["cosine"]]

    tau = choose_threshold(gold_cosines, args.min_pass_rate)
    passing = sum(1 for value in gold_cosines if value >= tau)

    report: dict = {
        "config": args.config,
        "model": cfg.model.name_or_path,
        "n_gold": len(gold_cosines),
        "min_pass_rate": args.min_pass_rate,
        "tau_drift": tau,
        "gold_pass_rate_at_tau": passing / max(1, len(gold_cosines)),
        "gold_cosine_mean": statistics.fmean(gold_cosines) if gold_cosines else None,
        "gold_cosine_stdev": statistics.stdev(gold_cosines) if len(gold_cosines) > 1 else 0.0,
        "gold_cosine_min": min(gold_cosines) if gold_cosines else None,
        "gold_cosine_max": max(gold_cosines) if gold_cosines else None,
    }

    csv_path = output_dir / "gold_cosines.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "cosine"])
        for index, value in enumerate(gold_cosines):
            writer.writerow([index, value])

    if args.bad_corpus:
        bad_dataset = SafetyTextDataset(args.bad_corpus)
        bad_texts = []
        for row in bad_dataset:
            if "prompt" in row and "response" in row:
                bad_texts.append(_format_supervised_text(tokenizer, str(row["prompt"]), str(row["response"])))
            else:
                bad_texts.append(str(row["text"]))
        bad_batches = _corpus_batches(
            bad_texts, tokenizer, cfg.model.max_seq_length, cfg.ewc.fisher_batch_size, device
        )
        bad_results = filtr.gradient_alignment_test(
            bad_batches, safety_anchor=anchor, tau_drift=-1.0, grad_clip=cfg.subliminal.grad_clip
        )
        bad_cosines = [r["cosine"] for r in bad_results if r["cosine"] == r["cosine"]]
        report["bad_corpus"] = {
            "path": args.bad_corpus,
            "n": len(bad_cosines),
            "auc_vs_gold": roc_auc(bad_cosines, gold_cosines),
            "rejected_at_tau": sum(1 for value in bad_cosines if value < tau),
        }
        bad_csv = output_dir / "bad_cosines.csv"
        with bad_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "cosine"])
            for index, value in enumerate(bad_cosines):
                writer.writerow([index, value])

    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"\ntau_drift = {tau:.4f}  -> set this in configs/subliminal.tau_drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
