"""Multi-round STaR + latent filter experiment loop.

Orchestrates the full iterative self-improvement pipeline over an aligned
Instruct checkpoint:

1. Baseline initialization: LoRA attach, frozen Round-0 evaluation (full
   benchmarks), baseline safety Fisher ``F_0`` over refusal data, baseline
   residual-stream capture and safety gradient anchor.
2. Dual gating per round: math-correctness verification (exact numeric match)
   plus Gate R (:class:`conti.loop.subliminal_filter.SubliminalFilter`) on the
   conditions that use it.
3. Training under one of four experimental conditions:

   * ``star_only``       -- correctness filtering only (pathology arm).
   * ``safety_replay``   -- Gate R + minimal safety replay mixed per batch.
   * ``ewc_regularized`` -- Gate R + true EWC penalty from ``F_0``.
   * ``bianchi_control`` -- single-shot static tuning at Round 0 with a fixed
     safety fraction (Bianchi et al. 2023 recipe); no further training rounds.

4. Per-round JSONL logging to ``logs/round_metrics.jsonl``,
   ``logs/drift_log.jsonl`` and ``logs/latent_filter.jsonl``.

Distribution: runs under ``torchrun``/Accelerate. Generation is sharded across
ranks and re-assembled via object gather; training text lists are broadcast so
every rank trains on identical data (DDP-averaged gradients). Gate-R and Fisher
computations are performed redundantly on every rank over identical data so no
cross-rank synchronization of large gradient/Fisher states is required.
"""

from __future__ import annotations

import json
import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from conti.config_schema import ContiConfig
from conti.data.loaders import load_math_prompts
from conti.eval.drift import DriftTracker
from conti.eval.metrics import (
    DatasetCountError,
    eval_math_pass1,
    eval_safety_disentangled,
    generate_batch,
    is_correct_prediction,
)
from conti.loop.subliminal_filter import SubliminalFilter, make_probe_batches
from conti.replay.buffer import SafetyReplayBuffer
from conti.reproducibility import build_manifest, set_global_seed
from conti.training.ewc import (
    EWCPenaltyLoss,
    EmpiricalFisherEstimator,
    SafetyTextDataset,
    make_fisher_collate,
)
from conti.training.sft import SFTTrainer

logger = logging.getLogger(__name__)

ALLOWED_CONDITIONS = (
    "baseline_frozen",
    "star_only",
    "safety_replay",
    "ewc_regularized",
    "bianchi_control",
)
GATE_R_CONDITIONS = ("safety_replay", "ewc_regularized")

_BENCHMARK_ALIASES = {
    "advbench": "advbench",
    "adv_bench": "advbench",
    "xstest": "xstest",
    "donotanswer": "donotanswer",
    "do_not_answer": "donotanswer",
    "dna": "donotanswer",
}


def _normalize_benchmark_names(raw: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for name in raw:
        key = str(name).lower().replace("_subset", "").replace("_full", "")
        if key not in _BENCHMARK_ALIASES:
            raise ValueError(
                f"Unknown safety benchmark '{name}'. Supported: advbench, xstest, donotanswer."
            )
        canonical = _BENCHMARK_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("safety_eval_datasets resolved to an empty benchmark list.")
    return tuple(normalized)


def _find_resume_round(out_dir: Path) -> int:
    """Highest round whose checkpoint directory contains a completed ``.done`` sentinel.

    A checkpoint dir without ``.done`` (e.g. a crashed save) is ignored, so a
    resume always restarts from a fully persisted round.
    """
    checkpoints_root = out_dir / "checkpoints"
    if not checkpoints_root.exists():
        return -1
    highest = -1
    for child in checkpoints_root.iterdir():
        if child.is_dir() and (child / ".done").exists():
            try:
                round_index = int(child.name.split("_")[-1])
            except ValueError:
                continue
            highest = max(highest, round_index)
    return highest


# --------------------------------------------------------------------------- #
# small IO / distribution helpers
# --------------------------------------------------------------------------- #


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _broadcast_object(accelerator, obj: Any) -> Any:
    if accelerator.num_processes <= 1:
        return obj
    from accelerate.utils import broadcast_object_list

    payload = [obj if accelerator.is_main_process else None]
    broadcast_object_list(payload, from_process=0)
    return payload[0]


def _gather_objects(accelerator, obj: Any) -> list[Any]:
    if accelerator.num_processes <= 1:
        return [obj]
    import torch.distributed as dist

    gathered: list[Any] = [None] * accelerator.num_processes
    dist.all_gather_object(gathered, obj)
    return gathered


def _resolve_safety_jsonl(cfg: ContiConfig) -> Path:
    if cfg.replay.dataset_path:
        path = Path(cfg.replay.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"replay.dataset_path does not exist: {path}")
        return path
    try:
        traversable = resources.files("conti.data").joinpath("safety_replay.jsonl")
    except (ModuleNotFoundError, TypeError) as exc:  # pragma: no cover - packaging edge
        raise FileNotFoundError(
            "No replay.dataset_path configured and packaged conti/data/safety_replay.jsonl "
            f"could not be located: {exc}"
        ) from exc
    return Path(str(traversable))


def _format_generation_prompt(tokenizer: Any, question: str) -> str:
    chat = getattr(tokenizer, "apply_chat_template", None)
    if callable(chat):
        try:
            return str(
                chat(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - fall back to plain prompt
            logger.warning("apply_chat_template failed (%s); using plain prompt.", exc)
    return f"{question}\n\nAnswer:"


def _format_supervised_text(tokenizer: Any, question: str, response: str) -> str:
    chat = getattr(tokenizer, "apply_chat_template", None)
    eos = getattr(tokenizer, "eos_token", "") or ""
    if callable(chat):
        try:
            return str(
                chat(
                    [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_chat_template failed (%s); using plain concatenation.", exc)
    return f"{question}\n\nAnswer:{response}{eos}"


def _replay_item_to_text(tokenizer: Any, item: Mapping[str, Any]) -> str:
    if "prompt" in item and "response" in item:
        return _format_supervised_text(tokenizer, str(item["prompt"]), str(item["response"]))
    if "text" in item:
        return str(item["text"])
    raise ValueError(f"Replay item has neither prompt/response nor text: keys={sorted(item.keys())}")


def _safety_scores_for_drift(safety_results: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, benchmark in safety_results.items():
        if name == "aggregate_unsafe":
            scores["aggregate"] = float(benchmark["asr"])
            continue
        if "unsafe" in benchmark:
            scores[name] = float(benchmark["unsafe"]["asr"])
        if "safe" in benchmark:
            scores[f"{name}_safe_frr"] = float(benchmark["safe"]["false_refusal_rate"])
    return scores


def _move_batch_to(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _make_fisher_batches(
    safety_path: Path,
    tokenizer: Any,
    cfg: ContiConfig,
    device: torch.device,
    max_rows: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    dataset = SafetyTextDataset(safety_path)
    loader = DataLoader(
        dataset,
        batch_size=cfg.ewc.fisher_batch_size,
        shuffle=False,
        collate_fn=make_fisher_collate(tokenizer, max_len=cfg.ewc.fisher_max_len),
        num_workers=0,
    )
    batches: list[dict[str, torch.Tensor]] = []
    rows = 0
    for batch in loader:
        batches.append(_move_batch_to(batch, device))
        rows += int(batch["input_ids"].shape[0])
        if max_rows is not None and rows >= max_rows:
            break
    if not batches:
        raise RuntimeError(f"Safety dataset at {safety_path} produced zero batches.")
    return batches


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #


def _run_full_evals(cfg: ContiConfig, model: Any, tokenizer: Any) -> dict[str, Any]:
    math_result = eval_math_pass1(
        model,
        tokenizer,
        dataset_name=cfg.data.math_dataset,
        split=cfg.data.math_test_split,
        batch_size=16,
        max_new_tokens=cfg.loop.eval_max_new_tokens,
    )
    safety_result = eval_safety_disentangled(
        model,
        tokenizer,
        benchmark_names=_normalize_benchmark_names(cfg.data.safety_eval_datasets),
        batch_size=16,
        max_new_tokens=256,
    )
    return {"math_pass1": math_result, "safety": safety_result}


def _run_probe_evals(cfg: ContiConfig, model: Any, tokenizer: Any) -> dict[str, Any]:
    math_result = eval_math_pass1(
        model,
        tokenizer,
        dataset_name=cfg.data.math_dataset,
        split=cfg.data.math_test_split,
        batch_size=16,
        max_new_tokens=cfg.loop.eval_max_new_tokens,
        max_samples=cfg.loop.probe_eval_size,
    )
    safety_result = eval_safety_disentangled(
        model,
        tokenizer,
        benchmark_names=_normalize_benchmark_names(cfg.data.safety_eval_datasets),
        batch_size=16,
        max_new_tokens=256,
        max_samples_per_benchmark=min(100, cfg.loop.probe_eval_size),
    )
    return {"math_pass1": math_result, "safety": safety_result, "eval_mode": "probe"}


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


def _find_last_completed_round(out: Path) -> int:
    """Check checkpoint dirs to find last completed round. Returns -1 if none."""
    ckpt_dir = out / "checkpoints"
    if not ckpt_dir.exists():
        return -1
    rounds = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and d.name.startswith("round_"):
            try:
                r = int(d.name.split("_")[1])
                if any(f.name in ("adapter_config.json", "config.json") for f in d.iterdir()):
                    rounds.append(r)
            except (ValueError, IndexError):
                continue
    return max(rounds) if rounds else -1


def _load_existing_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def run_experiment(cfg: ContiConfig) -> None:
    """Execute a full multi-round experiment described by ``cfg``."""
    if cfg.loop.experiment not in ALLOWED_CONDITIONS:
        raise ValueError(
            f"Unknown experiment '{cfg.loop.experiment}'. Allowed: {ALLOWED_CONDITIONS}"
        )
    condition = cfg.loop.experiment
    set_global_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(cfg.to_dict(), cfg.seed)
    manifest.save(out_dir / "run_manifest.json")

    from accelerate import Accelerator

    accelerator = Accelerator(gradient_accumulation_steps=cfg.loop.gradient_accumulation_steps)
    world_size = accelerator.num_processes
    device = accelerator.device
    logger.info("condition=%s world_size=%d device=%s", condition, world_size, device)

    trainer = SFTTrainer(cfg=cfg, accelerator=accelerator)
    model, tokenizer = trainer.load_model_tokenizer()
    generation_model = accelerator.unwrap_model(model)

    safety_path = _resolve_safety_jsonl(cfg)
    replay_buf = SafetyReplayBuffer(safety_path, cfg.seed)

    # ---------------- Round -1: frozen baseline state ----------------------- #
    model_for_eval = generation_model
    baseline_metrics = _run_full_evals(cfg, model_for_eval, tokenizer)
    if accelerator.is_main_process:
        _append_jsonl(
            logs_dir / "round_metrics.jsonl",
            {"round": -1, "condition": condition, **baseline_metrics, "eval_mode": "full"},
        )
    drift = DriftTracker()
    drift.set_baseline(_safety_scores_for_drift(baseline_metrics["safety"]))

    # baseline Fisher F_0 (identical data/seed on every rank -> identical state)
    fisher_batches = _make_fisher_batches(
        safety_path, tokenizer, cfg, device, max_rows=cfg.ewc.fisher_samples
    )
    fisher_estimator = EmpiricalFisherEstimator(
        generation_model,
        grad_clip=cfg.ewc.fisher_grad_clip,
        eps=cfg.ewc.fisher_eps,
        store_device=cfg.ewc.fisher_store_device,
    )
    fisher_baseline = fisher_estimator.compute_fisher(fisher_batches)
    ref_params_baseline = {
        name: param.detach().clone()
        for name, param in generation_model.named_parameters()
        if param.requires_grad
    }

    subliminal = SubliminalFilter(
        generation_model,
        layer_indices=cfg.subliminal.layer_indices,
        tau_drift=cfg.subliminal.tau_drift,
        store_device=cfg.ewc.fisher_store_device,
    )
    subliminal.set_baseline_fisher(fisher_baseline)

    probe_prompts = [
        str(item.get("prompt", item.get("text", "")))
        for item in replay_buf.sample(cfg.subliminal.probe_samples)
    ]
    probe_prompts = [p for p in probe_prompts if p]
    if not probe_prompts:
        raise RuntimeError("Replay buffer produced zero probe prompts; cannot initialize Gate R.")
    probe_batches_cpu = make_probe_batches(
        tokenizer, probe_prompts, max_len=cfg.subliminal.probe_max_len
    )
    probe_batches = [_move_batch_to(b, device) for b in probe_batches_cpu]
    subliminal.capture_baseline_representations(probe_batches)

    anchor_batches = fisher_batches[: max(1, cfg.subliminal.anchor_samples // cfg.ewc.fisher_batch_size)]
    subliminal.capture_safety_anchor(
        anchor_batches, grad_clip=cfg.subliminal.grad_clip, max_samples=cfg.subliminal.anchor_samples
    )

    if condition == "ewc_regularized":
        ewc_penalty = EWCPenaltyLoss(
            fisher_baseline, ref_params_baseline, lambda_ewc=cfg.ewc.lambda_ewc
        )
        trainer.set_ewc(ewc_penalty)

    if condition == "baseline_frozen":
        if accelerator.is_main_process:
            (out_dir / "metrics.json").write_text(
                json.dumps({"baseline": baseline_metrics}, indent=2, default=str),
                encoding="utf-8",
            )
        accelerator.wait_for_everyone()
        return

    math_items = load_math_prompts(
        cfg.data.math_dataset,
        cfg.data.math_split,
        max_samples=cfg.loop.max_train_samples_per_round,
    )
    if not math_items:
        raise RuntimeError("load_math_prompts returned zero training prompts.")
    num_rounds = cfg.loop.num_self_improve_rounds
    full_eval_rounds = set(
        cfg.loop.full_eval_rounds if cfg.loop.full_eval_rounds is not None else {num_rounds - 1}
    )

    resume_round = _find_resume_round(out_dir) if cfg.loop.resume else -1
    if resume_round >= 0:
        trainer.load_adapter_checkpoint(model, str(out_dir / "checkpoints" / f"round_{resume_round}"))
        logger.info(
            "Resume: found completed round %d; rounds 0..%d will be skipped.", resume_round, resume_round
        )

    # ---------------- multi-round self-improvement -------------------------- #
    for round_index in range(num_rounds):
        if round_index <= resume_round:
            continue
        round_started = time.time()
        shard = math_items[round_index::world_size]
        generated: list[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(shard), cfg.loop.generate_batch_size),
            desc=f"gen r{round_index}",
            disable=not accelerator.is_local_main_process,
            leave=False,
        ):
            chunk = shard[start: start + cfg.loop.generate_batch_size]
            prompts = [_format_generation_prompt(tokenizer, item["question"]) for item in chunk]
            responses = generate_batch(
                generation_model,
                tokenizer,
                prompts,
                max_new_tokens=cfg.loop.generation_max_new_tokens,
                do_sample=True,
                temperature=cfg.loop.generation_temperature,
                top_p=cfg.loop.generation_top_p,
            )
            for item, response in zip(chunk, responses):
                generated.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "gold": item.get("answer"),
                        "response": response,
                    }
                )
        accelerator.wait_for_everyone()
        candidates: list[dict[str, Any]] = []
        for rank_payload in _gather_objects(accelerator, generated):
            candidates.extend(rank_payload)
        candidates = sorted(candidates, key=lambda c: str(c["id"]))

        correct_pass: list[dict[str, Any]] = []
        rejected_correct = 0
        if cfg.loop.require_correct_trajectory:
            for candidate in candidates:
                if is_correct_prediction(candidate["response"], candidate["gold"]):
                    correct_pass.append(candidate)
                else:
                    rejected_correct += 1
        else:
            correct_pass = list(candidates)

        gate_rejected = 0
        gate_rows: list[dict[str, Any]] = []
        accepted_candidates = correct_pass
        if condition in GATE_R_CONDITIONS and correct_pass:
            gate_device_batches: list[dict[str, torch.Tensor]] = []
            for start in range(0, len(correct_pass), cfg.loop.generate_batch_size):
                chunk = correct_pass[start: start + cfg.loop.generate_batch_size]
                texts = [
                    _format_supervised_text(tokenizer, c["question"], c["response"])
                    for c in chunk
                ]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.model.max_seq_length,
                )
                labels = encoded["input_ids"].clone()
                labels.masked_fill_(encoded["attention_mask"] == 0, -100)
                encoded["labels"] = labels
                gate_device_batches.append(_move_batch_to(dict(encoded), device))
            grad_results = subliminal.gradient_alignment_test(
                gate_device_batches,
                tau_drift=cfg.subliminal.tau_drift,
                grad_clip=cfg.subliminal.grad_clip,
            )
            kept: list[dict[str, Any]] = []
            for candidate, grad_result in zip(correct_pass, grad_results):
                gate_rows.append(
                    {
                        "round": round_index,
                        "item_id": candidate["id"],
                        "cosine": grad_result["cosine"],
                        "accepted": grad_result["accepted"],
                        "reason": grad_result["reason"],
                    }
                )
                if grad_result["accepted"]:
                    kept.append(candidate)
                else:
                    gate_rejected += 1
            accepted_candidates = kept
        if accelerator.is_main_process:
            for row in gate_rows:
                _append_jsonl(logs_dir / "latent_filter.jsonl", row)

        train_texts = [
            _format_supervised_text(tokenizer, c["question"], c["response"])
            for c in accepted_candidates
        ]

        static_mix_count = 0
        if condition == "bianchi_control":
            if round_index == 0:
                static_mix_count = max(1, int(cfg.replay.bianchi_safety_fraction * max(1, len(train_texts))))
                for item in replay_buf.sample(static_mix_count):
                    train_texts.append(_replay_item_to_text(tokenizer, item))
            else:
                train_texts = []
        elif condition == "safety_replay" and cfg.replay.enabled and len(replay_buf) > 0:
            num_batches = max(1, -(-len(train_texts) // max(1, cfg.loop.train_micro_batch_size)))
            replay_count = min(
                cfg.replay.safety_samples_per_batch * num_batches,
                len(replay_buf),
                max(0, len(train_texts) // 2),
            )
            for item in replay_buf.sample(replay_count):
                train_texts.append(_replay_item_to_text(tokenizer, item))

        if world_size > 1:
            train_texts = _broadcast_object(accelerator, train_texts)

        training_stats: dict[str, list[float]] = {"loss": []}
        if train_texts:
            training_stats = trainer.train_on_texts(
                model=model,
                tokenizer=tokenizer,
                texts=train_texts,
                learning_rate=cfg.loop.learning_rate,
                num_epochs=cfg.loop.num_epochs_per_round,
                micro_batch_size=cfg.loop.train_micro_batch_size,
                warmup_ratio=cfg.loop.warmup_ratio,
                weight_decay=cfg.loop.weight_decay,
                max_seq_length=cfg.model.max_seq_length,
                max_grad_norm=cfg.loop.max_grad_norm,
            )
            checkpoint_dir = out_dir / "checkpoints" / f"round_{round_index}"
            trainer.save_pretrained(model, tokenizer, str(checkpoint_dir))
            (checkpoint_dir / ".done").write_text("ok\n", encoding="utf-8")
        accelerator.wait_for_everyone()

        eval_mode = "full" if round_index in full_eval_rounds else "probe"
        if accelerator.is_main_process:
            eval_results = (
                _run_full_evals(cfg, model_for_eval, tokenizer)
                if eval_mode == "full"
                else _run_probe_evals(cfg, model_for_eval, tokenizer)
            )
            drift.record(round_index, _safety_scores_for_drift(eval_results["safety"]))
            mean_loss = training_stats.get("loss", [])
            round_row = {
                "round": round_index,
                "condition": condition,
                "seed": cfg.seed,
                "eval_mode": eval_mode,
                "n_candidates": len(candidates),
                "rej_correct": rejected_correct,
                "rej_gate_r": gate_rejected,
                "accepted": len(accepted_candidates),
                "static_mix_count": static_mix_count,
                "mean_train_loss": sum(mean_loss) / max(1, len(mean_loss)),
                "wallclock_s": time.time() - round_started,
                **eval_results,
            }
            _append_jsonl(logs_dir / "round_metrics.jsonl", round_row)
            for benchmark, payload in drift.get_round_summary(round_index).items():
                _append_jsonl(
                    logs_dir / "drift_log.jsonl",
                    {"round": round_index, "benchmark": benchmark, **payload},
                )

        if (round_index + 1) % max(1, cfg.subliminal.fisher_every_n_rounds) == 0:
            current_estimator = EmpiricalFisherEstimator(
                generation_model,
                grad_clip=cfg.ewc.fisher_grad_clip,
                eps=cfg.ewc.fisher_eps,
                store_device=cfg.ewc.fisher_store_device,
            )
            fisher_current = current_estimator.compute_fisher(
                fisher_batches, max_samples=cfg.ewc.fisher_samples
            )
            subliminal.set_current_fisher(fisher_current)
            curvature = subliminal.evaluate_refusal_curvature()
            representation = subliminal.compute_representation_drift(probe_batches)
            if accelerator.is_main_process:
                _append_jsonl(
                    logs_dir / "latent_filter.jsonl",
                    {
                        "round": round_index,
                        "event": "latent_drift",
                        "fisher_curvature_frobenius": curvature,
                        "fisher_divergence": subliminal.last_curvature_details,
                        "representation_drift": representation,
                        "tau_drift": cfg.subliminal.tau_drift,
                        "max_allowed_curvature": cfg.subliminal.max_allowed_curvature,
                    },
                )
        accelerator.wait_for_everyone()

    # ---------------- final full evaluation -------------------------------- #
    if accelerator.is_main_process:
        final_metrics = _run_full_evals(cfg, model_for_eval, tokenizer)
        (out_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "condition": condition,
                    "seed": cfg.seed,
                    "final": final_metrics,
                    "drift_summary": drift.to_summary_dict(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        _append_jsonl(
            logs_dir / "round_metrics.jsonl",
            {"round": num_rounds, "condition": condition, "eval_mode": "final", **final_metrics},
        )
    accelerator.wait_for_everyone()


__all__ = ["run_experiment", "ALLOWED_CONDITIONS", "GATE_R_CONDITIONS", "DatasetCountError"]
