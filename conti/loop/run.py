from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from importlib import resources

from accelerate.utils import broadcast_object_list
from tqdm import tqdm

from conti.config_schema import ContiConfig
from conti.eval.drift import DriftTracker, DriftPoint
from conti.eval.metrics import eval_math_pass1, eval_safety_asr_proxy, eval_safety_multi_benchmark
from conti.logging import ExperimentLogger
from conti.replay.buffer import SafetyReplayBuffer
from conti.reproducibility import RunManifest, build_manifest, set_global_seed
from conti.training.sft import SFTTrainer
from conti.verifier.keyword import KeywordVerifier, CompositeVerifier
# pulling helpers from utils instead of navigating 5 nested folders
from utils import load_math_prompts, build_supervised_example, replay_item_to_text, user_prompt_only


def _broadcast_texts(accelerator, texts: list[str] | None) -> list[str]:
    if accelerator.num_processes <= 1:
        return texts or []
    payload = [texts if accelerator.is_main_process else None]
    broadcast_object_list(payload, from_process=0)
    return payload[0] or []


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _extract_num(text: str) -> str | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    # print(f"DEBUG: extracted num {nums[-1] if nums else None}")  # sanity check during eval
    return nums[-1] if nums else None


def _extract_gold_num(gold: str | None) -> str | None:
    if not gold:
        return None
    m = re.search(r"####\s*([-+]?[\d,]+)", gold)
    if m:
        return m.group(1).replace(",", "")
    return _extract_num(gold)


def _is_correct(generated: str, gold: str | None) -> bool:
    g = _extract_gold_num(gold)
    p = _extract_num(generated)
    if g is None or p is None:
        return False
    return p.replace(".", "") == g.replace(".", "")


def _validate_protocol(cfg: ContiConfig) -> None:
    exp = cfg.loop.experiment
    if exp == "baseline_single_sft" and cfg.loop.num_self_improve_rounds != 1:
        raise ValueError("baseline_single_sft must run exactly 1 round")  # ek hi round chalega bhai
    if exp != "phase2_verifier_buffer" and cfg.replay.enabled:
        raise ValueError("replay buffer only for phase2_verifier_buffer")
    if exp == "phase2_verifier_buffer" and not cfg.replay.enabled:
        raise ValueError("phase2_verifier_buffer needs replay.enabled=true")


@torch.inference_mode()
def _gen_batch(model, tokenizer, questions, max_new_tokens, device, temp, top_p):
    texts = [user_prompt_only(tokenizer, q) for q in questions]
    max_in = min(2048, getattr(tokenizer, "model_max_length", 2048) or 2048)
    enc = tokenizer(texts, padding=True, return_tensors="pt", truncation=True, max_length=max_in)
    # print(enc["input_ids"].shape)  # taking up too much VRAM? check this
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens, do_sample=True,
        top_p=top_p, temperature=temp,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = []
    for i, row in enumerate(out):
        inp_len = int(enc["attention_mask"][i].sum())
        gen.append(tokenizer.decode(row[inp_len:], skip_special_tokens=True).strip())
    return gen


def _make_verifier(cfg: ContiConfig):
    if cfg.verifier.mode == "keyword_and_model":
        return CompositeVerifier(cfg.verifier)
    return KeywordVerifier(cfg.verifier)


def _run_evals(cfg, model, tokenizer, device):
    math = eval_math_pass1(
        model, tokenizer, device,
        dataset_name=cfg.data.math_dataset,
        split=cfg.data.math_test_split,
        max_samples=cfg.data.max_prompts_per_eval,
    )
    safety = eval_safety_multi_benchmark(
        model, tokenizer, device,
        benchmark_names=cfg.data.safety_eval_datasets,
        max_per_benchmark=cfg.data.max_prompts_per_eval,
    )
    return {"math_pass1": math, "safety_asr": safety}


def _get_asr_scores(safety_results: dict) -> dict[str, float]:
    return {
        bench: r["asr_proxy"]
        for bench, r in safety_results.items()
        if isinstance(r, dict) and "asr_proxy" in r
    }


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
    _validate_protocol(cfg)
    set_global_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: RunManifest = build_manifest(cfg.to_dict(), cfg.seed)
    manifest.save(out / "run_manifest.json")

    from accelerate import Accelerator

    acc = Accelerator(gradient_accumulation_steps=cfg.loop.gradient_accumulation_steps)
    trainer = SFTTrainer(cfg=cfg, accelerator=acc)
    logger = ExperimentLogger(cfg, run_dir=out)
    drift = DriftTracker()  # har round ke baad check karenge kitna safety gira

    # baseline_frozen: evaluate only
    if cfg.loop.experiment == "baseline_frozen":
        model, tokenizer = trainer.load_model_tokenizer()
        model = trainer.ensure_model_prepared(model)
        model.eval()
        if acc.is_main_process:
            metrics = _run_evals(cfg, model, tokenizer, acc.device)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            drift.set_baseline(_get_asr_scores(metrics.get("safety_asr", {})))
        logger.finish()  # bas evaluate karo, kuch train mat karo

        return

    verifier = _make_verifier(cfg)

    replay_path = cfg.replay.dataset_path
    if replay_path is None:
        replay_path = resources.files("conti.data") / "safety_replay.jsonl"
    replay_buf = SafetyReplayBuffer(replay_path, cfg.seed)
    if not cfg.replay.enabled:
        replay_buf = SafetyReplayBuffer(None, cfg.seed)

    math_items = load_math_prompts(
        cfg.data.math_dataset, cfg.data.math_split,
        max_samples=cfg.loop.max_train_samples_per_round,
    )

    # ========== RESUME LOGIC ==========
    last_completed = _find_last_completed_round(out)

    if last_completed >= 0:
        start_round = last_completed + 1
        if start_round >= cfg.loop.num_self_improve_rounds:
            print(f"[RESUME] All {cfg.loop.num_self_improve_rounds} rounds already done. Skipping to final eval.")
            model, tokenizer = trainer.load_from_checkpoint(str(out / "checkpoints" / f"round_{last_completed}"))
            # jump straight to final save
            filt_log = _load_existing_jsonl(out / "logs" / "filter_log.jsonl")
            rnd_metrics = _load_existing_jsonl(out / "logs" / "round_metrics.jsonl")
            sup_rounds = []
        else:
            ckpt_path = str(out / "checkpoints" / f"round_{last_completed}")
            print(f"[RESUME] Found checkpoint at round {last_completed}. Resuming from round {start_round}.")
            model, tokenizer = trainer.load_from_checkpoint(ckpt_path)
            # reload existing logs
            filt_log = _load_existing_jsonl(out / "logs" / "filter_log.jsonl")
            rnd_metrics = _load_existing_jsonl(out / "logs" / "round_metrics.jsonl")
            sup_rounds = [[]] * start_round
            # restore drift state
            drift_data = _load_existing_jsonl(out / "logs" / "drift_log.jsonl")
            for pt in drift_data:
                drift.baseline_scores.setdefault(pt["benchmark"], pt["baseline_asr"])
                drift.history.append(DriftPoint(**{k: pt[k] for k in ("round_idx", "benchmark", "baseline_asr", "current_asr", "absolute_drift", "relative_drift")}))
    else:
        start_round = 0
        model, tokenizer = trainer.load_model_tokenizer()

        # ewc setup if needed
        if cfg.ewc.enabled:
            from conti.training.ewc import EWCPenalty
            ewc = EWCPenalty(model, lambda_ewc=cfg.ewc.lambda_ewc)
            trainer.set_ewc(ewc)

        # get baseline scores before we touch anything
        if acc.is_main_process:
            m = trainer.ensure_model_prepared(model)
            m.eval()
            base_metrics = _run_evals(cfg, m, tokenizer, acc.device)
            drift.set_baseline(_get_asr_scores(base_metrics.get("safety_asr", {})))
            logger.log_round(-1, {"baseline": base_metrics})

        filt_log: list[dict[str, Any]] = []
        rnd_metrics: list[dict[str, Any]] = []
        sup_rounds: list[list[str]] = []

    for ri in range(start_round, cfg.loop.num_self_improve_rounds):
        if cfg.loop.experiment == "baseline_single_sft" and ri > 0:
            break

        rnd_texts: list[str] = []
        bs = cfg.loop.generate_batch_size
        rej_correct = 0

        if cfg.loop.experiment == "baseline_single_sft":
            if acc.is_main_process:
                rnd_texts = _make_static_sft(tokenizer, math_items)
        else:
            mdl = trainer.ensure_model_prepared(model)
            dev = acc.device
            mdl.eval()

            pend_q: list[str] = []
            pend_m: list[dict[str, Any]] = []

            def flush():
                nonlocal rnd_texts, pend_q, pend_m, rej_correct
                if not pend_q or not acc.is_main_process:
                    return
                qs, meta = pend_q, pend_m
                pend_q, pend_m = [], []

                gens = _gen_batch(
                    mdl, tokenizer, qs,
                    max_new_tokens=min(cfg.loop.generation_max_new_tokens, cfg.model.max_seq_length // 2),
                    device=dev, temp=cfg.loop.generation_temperature,
                    top_p=cfg.loop.generation_top_p,
                )
                for q, g, m in zip(qs, gens, meta, strict=True):
                    if cfg.loop.require_correct_trajectory:
                        if not _is_correct(g, m.get("gold_answer")):
                            rej_correct += 1
                            continue

                    use_v = cfg.loop.experiment in ("phase1_verifier", "phase2_verifier_buffer")
                    if use_v:
                        v = verifier.check_text(g)
                        filt_log.append({
                            "round": ri, "id": m["id"],
                            "safe": v.safe, "reason": v.reason,
                        })
                        if not v.safe:
                            continue

                    rnd_texts.append(build_supervised_example(tokenizer, q, g))

            it = tqdm(math_items, desc=f"gen r{ri}") if acc.is_main_process else math_items
            for item in it:
                if acc.is_main_process:
                    pend_q.append(item["question"])
                    pend_m.append({"id": item["id"], "gold_answer": item.get("answer")})
                    if len(pend_q) >= bs:
                        flush()
            if acc.is_main_process:
                flush()

            # phase2: mix in replay buffer
            if cfg.loop.experiment == "phase2_verifier_buffer" and cfg.replay.enabled and len(replay_buf):
                n = min(
                    cfg.replay.safety_samples_per_batch * max(1, len(rnd_texts) // max(1, bs)),
                    len(replay_buf),
                )
                if acc.is_main_process:
                    for ex in replay_buf.sample(n):
                        try:
                            rnd_texts.append(replay_item_to_text(tokenizer, ex))
                        except (ValueError, KeyError, TypeError):
                            continue

        rnd_texts = _broadcast_texts(acc, rnd_texts)
        sup_rounds.append(rnd_texts)
        accepted = len(rnd_texts)

        rej_safety = 0
        if cfg.loop.experiment in ("phase1_verifier", "phase2_verifier_buffer"):
            rej_safety = sum(1 for x in filt_log if x["round"] == ri and not x["safe"])

        # train
        if cfg.loop.experiment in ("naive_continual", "phase1_verifier", "phase2_verifier_buffer", "baseline_single_sft"):
            stats = trainer.train_on_texts(
                model=model, tokenizer=tokenizer, texts=rnd_texts,
                learning_rate=cfg.loop.learning_rate,
                num_epochs=cfg.loop.num_epochs_per_round,
                micro_batch_size=cfg.loop.train_micro_batch_size,
                warmup_ratio=cfg.loop.warmup_ratio,
                weight_decay=cfg.loop.weight_decay,
                max_seq_length=cfg.model.max_seq_length,
                max_grad_norm=cfg.loop.max_grad_norm,
            )

            ckpt = out / "checkpoints" / f"round_{ri}"
            trainer.save_pretrained(trainer.ensure_model_prepared(model), tokenizer, str(ckpt))

            if acc.is_main_process and stats:
                for si, (loss, ewc_p) in enumerate(zip(stats.get("loss", []), stats.get("ewc_penalty", []))):
                    logger.log_step({"loss": loss, "ewc_penalty": ewc_p}, step=ri * 10000 + si)

        # eval
        if cfg.loop.eval_every_round and acc.is_main_process:
            model.eval()
            ev = _run_evals(cfg, trainer.ensure_model_prepared(model), tokenizer, acc.device)
            ev["round"] = ri
            ev["accepted"] = accepted
            ev["rej_verifier"] = rej_safety
            ev["rej_correctness"] = rej_correct

            cur_asrs = _get_asr_scores(ev.get("safety_asr", {}))
            drift.record(ri, cur_asrs)
            ev["drift"] = drift.get_round_summary(ri)

            rnd_metrics.append(ev)
            logger.log_round(ri, ev)
            logger.log_drift(ri, ev.get("drift", {}))

        # flush logs after every round so they survive crashes
        if acc.is_main_process:
            _write_jsonl(out / "logs" / "filter_log.jsonl", filt_log)
            _write_jsonl(out / "logs" / "round_metrics.jsonl", rnd_metrics)
            drift.save(out / "logs" / "drift_log.jsonl")
            print(f"[CHECKPOINT] Round {ri} complete. Logs saved.", flush=True)

        acc.wait_for_everyone()

    # save everything
    if acc.is_main_process:
        _write_jsonl(out / "logs" / "filter_log.jsonl", filt_log)
        _write_jsonl(out / "logs" / "round_metrics.jsonl", rnd_metrics)
        (out / "supervised_counts.json").write_text(
            json.dumps([len(x) for x in sup_rounds]), encoding="utf-8",
        )
        drift.save(out / "logs" / "drift_log.jsonl")

        model.eval()
        final = _run_evals(cfg, trainer.ensure_model_prepared(model), tokenizer, acc.device)
        final["drift_summary"] = drift.to_summary_dict()
        (out / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

        # ═══════════════════════════════════════════════════════
        #  FINAL RESULTS — printed to stdout for W&B capture
        # ═══════════════════════════════════════════════════════
        _print_final_report(cfg, final, rnd_metrics, drift)

        # W&B artifact backup
        if cfg.logging.use_wandb:
            try:
                import wandb
                for f in (out / "logs").glob("*"):
                    wandb.save(str(f), policy="now")
                wandb.save(str(out / "metrics.json"), policy="now")
            except Exception:
                pass

    logger.finish()


def _print_final_report(cfg, final, rnd_metrics, drift):
    """Print a beautiful ASCII report to stdout — W&B captures this automatically."""
    import sys

    sep = "=" * 72
    thin = "-" * 72

    print(f"\n\n{sep}", flush=True)
    print(f"  CONTI SAFETY — FINAL EXPERIMENT REPORT", flush=True)
    print(f"  Model:      {cfg.model.name_or_path}", flush=True)
    print(f"  Experiment: {cfg.loop.experiment}", flush=True)
    print(f"  Rounds:     {cfg.loop.num_self_improve_rounds}", flush=True)
    print(f"{sep}\n", flush=True)

    # --- Math ---
    math = final.get("math_pass1", {})
    print(f"  MATH (GSM8K Pass@1)", flush=True)
    print(f"  {thin}", flush=True)
    print(f"  Final Accuracy:  {math.get('accuracy', 0):.1%}  (n={math.get('n', 0)})", flush=True)
    print(f"  95% CI:          [{math.get('ci95_low', 0):.3f}, {math.get('ci95_high', 0):.3f}]", flush=True)

    # Per-round math progression
    if rnd_metrics:
        print(f"\n  Per-Round Math Progression:", flush=True)
        print(f"  {'Round':<8} {'Accuracy':<12} {'Accepted':<12} {'Rej(wrong)':<12}", flush=True)
        print(f"  {thin}", flush=True)
        for rm in rnd_metrics:
            r = rm.get("round", "?")
            a = rm.get("math_pass1", {}).get("accuracy", 0)
            acc_n = rm.get("accepted", 0)
            rej_c = rm.get("rej_correctness", 0)
            print(f"  {r:<8} {a:<12.1%} {acc_n:<12} {rej_c:<12}", flush=True)

    # --- Safety ---
    safety = final.get("safety_asr", {})
    print(f"\n  SAFETY (Attack Success Rate — lower is safer)", flush=True)
    print(f"  {thin}", flush=True)
    print(f"  {'Benchmark':<25} {'ASR':<10} {'n':<8} {'95% CI':<20}", flush=True)
    print(f"  {thin}", flush=True)
    for bench, vals in safety.items():
        if not isinstance(vals, dict) or "asr_proxy" not in vals:
            continue
        asr = vals["asr_proxy"]
        n = vals["n"]
        lo = vals.get("ci95_low", 0)
        hi = vals.get("ci95_high", 0)
        print(f"  {bench:<25} {asr:<10.1%} {n:<8} [{lo:.3f}, {hi:.3f}]", flush=True)

    # --- Drift ---
    ds = final.get("drift_summary", {})
    print(f"\n  SAFETY DRIFT SUMMARY", flush=True)
    print(f"  {thin}", flush=True)
    print(f"  Last round:       {ds.get('last_round', '?')}", flush=True)
    print(f"  Worst benchmark:  {ds.get('worst_benchmark', '?')}", flush=True)
    print(f"  Worst abs drift:  {ds.get('worst_abs_drift', 0):.4f}", flush=True)
    fd = ds.get("final_drift", {})
    print(f"\n  Final Drift per Benchmark:", flush=True)
    for bench, d in fd.items():
        direction = "↑ WORSE" if d > 0.01 else ("↓ better" if d < -0.01 else "= stable")
        print(f"    {bench:<25} {d:+.4f}  {direction}", flush=True)

    # --- JSON dump for machine parsing ---
    print(f"\n{sep}", flush=True)
    print(f"  RAW JSON (for programmatic access):", flush=True)
    print(f"{sep}", flush=True)
    print(json.dumps(final, indent=2), flush=True)
    print(f"{sep}", flush=True)
    print(f"  END OF REPORT", flush=True)
    print(f"{sep}\n", flush=True)
    sys.stdout.flush()


def _make_static_sft(tokenizer, math_items):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train")
    texts = []
    for i, row in enumerate(ds):
        if i >= len(math_items):
            break
        texts.append(build_supervised_example(tokenizer, row["question"].strip(), row["answer"].strip()))
    return texts
