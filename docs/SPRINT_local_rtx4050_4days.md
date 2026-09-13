# Local Sprint — RTX 4050 (Days 1–4), DGX-Ready Handoff

**Constraint:** RTX 4050 (6 GB) until Day 4 → **no 7–8B training/generation/eval locally** (bf16 weights alone exceed VRAM; quantized results don't count for the paper). Everything below uses tiny models (GPT-2 124M, TinyLlama-1.1B, Llama-Guard-3-**1B** ≈2 GB) for correctness testing. Goal: **when the 8×A100 DGX lands on Day 5, we `git clone && bash launch.sh` and the pilot starts same-day.**

---

## Day 1 — Hygiene + Evaluation Freeze
- [ ] Repo cleanup: delete `(1)` / `-Revenant` duplicate files; `.gitignore` outputs/checkpoints; flag WandB key rotation.
- [ ] Verify dataset loaders + counts: GSM8K 1,319 · AdvBench 520 · DoNotAnswer 939 · **XSTest split 200-safe / 50-unsafe** (fix loader if pooled).
- [ ] Judged-ASR harness: refusal-keyword rate + judge scoring path; wire Llama-Guard-3 (8B on DGX, 1B for local spot-checks); publish judge prompts in `configs/`.
- [ ] Frozen round-0 protocol doc: greedy decode, fixed seed, single shared eval per model (kills the Table-1-vs-round-0 mismatch bug).
- ✅ *Exit: `pytest tests/test_eval.py` green; loaders print exact expected counts.*

## Day 2 — Latent Telemetry (the measurement layer)
- [ ] Refusal-direction extraction (difference-in-means over harmful/benign probe pairs, Arditi et al. method) — implemented + unit-tested on GPT-2.
- [ ] Hidden-state capture hooks (layer ℓ*, mean-over-tokens) + SDS-act / SDS-param metrics.
- [ ] Probe set definition: fixed harmful/benign prompt pairs, versioned in `conti/data/probes/`.
- ✅ *Exit: telemetry recovers a sane direction & nonzero drift signal on a deliberately-mis-finetuned GPT-2 toy run.*

## Day 3 — Science Modules
- [ ] `subliminal_filter.py` logic: per-candidate LoRA-space gradient g_i, safety-anchor gradient g_safe, cosine scoring, τ calibration harness (≥90% human-gold pass-rate constraint) — unit-tested on tiny LoRA.
- [ ] `ewc_baseline.py`: diagonal Fisher over safety data (sharded-ready API for 8 GPUs), EWC penalty fused into trainer; verify penalty reduces drift on toy forgetting task.
- [ ] Semantic gate integration point: Llama-Guard verdict → filter log schema (`filter_log.jsonl` v2 with stage/rejection/score columns).
- ✅ *Exit: synthetic test where gradient filter ranks "drift-inducing" toy data below benign data.*

## Day 4 — Loop Wiring + DGX Launch Kit
- [ ] STaR loop refactor: generation → semantic gate → subliminal gate → SFT (+replay/EWC arms) → telemetry → eval, with resume logic preserved.
- [ ] End-to-end **smoke test on the 4050**: GPT-2-scale, 2 rounds, n=50 prompts, <2 h wallclock — proves full pipeline correctness.
- [ ] DGX kit: `torchrun`/accelerate launch scripts, 8×A100 configs for the full grid (E1 conditions × 2 models × seeds; E3 arms; EWC/Bianchi baselines), config-schema validation, README runbook.
- [ ] Push to GitHub (`Aadisharma1/Conti`) — tag `dgx-ready`.
- ✅ *Exit: smoke test green end-to-end; fresh-clone instructions verified.*

---

## NOT done locally (starts on DGX, Day 5+)
Round-0 frozen baselines (both Instruct models) → M1 pilot + **Gate G1** (does drift exceed seed noise?) → main grid → mediation arms → EWC/Bianchi baselines.

## Risks during sprint
- HF gated-model access (Llama-3-8B-Instruct, Llama-Guard-3): request tokens **Day 1**, approval can take days.
- Don't let Day-3 math perfectionism eat Day-4 wiring — the smoke test is the deliverable that saves the cluster week.
