# Supervisor Brief — Pivot to Subliminal Safety Erosion in Self-Improving LLMs

**Date:** 2026-08-21 · **Status:** awaiting sign-off · **Full detail:** `docs/IDEATION_subliminal_pivot.md`

---

## The Idea (30 seconds)

STaR-style self-improvement trains an aligned LLM on its own math reasoning traces. Those traces are textually benign — **Llama-Guard-3 rejects essentially none of them** — yet safety alignment still erodes round after round. Following Cloud et al. (arXiv:2507.14805), we argue the damage travels through a **latent, sequence-level channel ("subliminal learning")** that semantic filters cannot see by construction. Our fix is a **Subliminal Filter**: a gradient-alignment test in the student's LoRA space that blocks drift-inducing trajectories before training, used *alongside* the semantic filter.

> Tagline: **Training on safe data leads to increased unsafety.**

---

## Why the Last Paper Was Rejected → What Changes

| Old flaw | New plan |
|---|---|
| Base LLaMA/Qwen checkpoints (unaligned can't "forget") | `Llama-3-8B-Instruct` + `Qwen2.5-7B-Instruct` only; fresh baselines |
| Toxicity filter never fired on math (dead code) | Becomes the **motivating finding**: 0 semantic rejections + rising ASR = filters are blind |
| EWC criticized but never run (+ SEAL miscited as using EWC) | True EWC baseline (Fisher on safety data); SEAL cited only for bilevel data selection |
| Replay buffer ≈ replication of Bianchi et al. 2023 | Bianchi cited for single-shot static tuning; our delta = multi-round loop + latent filter |
| Single seed, inconsistent round-0s, pooled XSTest | ≥3 seeds mean±std; one frozen greedy round-0 shared by all tables; XSTest split safe/unsafe; judged ASR |

---

## What's Novel

1. First study of subliminal safety erosion in an **iterated, open-weight STaR loop** (Cloud et al.: single distillation event only).
2. **Mediation experiment** separating true latent transmission from ordinary fine-tuning forgetting (teacher-swap: self vs cross-base-model vs human-gold traces).
3. **Subliminal Filter** — first defense operating below the token level (gradient-alignment + refusal-direction telemetry), where Cloud et al. explicitly leave filtering open.

---

## Plan at a Glance

| Phase | When | Compute | Deliverable / gate |
|---|---|---|---|
| **Sprint D1–D4** | next 4 days | RTX 4050 (6 GB) | All science modules built + unit-tested on tiny models; end-to-end smoke test; DGX launch kit pushed to GitHub, tag `dgx-ready` (see `docs/SPRINT_local_rtx4050_4days.md`) |
| **M1** Pilot | Day 5–7 (on cluster arrival) | 8×A100 | Round-0 frozen baselines; naive STaR × 10 rounds w/ telemetry → **Gate G1: does Instruct-checkpoint drift exceed seed noise?** |
| **M2** Main grid | wk 2–3 | 8×A100 | 2 models × 4 conditions (naive / +semantic / +semantic+subliminal / +replay) × 3 seeds + teacher-swap mediation |
| **M3** Baselines & stats | wk 3–4 | 8×A100 | True EWC, Bianchi-static control, threshold ablations, multi-seed stats |
| **M4** Write-up | wk 4–5 | local | ICLR manuscript + self-audit |

**Compute:** RTX 4050 available **only Days 1–4** — used for code, unit tests, and a GPT-2-scale smoke run (no 7–8B work fits in 6 GB). All experiments (~15–25 node-hours per 10-round run; grid ≈ 24 runs) on the 8×A100 DGX from Day 5. The sprint exists precisely so the cluster starts producing results on day one.
**Fallback:** if M2 slips past ~1 week → ICML 2027 (late-Jan) with the fuller program.

---

## Decisions Needed From You (recommendations in brackets)

1. Venue: attempt ICLR 2027 MVP now, or target ICML 2027 directly? *[build MVP, decide at M2]*
2. Include the misaligned-teacher arm (insecure-code FT teacher generating benign correct math traces; internal-only, no harmful outputs released)? *[yes — strongest mechanism evidence]*
3. Headline filter = gradient-alignment in LoRA space, with refusal-direction probe as cheap pre-gate? *[yes]*
4. LoRA everywhere (one full-FT sanity arm if budget)? *[yes]*
5. Seeds: 3 on headline conditions, 1 on ablations (stated in paper)? *[yes]*
6. Eval depth: full benchmarks at rounds {0,5,10}, fixed n≈300 probe between? *[yes]*

---

## Where This Can Go (beyond the single paper)

**Scientific upside if the mechanism holds (Gate G1 / E3):**
- A general law worth stating: *any iterated accept-filtered self-distillation loop monotonically amplifies whatever its filter doesn't protect* — safety is just the first victim; factuality, calibration, refusal-style all follow. One clean result opens 3–4 follow-up papers (other traits, other loop types, theory extension of Cloud et al.'s Thm. 1 to iterated acceptance sets).
- Bridges two literatures that don't currently talk (continual learning ↔ subliminal learning) — good citation gravity.

**Artifacts & community value:**
- `SubliminalFilter` as an open-source, plug-in gate for any generate→filter→train pipeline (usable by anyone running STaR/ReST/self-rewarding loops).
- **SubliminalDrift benchmark**: versioned probe sets + telemetry protocol (refusal-direction tracking) for auditing *any* post-training loop — fills a real gap: today's audits are all token-level.
- Judged-ASR eval suite with split XSTest as a reusable, correctly-designed safety eval.

**Practical / industry relevance:**
- Every lab doing multi-round post-training (RLHF iterations, self-improvement, distillation from stronger teachers) faces exactly this blind spot; a latent early-warning metric (SDS leading ASR by several rounds) is directly deployable as a training-time safety monitor.
- Supply-chain angle: companies fine-tuning on *other labs'* distilled data inherit traits invisibly — our cross-base teacher-swap result quantifies that risk.

**Scaling path (later, compute permitting):** 70B-class models (full-FT vs LoRA drift comparison), 50+ round horizons, agentic loops (tool use trajectories instead of math CoT), and interaction with active alignment refreshes (does periodic DPO re-anchoring substitute for filtering?).

---
*Nothing here repeats the old mistakes: every claim is bound to the experiment that earns it (claims ledger in the full doc). Code and manuscript start only after sign-off.*
