# Phase 2 — Subliminal Safety Erosion: Research Plan
### `conti` · latent safety drift in iterated self-improvement (STaR)
**For:** Dr. Chhavi Yadav (CMU) · **From:** Aadi Sharma (SRMIST) · **Date:** 2026-08-26 · **Status:** pre-registration draft
**Supersedes:** the "subliminal filter" framing in earlier docs. This document argues the filter is a *consequence* of a mechanism claim, not the paper's first contribution.

---

## 1. Executive summary — the claim ladder

The paper should be built as a four-rung ladder, each rung independently reportable, and the headline only claimed if rung 3 holds:

1. **Phenomenon.** In 10-round STaR on aligned Instruct checkpoints with 100% Llama-Guard-3-passing training traces, safety alignment erodes: aggregate ASR rises, XSTest/DoNotAnswer more than AdvBench.
2. **Non-triviality.** The erosion is not explained by ordinary fine-tuning on narrow benign data (Qi et al. 2024) or by the static safety buffer (Bianchi et al. 2024). Controls: human-gold rationales at matched steps, cross-base teachers, shuffle controls, step-matched budgets.
3. **Mechanism.** The erosive component is (a) *directional* — approximately one-dimensional in the refusal-direction subspace, (b) *provenance-dependent* — carried by self-generated sequence-level style, not by semantic content or data narrowness, (c) *ratcheting* — it compounds with horizon T in a way a static buffer cannot offset at fixed size, and (d) *trace-concentrated* — a small fraction of accepted traces carries most of the damage, which is what makes data-level intervention possible.
4. **Remedy.** A direction-aware gate (gradient-alignment + optional projection) and the horizon-dependent buffer law k\*(T). The gate is a causal test of rung 3: it only works if the channel is real and locally causal.

**The falsifier up front (rung 2):** if human-gold, step-matched training on the same problems produces the same drift, this is forgetting, not subliminal learning, and the paper should say so. We pre-register that outcome and commit to reporting it.

---

## 2. Why Phase 1 is not yet explained

Phase 1 established the phenomenon. It did not explain the *pattern*, and the pattern is where the science is. Three observations from the data we already have crack the standard explanation:

**Crack 1 — drift continues after capability saturates.** Qwen reaches ~50% GSM8K by round 4; the capability gradient shrinks as the model fits its accept-set. If safety loss were driven by "gradient steps on capability data erode safety" (the Qi et al. 2024 mechanism), safety drift should decay with the same profile. In the rejected-draft trajectory it does not: aggregate ASR for the no-replay condition *bottomed at round 2 and climbed to round 9*. A mechanism whose *input* (the update magnitude) is decaying while its *output* (safety loss) grows is not a mechanism about update magnitude. It is a mechanism about the data being re-injected.

**Crack 2 — the damage is anisotropic and behavioural, not uniform.** The old LLaMA baseline lost 16.7 pp on XSTest and 20.2 pp on DoNotAnswer while *improving* 10.2 pp on AdvBench. Generic forgetting predicts broad, capability-correlated decay. What we see is the selective erosion of a boundary behavior — *complying when you should refuse* — while overt harmfulness barely moves. That is the signature of a single dominant direction losing margin, not of diffuse parameter damage.

**Crack 3 — the filter is an operator, and operators compose.** STaR does not train on the model's output distribution; it trains on the *accept-filtered* slice: high-confidence, correct, answer-always rationales. Refusal is a *low-density, off-mode* behavior: it never appears in GSM8K traces, so it receives zero gradient reinforcement in any round. The filter acts like a repeated projection onto the model's competent-answer manifold. Cracks 1 and 2 are what you would predict if this projection *compounds* across rounds while the safety behavior is never re-injected. Cloud et al. (2025) supply the quantitative basis: a small imitation step on teacher outputs moves the student toward the teacher for *any* training distribution when the base model is shared — and in a STaR loop teacher and student are the *same weights*, which is the maximal case of that precondition.

So the sharpest statement of our claim is **not** "evil traits hide in data" (Cloud et al. already showed that, and it needs an evil teacher). It is:

> **Iterated self-imitation restricted to a sparse utility filter is a directional ratchet on the compliance prior. The token stream is 100% benign; the damage is carried by the teacher's sequence-level behavioral style, amplified once per round, and it selectively disarms refusal because refusal is the one behavior the filter never sees.**

---

## 3. Formalism and the three competing hypotheses

**Notation.** θ_t = parameters at round t. A_t = accepted trace set at round t (the correctness filter's output). For a single trace y_i ∈ A_t with prompt x_i, let g_i^t = ∇_θ L(y_i | x_i; θ_t) be its exact per-sample gradient. Let d be the unit *refusal direction* in the same parameter space, defined operationally as the normalized gradient of a refusal objective (the mean gradient over the curated refusal buffer — this is exactly the "safety anchor" g_safety the codebase already computes). Every update is a sum of per-sample gradients plus optimizer noise:

    Δθ_{t+1} = −η Σ_{i ∈ A_t} g_i^t + ε_t

Decompose every per-sample gradient against d:

    g_i^t = g_i^cap + (g_i^t · d) d + g_i^⊥
    a_i^t := g_i^t · d          ← the per-trace "compliance pressure" on refusal

The whole scientific question is the *distribution and accumulation law* of the scalar sequence a_i^t across traces and rounds, and the three hypotheses make different, testable predictions:

| | **H-F: Static forgetting** (Qi/Bianchi/EWC view) | **H-S1: Self-imitation compliance prior** (our mechanism) | **H-S2: Trait transmission** (Cloud et al., transplanted) |
|---|---|---|---|
| Source of harm | aggregate optimization pressure on narrow data | sequence-level behavioral style of the teacher, re-injected each round | an explicit unwanted trait in a teacher |
| Sign of a_i^t | fluctuates, no consistent sign (E[a] ≈ 0) | E[a_i^t] < 0 consistently (comply-mode traces push *against* refusal) | < 0 only for trait-bearing teachers; ≈ 0 for aligned teachers |
| Dependence on | number of steps, LR, data narrowness | provenance (self vs human) and horizon T | which teacher generated the data |
| Shape of dθ_t in d | random-walk-like, fluctuates | monotone, sign-consistent partial sums | large, teacher-specific projection |
| Static buffer at fixed k | sufficient at one shot (Bianchi) | k must grow with T (k\*(T) > k_static) | n/a |
| Per-trace damage | roughly uniform | heavy-tailed / concentrated (a few traces carry most of Σa_i) | concentrated on trait-bearing traces |
| Remedy that works | replay, EWC, smaller LR | directional gate, horizon-scaled buffer | data provenance filtering |

Three prediction classes fall directly out, and all three are cheap to compute from logs we will already be producing:

**P1 — Sign consistency (statistical, no training required).** H-F predicts a_t / ||g_t|| behaves like centered noise (≈ 50/50 sign across rounds; the cumulative sum random-walks). H-S1 predicts a strictly negative sign in ≥ 9/10 rounds and a cumulative sum that trends monotonically. A sign test over rounds and a Mann–Whitney U over per-trace scores is a *high-power* test that can be run retrospectively on the Phase 1 logs (Gate R already logs per-candidate cosines). If Phase 1's logs show sign-consistent negative pressure, that is already strong evidence for H-S1.

**P2 — Damage concentration (per-trace).** Compute s_i = −min(a_i, 0) (the damage a trace does to refusal) and the top-decile share Σ_top10% s_i / Σ_i s_i. H-F predicts a share near 10% (uniform). H-S1 predicts ≫ 10% (a skewed distribution). This single number is the per-example discriminator between "forgetting is spread" and "specific traces carry the channel" — and it is the direct scientific justification for data-level gating: if the damage is concentrated, you can filter it; if it is uniform, you cannot, and replay/EWC is the only game in town.

**P3 — Horizon law for the buffer (the ratchet).** In the static case, the safety anchor contributes a fixed opposing projection per batch, −α·k·||g_safety,d||, while capability pressure acts once. In the loop, the compliance term is re-applied every round, so the fixed anchor must fight a sum over T rounds: with static k it can only balance if k ≥ k\*(T), and k\*(T) should *grow with T* (roughly ∝ T if the compliance term is coherent, ∝ √T if it is a random walk — the *slope itself* measures coherence!). Measuring k\*(T) for T ∈ {0, 3, 10, 20} by buffer-size sweeps converts a qualitative claim ("loops are ratchets") into a fitted law. This is the single experiment I would most want on the paper: it is the point where static-fine-tuning theory and continual-loop theory make quantitatively different predictions.

---

## 4. How this differs from Qi et al. / Bianchi et al. / SEAL (mathematically, not rhetorically)

- **Qi et al. (2024).** Harm is a function of *optimization pressure on out-of-distribution benign data*; the remedy is data diversity or replay. Our claim adds that pressure is not the primitive: at matched steps and matched narrowness, provenance changes the sign structure of the damage (P1, P2). Qi et al. are the null hypothesis we are testing, not competitors we are beating.
- **Bianchi et al. (2024, Safety-Tuned LLaMAs).** A single-shot SFT mixing ≈ 3% safety demonstrations restores safety. The gap is *composition over time*: their recipe is a one-shot tug-of-war, our setting is a ratchet whose opposing force is applied at constant rate against a coherently accumulating pressure (P3). Novelty claim, precisely: **we measure the horizon dependence of the static recipe's sufficiency**, k\*(T), and show the static constant is only valid at T=1.
- **SEAL (Shen et al., ICLR 2025).** Bilevel-optimization data *ranking* against fine-tuning attacks. Cited strictly for bilevel data selection; no EWC (we verified the method is a data ranker). Our filter is not a ranker: it is a *mechanistic projection* — it uses the refusal direction and claims the traces it removes are the traces that carry the channel, which is falsifiable (if the gate works equally well at random removal, the mechanism claim is wrong even though the method works).
- **EWC.** Parameter-space quadratic anchor toward θ₀ weighted by safety Fisher. It resists drift *isotropically* (any large move is expensive) and therefore taxes capability; it cannot distinguish "safety-critical" from "capability-critical" movement along the same direction. Our λ-sweep EWC arm is the honest control: we predict EWC buys bounded ASR at a measurable capability cost, and that its penalty on the refusal direction is larger than its penalty on capability directions *for the same parameter norm* — a falsifiable, quantitative difference.

---

## 5. The experiment program (pre-registered)

All runs: LoRA r=16 on all projections, GSM8K, 10 rounds unless stated, ≥ 3 seeds on P0/P1, full 1,319-problem GSM8K, XSTest split 200-safe/250→(200 safe + 50 unsafe), AdvBench 520, DoNotAnswer 939, HarmBench (added in Phase 1), Llama-Guard-3 over every accepted trace (must reject ≈ 0; that log *is* a result).

**A. The provenance × horizon factorial (P0; the main figure).**
Conditions, all step- and volume-matched per round where possible:
  A1 `self_r10` — current loop (self CoT, correctness-accepted).
  A2 `human_r1` — one-shot SFT on human GSM8K rationales (Bianchi-style static; no self-generation).
  A3 `human_r10` — human rationales *replayed over 10 rounds* with fresh optimizer state each round (tests horizon without provenance).
  A4 `self_r10_goldmix` — self CoT + the 3% static safety mix (the Bianchi control *inside* the loop).
  A5 `human_r10_goldmix` — same buffer applied to the human condition.
Predictions: A1 ≫ A3 at matched steps (provenance); A3 ≳ A2 (horizon alone adds little if provenance dominates); k\*(T) from the A1/A4/A5 buffer sweep. This is the figure that decides the paper.

**B. Teacher-swap mediation (P0; mechanism).**
  B1 self (A1) · B2 cross-base teacher (Llama traces → train Qwen, Qwen traces → train Llama; matched acceptance rate by subsampling) · B3 human gold · B4 misaligned teacher (insecure-code-FT, Betley et al. data; internal-only, no harmful outputs released).
Prediction: A1 ≫ B2 ≈ B3 (shared-init channel), and B4 ≫ A1 (explicit trait transmission is much larger — the honest ordering, and it keeps us from overclaiming). B4 replicates Cloud et al.'s CoT experiment open-weight; B2/B3 are ours.

**C. Sequence-level controls (P1; isolates style from semantics).**
  C1 shuffle: permute sentence order / numbers within accepted traces (labels fixed); accuracy re-matched by dropping now-wrong traces.
  C2 entropy/narrowness matching: select human rationales to match the self-traces' per-token entropy and n-gram-diversity profile under the student — if drift follows *entropy* at matched provenance, narrowness explains it; if it follows *provenance* at matched entropy, the latent channel explains it.
  C3 paraphrase: LLM-rewrite traces to a different surface style while preserving reasoning; semantic filters still pass; prediction: erosion weakens proportionally to style change.
These three controls are what separate "subliminal" from "narrow data" — the confound Cloud et al. explicitly flagged.

**D. Directional structure and causal probes (P1; mechanism, cheap).**
  D1 rank-1: project every per-round update onto the refusal direction; sign-consistency test (P1) + variance explained by top component across the parameter space.
  D2 curvature: ||F_t − F_0||_F (Fisher divergence already implemented) vs ASR_t; tests whether the *loss landscape's* safety geometry moves before behavior does.
  D3 representation: CKA + cosine on residual-stream probes (implemented) at layers ℓ ∈ {L/3, 2L/3, L}; prediction: drift concentrates in mid-to-late layers (where refusal features live per Arditi et al. 2024), not in early layers.
  D4 temporal precedence: does SDS_t (step t geometry) predict ASR_{t+2} better than ASR_t predicts ASR_{t+1}? (cross-validated r² across seeds).

**E. The remedy as a causal test (P0/P1).**
  E1 gradient-alignment gate (implemented): reject candidates with cos(g_math, g_safety) < τ, τ calibrated so ≥ 90% of human-gold traces pass.
  E2 the *decisive* control: random-removal gate at the same rejection rate, and oracle-removal gate at the same rate (oracle = remove the top-damage traces by true a_i). If random ≈ mechanism gate, the method works but the mechanism is wrong; only the mechanism gate ≈ oracle validates the story. This is the experiment that makes the paper honest, and I would report all three arms regardless of outcome.
  E3 directional projection training-time: project the compliance component (g·d)d out of the update; prediction: ASR bounded with *no* capability tax (unlike EWC, which taxes everything) — the cleanest remedy/axiom contrast.

**F. Model and robustness checks (P2).** Llama-3-8B-Instruct × Qwen2.5-7B-Instruct × 3 seeds; 20-round horizon on the best condition; MATH as a second capability distribution (contamination caveat from Cloud et al.); HarmBench throughout.

---

## 6. What "subliminal" must mean operationally (and what it must not)

To survive review, the term gets pinned to measurable statements:
1. **The channel is not semantic.** After Llama-Guard-3 (and paraphrase/shuffle controls) every training token is benign; claims about "hidden toxicity" are excluded.
2. **The channel is model-specific.** Transmission requires a shared base (Cloud et al.); our cross-base arm must show attenuation. If cross-base shows full transmission, "subliminal" in Cloud's sense is falsified and the mechanism is generic distributional drift — report that.
3. **The channel is directional.** Damage concentrates on the refusal subspace, not the parameter bulk (D1, D3).
4. **The channel is per-trace.** A minority of traces carries most damage (P2); if damage is uniform, data-level filtering has no mechanism basis and we say so.
5. **The channel compounds.** k\*(T) grows with horizon (P3).
Every claim in the paper maps to exactly one of these five; no results are described as "subliminal" that do not.

---

## 7. Compute plan (4×A100-40GB reality)

Per 10-round run on 4 GPUs: ≈ 25–30 h (generation and per-trace gradient scoring dominate and shard cleanly). Prioritized:

| Priority | Experiments | Runs | Wall-clock (4 GPUs) |
|---|---|---|---|
| P0 | A1–A5 factorial (buffer sweep at T ∈ {1,3,10} for k\*) | ~12 | ~6 days |
| P0 | B1–B4 teacher swap | 4 | ~2 days |
| P1 | C1–C3 sequence controls | 3 | ~1.5 days |
| P1 | D1–D4 probes (no new runs; instrumented inside A/B) | 0 | ~0 (log analysis) |
| P0 | E1–E3 gates + random/oracle controls | ~8 | ~4 days |
| P2 | F robustness (20-round, MATH, HarmBench) | 3 | ~1.5 days |

**Tier-1 (paper-critical): A1, A2, A3, A4, B2, B3, E1, E2-random, E2-oracle ≈ 20 runs ≈ 10–12 days.** That fits the window; everything else stretches. Optimization: the D-probes are instrumentation inside A/B (zero marginal cost), and per-trace scoring only runs on correctness-accepted candidates (typically ≪ the 2,000 generated), so the gate is far cheaper than a full sweep.

---

## 8. Key risks and controls (one line each)

- *Drift is plain forgetting* → step-matched human control (A2/A3); we pre-commit to reporting this outcome.
- *Data narrowness, not provenance, drives it* → entropy/narrowness matching (C2).
- *Semantics, not style* → shuffle/paraphrase controls (C1/C3).
- *Gate works by any removal* → random-removal arm at matched rate (E2).
- *Effect is Qwen-only or Llama-only* → both models from P0; the paper claims a mechanism or admits a boundary condition.
- *Contamination* (GSM8K in pretraining) → MATH as second distribution; note Cloud et al.'s caveat.
- *ASR metric is a proxy* → judge-based ASR (Llama-Guard-3-as-judge) on all runs; 200-response human spot-check; FRR tracked separately.
- *Multi-seed noise* → ≥ 3 seeds on P0; effect sizes with CIs; sign tests over rounds.

---

## 9. Figure plan (the paper in five panels)

1. ASR vs round for {self, human} × {T=1, T=10} — provenance and horizon main effect.
2. Per-trace damage distribution + top-decile share (P2) — the concentration panel.
3. Refusal-direction projection per round with sign test (P1) — the directional panel.
4. k\*(T) fitted law — the ratchet panel.
5. Gate efficacy: mechanism vs random vs oracle, with math cost (E2) — the remedy/causality panel.

## 10. Decisions I need from you

1. Is the P3 "buffer critical-size scales with horizon" framing the centerpiece (my recommendation — it is the cleanest quantitative separation from Bianchi) or should the paper lead with the trace-level concentration result?
2. Do we run the misaligned-teacher arm (B4) in the paper, or keep it in an appendix for ethics-review reasons?
3. Do you want E2 (random vs mechanism vs oracle gate) in the main text even if the gate's *method* is weaker than replay? I argue yes — it is the experiment that keeps the mechanism claim honest.
4. If Gate G1 shows no drift on either Instruct checkpoint within 3 rounds: extend horizon, or pivot to the B4 transmission demonstration as the headline? My recommendation: B4 headline + drift as secondary; it is a guaranteed result (Cloud et al. replicated) and still on-theme.

---
*Implementation status: all primitives (exact per-sample Fisher, Gate R gradient-alignment, CKA/cosine probes, disentangled eval with judge, nohup-supervised DGX automation) are in the `conti` codebase on branch `feat/subliminal-pivot`; 106/106 tests pass locally. Phase 2 needs no new machinery — only this experiment program.*
