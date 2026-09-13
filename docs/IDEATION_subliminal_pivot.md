# Ideation Document — Pivoting `conti` to Subliminal Safety Erosion in Self-Improving LLMs

**Status:** v1.0 (for instructor / team sign-off before any code or manuscript is written)
**Date:** 2026-08-21
**Inputs:** (a) rejected COLM 2026 workshop draft *"Self-Taught Reasoners Forget How to Be Safe"*; (b) instructor notes on Subliminal Learning; (c) reviewer constraints (Section I of the task brief); (d) Cloud et al. 2025 (arXiv:2507.14805) full text; (e) repo state (`Aadisharma1/Conti`) and `context.md`.

---

## 0. Executive Summary

The rejected paper died of five wounds: base (unaligned) checkpoints, a "dead code" toxicity filter that rejected **0 of ~40K** trajectories, a ghost EWC baseline that was criticized but never run, a miscitation of SEAL, and single-seed point estimates. The instructor's pivot tells us what the *next* paper is about:

> **Training on safe data leads to increased unsafety — because the signal that erodes safety lives below the token level, where semantic safety filters cannot see it. Remedy: a Subliminal Filter that reads model internals (activations / gradients), combined with a semantic filter.**

This doc argues the pivot is scientifically strong *and* that our existing infrastructure is nearly the perfect testbed for it — but only if we are honest about one central tension the reviewers of paper #2 will immediately spot:

> **A STaR loop is self-distillation (teacher = student). Cloud et al.'s theory says the student moves *toward* the teacher. If the teacher is the aligned model itself, why would safety erode — and what exactly is "subliminal" about it?**

Resolving that tension with a falsifiable mechanism story plus a mediation experiment (**E3: teacher-swap**, §6.3) is the difference between a top-tier paper and a second rejection. The plan below therefore has two pillars: (1) **prove the mechanism** (or honestly fail and reframe), and (2) **engineer the fix** (the Subliminal Filter) with real baselines (true EWC, continual vs. single-shot replay à la Bianchi et al. 2023) on aligned Instruct checkpoints, multi-seed, with a bulletproof evaluation protocol.

**Recommendation in one line:** 10-round STaR loops on `Meta-Llama-3-8B-Instruct` + `Qwen2.5-7B-Instruct`; Llama-Guard-3 as the semantic filter (shown blind); gradient-alignment filter in LoRA space as the headline Subliminal Filter; teacher-swap mediation as the mechanism evidence; true EWC + Bianchi-style static replay as baselines; ≥3 seeds on headline conditions.

---

## 1. Post-Mortem of the Rejected Paper (what exactly killed it)

| # | Flaw | Evidence in the rejected draft | Fix in the pivot |
|---|------|-------------------------------|------------------|
| F1 | **Unaligned base model** | Paper used base `meta-llama/Meta-Llama-3-8B` (3.6% GSM8K baseline). Worse: `context.md` shows configs ran `Qwen/Qwen2.5-7B` — the *base* Qwen, **not** Instruct — while the paper's §5.1 claims the instruction-tuned Qwen was used. This is an integrity landmine independent of reviewer taste. | Both models swapped to `-Instruct`; all baselines re-established from scratch; round-0 evaluated once and reused everywhere. |
| F2 | **Dead-code filter** | §5.1 admits Phase 1 ≡ Naive STaR ("near-zero rejection rate"); Table 4 shows 0 safety rejections everywhere; limitations §6.1 concedes the verifier "never fired." | No longer a bug — it becomes **motivating evidence**: 0/40K semantic rejections while ASR drifts +11.1 pp is exactly the "filters are blind" phenomenon. But we must now run a *real* semantic filter (Llama-Guard-3) and still show blindness (E2). |
| F3 | **Ghost EWC baseline** | §6 claims "data-level interventions outperform parameter-level constraints such as EWC" and "EWC-augmented training caused false-refusal rates to spike" — **no EWC run exists in the paper** (the repo's `ewc.py` even has a TODO: "actually compute fisher on safety data"). | Run true EWC with diagonal Fisher computed over *safety* data, λ sweep, report both safety drift AND over-refusal. Only then may we compare. |
| F4 | **SEAL miscitation** | Draft claims SEAL (Shen et al., 2025) "proposed bilevel data selection with EWC constraints" and "reported 5–15% ASR increases over 2–3 rounds of self-improvement." Verified: SEAL (arXiv:2410.07471, ICLR 2025) is a **bilevel-optimization data ranker** — no EWC, and it is about defense against fine-tuning attacks, not self-improvement rounds. Both claims wrong. | Cite SEAL **only** for bilevel data selection. The "5–15% over rounds" claim is dropped (it was apparently our own result, mis-attributed). |
| F5 | **Novelty: replay buffer ≈ Bianchi et al. (2023)** | k=4 safety samples (<1%) per batch in a single training pipeline reads as a replication of Safety-Tuned LLaMAs (Bianchi et al., arXiv:2309.07875, ICLR 2024: ~3% safety examples mixed into one static instruction-tuning run restores safety). | Position Bianchi et al. as **single-shot, static** tuning. Our delta: (i) **multi-round** self-improvement where the data distribution shifts every round; (ii) **latent-level filtering**; (iii) explicit study of *when static replay stops being enough* (rounds 10+, drift re-accelerates?). |
| F6 | **Analytics sloppiness** | Single seed (42); Table 1 "baseline" ≠ Table 5/6 round-0 rows (e.g., Qwen AdvBench 35.2 vs 25.8/23.5) — because round-0 was re-evaluated per condition with sampled generation; XSTest described as "n=450, 200 unsafe + 250 safe" (XSTest is 250 prompts: 200 safe / 50 unsafe — the loader or the text is wrong); ASR uses a `|response| > τ_len` length heuristic (long benign answers counted as attacks). | ≥3 seeds, mean ± std; one frozen greedy round-0 eval per model shared across all tables; re-verify XSTest loader; replace length-heuristic ASR with Llama-Guard-3-as-judge (+ report plain refusal rate separately). |

**Take-away:** F1–F6 are all fixable, and F2 flips from weakness to motivation — but only if the new paper *earns* the subliminal claim rather than asserting it (§4–§6).

---

## 2. The New Thesis

**Tagline (instructor's):** *Training on safe data leads to increased unsafety.*

**Working title candidates:**
1. *Safe Data, Unsafe Model: Subliminal Safety Erosion in Self-Improving Language Models*
2. *Below the Token: Why Safety Filters Cannot Stop Alignment Drift in STaR Loops — and What Can*
3. *Subliminal Safety Erosion: Latent Transmission of Compliance in Iterated Self-Distillation*

**One-paragraph thesis.** Modern self-improvement pipelines (STaR and descendants) fine-tune an aligned LLM on its own, correctness-filtered reasoning traces. These traces are textually benign — they pass state-of-the-art semantic safety filters essentially always. We hypothesize that the erosion of safety alignment observed over rounds is carried not by semantic content but by **latent, sequence-level statistical structure** of self-generated text — the same channel that Cloud et al. (2025) call *subliminal learning*. Because the channel is below the token level, semantic filters (keyword lists, toxicity classifiers, Llama-Guard-3, even LLM judges — Cloud et al. show judge ROC ≈ chance on this signal) are *structurally blind* to it. We (i) reproduce safety erosion on aligned Instruct checkpoints under 10-round STaR with a best-in-class semantic filter active, (ii) provide mechanism evidence distinguishing the subliminal channel from generic fine-tuning forgetting (teacher-swap mediation), (iii) introduce a **Subliminal Filter** — a gradient-alignment test in the student's parameter space — and (iv) show that semantic + subliminal filtering, optionally anchored by a minimal safety replay buffer, bounds drift across rounds where single-shot recipes (Bianchi et al. 2023) and parameter-level consolidation (EWC) each fail in characteristic ways.

---

## 3. What Cloud et al. (2025) Actually Establishes (and what it doesn't)

Citation: **Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans. "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data." arXiv:2507.14805 (2025).** Affiliations include Truthful AI, the Anthropic Fellows Program, Anthropic, ARC, Warsaw UTech, UC Berkeley. (Note: it is *not* purely "an Anthropic paper" — cite as Cloud et al.; the task brief's "Anthropic, 2507.14805" phrasing should not leak into the manuscript.)

Established by them (verified from full text):
- **Traits transmit through semantically unrelated data.** Teachers with a trait T (liking owls; being misaligned) generate number sequences / code / CoT; students finetuned on them acquire T. Misalignment transmits: e.g., teacher's misalignment 79.6% → student ~9–16% from *number sequences alone* (34 banned numbers filtered out), student 0% → ~10% misaligned free-form responses; TruthfulQA MC1/2 both degrade.
- **CoT transmission is the result most relevant to STaR.** An insecure-code finetuned (misaligned) teacher's GSM8K CoT → student misalignment rises 0% → 8%, at essentially unchanged GSM8K accuracy. This is *our training distribution*.
- **Filtering fails, even with strong judges.** GPT-4.1-mini as an alignment judge on CoT: ROC ≈ random; LLM filtering could not block transmission. Number-format filters likewise failed. This is the instructor's leg (ii): filters let the responses pass.
- **Shared initialization is required.** GPT-4.1 ↔ GPT-4o (same init) transmit; different base families do not. **A STaR loop is the maximal case: teacher and student are the *same weights*.**
- **Theory (their Thm. 1):** for any student training distribution, a small SFT step on teacher outputs yields parameter movement with non-negative dot product with the teacher's own finetune direction: `Δθ_S · Δθ_T ≥ 0` (under shared init, moderate teacher steps). Subliminal learning is generic — they even reproduce it in a 2-layer MLP classifier.
- **Sequence-level structure matters (their shuffle experiment):** shuffling numbers within/across responses eliminates most transmission → the carrier is *sequence-level statistical structure*, not tokens in isolation. Also: does **not** transmit via ICL; prompted classifiers cannot detect the signal (≈ chance); transmission is independent of teacher's task performance; MMLU cost ~2–4 pp.
- **Their stated limitations (our openings):** artificial trait setups; single distillation event (10 epochs), *not iterated*; GPT-4.1 may have seen GSM8K in pretraining; open question what real-world traits/signals transmit.

**What it does NOT establish (we must not inherit overclaims):**
1. No *iterated* loops — nobody has shown the effect compounds over generation→filter→train rounds (our setting).
2. No *aligned-teacher* erosion story — their teachers are given a trait; our teacher is the aligned model itself (§4).
3. No open-weight replication at 7–8B scale with standard alignment suites (theirs: GPT-4.1/GPT-4o mini + open-weight number-sequence side experiments).
4. No *defense* — they explicitly say semantic filtering may be insufficient "even in principle." Nobody has built a filter that reads the latent channel. **That is our contribution.**

---

## 4. The Central Intellectual Problem: Why Does Self-Distillation Erode Safety?

Cloud et al.'s theory says the student moves *toward the teacher*. In STaR, teacher = self, so naively the theory predicts alignment is *preserved*. Yet we (and Qi et al. 2024, and the fine-tuning-attacks literature) observe erosion. Three candidate channels, which our experiments must **disentangle**:

- **C1 — Generic fine-tuning forgetting (the "boring" channel).** *Any* narrow-distribution SFT erodes safety behaviors that have no gradient support in the new data (Qi et al. 2024 showed benign data suffices). Teacher-agnostic, data-content-agnostic, driven purely by distribution shift + LR/steps.
- **C2 — Acceptance-set narrowing / persona drift (our mechanistic proposal).** The STaR loop does not train on the model's *full* output distribution — it trains on the **correctness-filtered slice**: confident, compliant, answer-always reasoning traces. Iterated SFT on this slice pulls the model's global behavior toward a "comply-and-solve" persona. Refusal on borderline queries is the behavioral opposite of that persona, and it is *never reinforced*. Under their Thm. 1 applied iteratively (teacher at round r is the round r−1 student), each round's drift is preserved and *compounded* — nothing in the loop ever restores directions lost by the acceptance filter. This is a *self*-subliminal channel: the trait transmitted is the teacher's reasoning style, which is *negatively correlated with refusal behavior* even though every token is benign. Predicts: drift concentrated on borderline-compliance benchmarks (DoNotAnswer, XSTest-safe-compliance) rather than on overt harm (AdvBench) — **exactly the pattern in our rejected data** (AdvBench *improved* −10.2 pp while XSTest +16.7 / DoNotAnswer +20.2 on LLaMA P1).
- **C3 — Cross-generation trait transmission (their setting, transplanted).** If any upstream teacher with unwanted traits contributed data (other models' distillates in the pretraining/eval stack, or in future multi-model self-improvement), traits flow through benign traces. In our controlled experiments we can *create* this arm deliberately (misaligned teacher, E3c) to bound how bad the channel can get.

**Discriminating predictions (this is what makes it science):**
| Experiment (see §6) | If C1 (forgetting) | If C2/C3 (subliminal) |
|---|---|---|
| E3a teacher-swap: train on **self-generated** vs **cross-base-model-generated** vs **human-gold** GSM8K traces, matched volume/accuracy/acceptance-rate | Drift ≈ equal across arms (it's just distribution shift) | Self/same-base ≫ cross-base ≈ human-gold (shared-init requirement from Cloud et al.) |
| E3b dose-response: fraction of self-generated vs human data at fixed total volume | Drift ∝ volume only | Drift superlinear in self-fraction |
| E3c misaligned teacher: benign, correct traces from an insecure-code-FT teacher (Betley et al. 2025 data) | No difference vs aligned teacher | Student ASR rises *more*, semantic filter still rejects ≈0 |
| E-telemetry: latent drift score (refusal-direction projection, §7.2) vs rounds | Tracks ASR with no lead | Rises *before* ASR (leading indicator), while ASR still flat |

If E3 comes back "all C1", the paper does not die — it reframes honestly: *"safety erosion in STaR is distribution-shift forgetting that semantic filters cannot see coming; we detect it in latent space rounds early and block it with a gradient-level filter."* That is still a strong paper. But we must run E3 before writing the mechanism section, and we must not use the word "subliminal learning" for anything E3 hasn't supported. **The claims ledger (§8) enforces this.**

---

## 5. Instructor Checklist → Workstreams

| Instructor to-do | Where it happens |
|---|---|
| 1. Find papers which use safety filters | §9 lit ledger: Llama-Guard-3 (Inan et al. 2023), toxicity classifiers in RLHF pipelines, STaR/V-STaR/ReST-EM correctness filters, SEAL's data screening, constitutional-AI-style filtering, distillation pipelines that screen teacher outputs. Deliverable: Related-Work §3.2 table "filters used, level (token/semantic), what they cannot see." |
| 2. Find papers showing safe data → unsafe behavior | Qi et al. 2024 (benign FT compromises safety); Shadow Alignment (Yang et al. 2023, ~100 examples); Betley et al. 2025 (insecure code → emergent misalignment); Cloud et al. 2025 (subliminal); self-consuming loops / model collapse (Alemohammad et al. 2023 — iterated self-training amplifies quirks); plus fine-tuning-attack lit (Zou et al. 2023 AdvBench, etc.). Deliverable: Related-Work §3.3 "safe-data erosion taxonomy" with the gap: none isolate the latent channel in *multi-round self-improvement*. |
| 3. Design a subliminal filter | §7 of this doc (design space, math, recommendation). |
| 4. Prove filters pass the harmful-but-benign-looking data | E2: run Llama-Guard-3 (and an LLM judge) over every accepted trajectory in the main runs; report rejection counts (prediction: ~0) *alongside* the latent drift telemetry (§7.2) climbing. Also cite Cloud et al.'s judge-ROC≈chase result and replicate cheaply on our traces. |

---

## 6. Experiment Program

All heavy runs on the 8×A100-40GB node; all loops LoRA-based (see §10 rationale). Local RTX 4050 only for unit tests / tiny-model instrumentation / plots (§10).

### E0 — Integrity & re-baselining (cheap, do first)
- Correct model IDs: `meta-llama/Meta-Llama-3-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (HF gated — tokens/licensing confirmed; Llama-Guard-3-8B also gated).
- One frozen, greedy round-0 evaluation per model (GSM8K + AdvBench + XSTest + DoNotAnswer + probe set), reused in every table of the paper.
- Fix XSTest loader (verify 200 safe / 50 unsafe; report separately — see §8), replace ASR length-heuristic with judge-based scoring (Llama-Guard-3-as-judge; spot-check 200 responses by hand).
- Clean repo: delete all `(1)` / `-Revenant` duplicate files; re-verify every metric against raw logs; rotate the exposed WandB key (noted in `context.md`).

### E1 — Main drift result (headline): does STaR erode safety on aligned Instruct checkpoints *with a real semantic filter active*?
- 10-round STaR, 2,000 candidates/round, GSM8K, per model.
- Conditions (the paper's backbone):
  1. **Naive STaR** (correctness filter only) — the pathology arm.
  2. **+ Semantic filter** (Llama-Guard-3 gates every accepted trace) — prediction: ≈0 rejections, drift ≈ Naive → "filters are blind" result.
  3. **+ Semantic + Subliminal filter** (§7) — headline fix.
  4. **+ Semantic + Subliminal + safety replay (k=4, <1%)** — full stack (this is also the Bianchi-delta condition: replay now sits inside a *multi-round* loop).
- ≥3 seeds for conditions 1–4 (both models). Report per-round: GSM8K pass@1, judged ASR per benchmark, refusal rate, false-refusal rate on XSTest-safe, latent drift score (§7.2), filter acceptance rates.

### E2 — Filter-blindness audit (mechanism, leg ii)
- Over every accepted trajectory of E1: Llama-Guard-3 verdicts + an independent LLM judge (e.g., GPT-4.1-class, mirroring Cloud et al.) — report rejection ROC-style: they should be unable to separate "drift-inducing" from benign traces. Label for the ROC comes from E4's gradient scores (can the semantic judge predict what the gradient filter catches? Prediction: no).

### E3 — Teacher-swap mediation (mechanism, leg i — **the make-or-break experiment**)
Single-generation distill arms (3 rounds each, 1 model initially — cheap):
- **E3a arms** (matched data volume, matched GSM8K acceptance-rate/accuracy by subsampling): (i) self-generated traces (student = teacher, same base); (ii) cross-base traces (Llama-Instruct traces → train Qwen-Instruct, and vice versa); (iii) human gold GSM8K solutions. Measure ASR + latent drift. C1-vs-C2 discrimination per §4 table.
- **E3b dose-response**: mix self:human at {100:0, 75:25, 50:50, 0:100}, fixed total steps.
- **E3c misaligned teacher** (only if instructor approves; data internal-only, no harmful outputs released): insecure-code FT teacher (Betley et al. 2025 public dataset) generates correct GSM8K traces → student ASR should rise beyond the self arm while semantic filters still pass everything. This is the open-weight, in-continual-loop replication of Cloud et al.'s CoT experiment — a very strong result if it lands.
- **E3d shuffle control** (cheap, decisive): within-trace token/number shuffling of training traces. If drift is C2 (sequence-level persona signal), shuffling should reduce latent drift at matched perplexity/loss.

### E4 — Subliminal filter calibration & ablations
- Threshold τ calibration: tune on validation so that ≥90% of *human gold* GSM8K traces pass (capability protection) while ranking E3c-teacher traces as unsafe (if E3c approved) or top-quantile drift-inducing traces from E1.
- Ablate: filter variant (gradient-alignment vs Fisher-overlap vs style-fingerprint), layer/parameter scope (LoRA adapters of which modules), threshold sweep, filtering vs down-weighting vs gradient projection.
- Overhead accounting: wallclock per round with/without filter (reviewers will ask).

### E5 — Baselines that were missing (ghost-baseline exorcism)
- **True EWC**: diagonal Fisher over the *safety* dataset (refusal examples + probe prompts), λ ∈ {50, 500, 5000}, fused into the same loop; report drift AND over-refusal (XSTest-safe FRR) AND math. Only vs. this run may we write a single comparative sentence about EWC.
- **Bianchi-static control**: single-shot safety mix (~3%) *without* per-round re-anchoring — tests whether the single-shot recipe survives 10 rounds of distribution shift (novelty delta for F5).
- (Optional) SEAL-style data ranker as an additional data-selection baseline — cite correctly, run only if time.

### E6 — Robustness
- Longer horizons on one condition (20 rounds) — does the full stack *keep* drift bounded or merely delay? (The rejected paper's own Fig. 2 showed non-monotone drift; round-9 rebound for P2 suggests "bounded" needs a longer horizon to claim.)
- Second capability dataset (e.g., MATH or code explanation) if time; greedy vs sampled eval agreement.

---

## 7. The Subliminal Filter — Design Space

Goal: score each *accepted* STaR trajectory y (already semantically clean, correctness-verified) by the latent safety effect of training on it; reject/downweight before it enters the round's training set. (Terminology note for the paper: our gate sits on the **self-generated pool each round**; the "safety replay buffer" is the separate, curated refusal set. The task brief's phrase "before they enter the replay buffer" should be re-worded in the manuscript to avoid conflating the two.)

### 7.1 Notation (to be defined identically in the paper — zero undefined symbols)
- θ: student parameters (LoRA adapters Δθ over projection modules; frozen base θ₀).
- L_math(D): SFT loss on the round's accepted math set D.
- B_safe: curated safety replay set (harmful prompt → refusal), |B_safe| fixed; k = 4 replay samples per training batch (<1%).
- g_safe = ∇_θ (1/|B_safe|) Σ L(b) : anchor gradient — the direction that *reinforces* refusal behavior.
- g_i = ∇_θ L(y_i): per-candidate gradient, computed in LoRA space (cheap: adapters only, r=32).
- α ≥ 0: replay loss weight; λ ≥ 0: EWC weight; τ: filter threshold — all calibrated on validation splits, reported with the calibration protocol (F4/F6 hygiene).

### 7.2 Telemetry first (you cannot filter what you don't measure)
Latent drift score per round r, all cheap on 8×A100:
- **SDS-act**: mean projection of hidden states (layer ℓ*, difference-in-means "refusal direction" extracted per Arditi et al. 2024 from harmful/benign probe pairs) over a fixed safety probe set — tracks alignment geometry directly.
- **SDS-grad**: fraction of accepted candidates with cos(g_i, g_safe) < 0 (gradient-conflict rate).
- **SDS-param**: cosine distance between current LoRA weights and round-0 snapshot.
- Hypothesis (publishable alone): SDS rises *rounds before* ASR does → early-warning justification for filtering at data level.

### 7.3 Filter variants
| Variant | Rule (accept y_i iff …) | Cost/round (2k cands) | Pros | Cons |
|---|---|---|---|---|
| **B. Gradient-alignment (headline)** | cos(g_i, g_safe) ≥ τ | ~2k LoRA-space backwards ≈ 30–40 min on 8×A100 | Mechanistic (directly reads the update's effect on safety); theoretically tied to Thm. 1 (reject candidates whose update direction is anti-aligned with the safety anchor); calibratable ROC | Most expensive; depends on B_safe quality |
| **C. Fisher-overlap** | Σ_j F_safe,j·g_ij² / ‖g_i‖² ≤ τ_F | Fisher once (reuse EWC machinery) + per-candidate weighted norms | Bridges to EWC story; cheap after Fisher | Coarser (parameter-importance, not direction) |
| **A. Activation-direction** | projection of y_i's mean hidden state on refusal direction ≤ τ_a | ~free (forward-only) | Cheapest; interpretable | Weakest causal link (a *benign* trace can sit low/high on the direction for content reasons) — use as telemetry, maybe pre-gate |
| **D. Style/sequence fingerprint** | divergence of sequence-level stats (token-entropy profile, n-gram repetition, formatting quirks; cf. their shuffle finding) from human-gold distribution ≤ τ_d | ~free | Model-agnostic; directly mirrors the demonstrated sequence-level carrier | Not safety-specific; over-filters strong-but-idiosyncratic solvers |

**Recommended composition (the paper's "Safety + Subliminal" filter):** semantic gate (Llama-Guard-3) → cheap pre-gate A/D → gradient gate B on survivors. Report each stage's rejection counts and false-positive rate on human gold GSM8K. Soft variant (down-weight 1[·] → w_i = sigmoid(β(cos(g_i,g_safe) − τ))) as an ablation — rejection isn't the only knob.

### 7.4 Combined training objective (per round, every symbol defined)
L(θ) = L_math(D_accept; θ) + α·L_safety(B_safe; θ) + (λ/2)·Σ_j F_safe,j (θ_j − θ*_j)²
with D_accept = semantically-and-subliminally filtered set; θ* = round-0 snapshot (EWC arm only); F from §E5. Headline method uses α with k=4 replay; λ=0 unless in EWC arm; τ, α, k, λ all listed in a hyperparameter table with calibration protocol. (The rejected draft's α was never defined — this table kills F6's "undefined variables" complaint.)

---

## 8. Evaluation & Analytics Hygiene (non-negotiables)

1. **Benchmarks, split correctly:** AdvBench 520 (comply = bad); DoNotAnswer 939 (comply = bad); **XSTest split into 200 safe (comply = good; refusal = false refusal, FRR) and 50 unsafe (comply = bad)** — never pool them again. GSM8K 1,319 test, pass@1.
2. **ASR by judge, not heuristics:** Llama-Guard-3-as-judge (or fixed LLM judge with published prompt) decides comply/refuse/harmful; plain refusal-keyword rate reported alongside; 200-response human spot-check per model; Wilson CIs on proportions.
3. **One frozen round-0** per model (greedy, fixed seed), reused in every table/figure; any table whose round-0 differs by >CI width is a bug.
4. **Multi-seed:** 3 seeds (data-order + generation seeds) on headline conditions; mean ± std; paired tests for drift (final vs round-0 within seed); ablations 1 seed, labeled as such.
5. **Pre-registered predictions:** freeze E1–E3 predictions (in this doc, §4 table and §6) before unblinding results — mention in the paper that predictions were registered in the internal protocol (docs/RESEARCH_PROTOCOL).
6. **Compute/overhead reporting** for the filter (reviewers WILL ask).

---

## 9. Literature & Citation Ledger

**Verified (safe to cite):**
- Cloud et al. 2025, arXiv:2507.14805 — subliminal learning; trait transmission; filtering fails; shared-init requirement; Thm. 1; shuffle experiment; MLP demo. (Cite the specific experiments we lean on; do not attribute it solely to "Anthropic".)
- Bianchi et al., "Safety-Tuned LLaMAs" (arXiv:2309.07875, ICLR 2024) — ~3% safety examples in single-shot instruction tuning restore safety; over-refusion trade-off documented. Cite ONLY for single-shot static tuning; our delta = multi-round + latent filter.
- Shen et al., SEAL (arXiv:2410.07471, ICLR 2025) — bilevel-optimization data ranker for safe fine-tuning. **No EWC in SEAL.** Optional baseline.
- Qi et al. (ICLR 2024) — benign fine-tuning compromises safety.
- Kirkpatrick et al. 2017 — EWC. Zelikman et al. 2022 STaR; Hosseini et al. 2024 V-STaR; Singh et al. 2024 ReST-EM. Röttger et al. 2023 XSTest; Wang et al. 2023 DoNotAnswer; Zou et al. 2023 AdvBench; Cobbe et al. 2021 GSM8K. Hu et al. 2022 LoRA.

**To bring in (verify each before citing — no inherited sins):**
- Inan et al. 2023 (Llama Guard); dutas/Llama-Guard-3 card — semantic SOTA filter.
- **Arditi et al. 2024 (arXiv:2406.11717), "Refusal in LMs is mediated by a single direction"** — refusal-direction extraction (our SDS-act + variant A). Also Zou et al. 2023 Representation Engineering (arXiv:2310.01405).
- **Betley et al. 2025, emergent misalignment (arXiv:2502.17424)** — insecure-code FT teacher for E3c; their public dataset.
- Yang et al. 2023, Shadow Alignment (~100 examples unlock harm) — safe-data-erosion taxonomy.
- **Alemohammad et al. 2023/2024, "Self-Consuming Generative Models Go MAD"** — iterated self-training amplification (C2 support).
- Wang et al. 2025, persona-features / emergent misalignment control (arXiv:2506.19823) — internal-feature steering of misalignment (related to subliminal channel).
- Lopez-Paz & Ranzato 2017 (GEM), Yu et al. 2020 (PCGrad) — gradient-projection relatives of variant B; Koh & Liang 2017 influence functions (gradient-scoring lineage; the stray `2409.05208` PDF in the repo root is an influence-attribution paper — related background for §7.3, keep or remove later).
- Continual-learning-for-LLM-safety survey pass (find what exists 2024–2026; the gap claim "nobody isolates the latent channel in multi-round self-improvement" must survive a fresh search).
- Instructor's linked X thread (fjzzq2002) — treat as color, not citation; verify author attribution before any use.

**Never again:** SEAL↛EWC; SEAL↛self-improvement rounds; "Anthropic's subliminal learning" without qualification.

---

## 10. Compute Plan (8×A100 remote / RTX 4050 local)

- **Remote (production):** LoRA 8B on 40GB×8 → plain DDP/accelerate suffices (bf16 weights ~16GB/GPU; adapters tiny); FSDP only if we add a full-FT robustness arm. EWC diagonal Fisher sharded across ranks (8B floats = 32GB, ~4GB/rank — fine, or LoRA-modules-only Fisher to match the training parameterization; decide in M1, state choice in paper).
- **Local (4050, 6GB):** unit tests for filter math (torch CPU/GPU), hidden-state extraction + refusal-direction pipeline on GPT-2/TinyLlama-scale dummy models, threshold-calibration harness, plotting, LaTeX. **No 8B loops, no benchmark gens locally** (enforced by a `--smoke` profile in CI).
- **Budget sketch (to be measured in M1 pilot, not trusted a priori):** per round per run: generation ~15–25 min; Llama-Guard-3 gate ~5–10 min; gradient filter ~30–40 min; LoRA SFT ~10–15 min; full evals ~30–45 min → ≈1.5–2.5 h/round → **15–25 node-hours per 10-round run**. Headline grid E1: 2 models × 4 conditions × 3 seeds = 24 runs ≈ 360–600 node-hours. Mitigations if tight: full safety evals only at rounds {0,5,10} + fixed n≈300 probe at intermediate rounds; E3 arms are 3-round single-distill runs (≈3–5 node-hours each). **M1 must produce the real number before we commit the grid.**

## 11. Timeline Reality Check (today: 2026-08-21)

ICLR 2027 full-paper deadline is ≈ late Sept 2026 (ICLR 2026 was Sept 24, 2025) — **~4–5 weeks out.** Honest assessment: the full program above does not fit; the MVP ladder does, *if* compute is dedicated and M1's gate passes:

- **Scope ladder:** *MVP = E0 + E1 (4 conditions, 3 seeds) + E3a (+E3d) + E5-EWC/Bianchi (1 seed) + telemetry.* *Target = + E2 full audit + E3c + ablations.* *Stretch = + E6 + second dataset.*
- **M0 (this week):** instructor sign-off on §12 decisions; repo cleanup; model/gated-dataset access verified; eval module frozen (judged ASR + split XSTest); telemetry built.
- **M1 (wk 1–2):** pilot — 1 model, naive STaR, 1 seed, 10 rounds, full telemetry. **Gate G1: does drift on the Instruct checkpoint exceed seed noise?** (Old evidence says LLaMA-base drifted +11 pp but Qwen-Instruct ≈ flat; nobody has run LLaMA-3-8B-*Instruct*.) Measure true per-round cost → finalize grid size. If G1 fails on both models → escalate LR/rounds or pivot framing to "filters are blind + latent early-warning" (still viable, §4).
- **M2 (wk 2–3):** E1 MVP grid + E3a/d.
- **M3 (wk 3–4):** E5 baselines, seeds completion, stats, E4 calibration figures.
- **M4 (wk 4–5):** manuscript (template: `iclr2027_conference.sty`), AC-style self-audit vs the §8 checklist, submit.
- **Fallback venue if M2 slips >1 week:** ICML 2027 (late-Jan deadline) with the full program including ablations — a materially stronger paper. Decision checkpoint at end of M2.

## 12. Open Decisions (need instructor/user sign-off — recommendations marked)

1. **Venue:** attempt ICLR 2027 MVP or go straight to ICML 2027 full program? **Rec: build MVP; decide at M2 gate.**
2. **Headline filter variant:** B (gradient-alignment) with A/D as cheap stages, C as EWC-bridge ablation. **Rec: yes.**
3. **E3c misaligned-teacher arm:** include (internal-only, no harmful outputs released, standard practice in this lit — Qi/Betley/Shadow all do it)? **Rec: include if instructor comfortable; it's our strongest mechanism evidence.**
4. **Parameterization:** LoRA everywhere (gradient filter then lives in adapter space — cheap, but a reviewer may ask whether full-FT drift differs). **Rec: LoRA primary + one full-FT/FSDP sanity arm if budget allows.**
5. **Seeds:** 3 on headline, 1 on ablations — acceptable? **Rec: yes, stated explicitly.**
6. **Eval depth:** full benchmarks at rounds {0,5,10}, n≈300 probes otherwise. **Rec: yes for MVP.**
7. **Models:** the two mandated Instruct checkpoints only. **Rec: yes.**

## 13. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Instruct checkpoints don't drift (G1 fails) | Medium (Qwen-Instruct was ~flat before; LLaMA-Instruct never run) | Escalate rounds/LR; E3c arm still demonstrates the channel; reframe per §4 contingency |
| Drift exists but E3 says "it's just forgetting" (C1) | Medium | Honest reframe; paper remains: blind filters + latent early-warning + filter/baseline comparison |
| Gradient filter over-rejects → kills math gains | Medium | Threshold calibrated to ≥90% human-gold pass rate; soft down-weighting ablation; report acceptance rates |
| Compute grid doesn't fit before deadline | High | Scope ladder; intermediate-round probe evals; ICML fallback |
| Judge/ASR metric disputed by reviewers | Medium | Dual metric (judge + refusal-rate), published prompts, human spot-check |
| Gated HF models (Llama-3, Llama-Guard-3) access delays | Low | Verify in M0 |
| Subliminal framing attacked as buzzword | Medium | Claims ledger: the word "subliminal" only where E3 supports it; everywhere else "latent/gradient-level" |

## 14. Claims Ledger (what we may write, and only after which experiment)

| Claim | Requires |
|---|---|
| "STaR loops erode safety on aligned Instruct checkpoints" | E1 cond.1 vs frozen round-0, 3 seeds |
| "State-of-the-art semantic filters reject ≈0 of the responsible data" | E1 cond.2 ≈ cond.1 + E2 audit log |
| "The drift is (partly) carried by a latent, sequence-level channel" | E3a (self ≫ cross-base ≈ human) and/or E3d shuffle result |
| "Misalignment transmits through benign math CoT in open-weight models" | E3c |
| "Latent drift precedes behavioral drift" | E1 telemetry (SDS leads ASR) |
| "Our Subliminal Filter bounds drift where EWC/Bianchi-static do not" | E1 cond.3/4 vs E5, with over-refusal + math trade-offs reported |
| "EWC causes over-refusal" | E5 EWC run ONLY — never before |

---

*Next step after sign-off of this doc: freeze decisions §12, then M0 (repo cleanup + eval freeze), then M1 pilot (G1 gate). Code (`subliminal_filter.py`, `ewc_baseline.py`, `distributed_star.py`, telemetry module, judged-ASR eval) and the ICLR manuscript are built only after this doc is approved — per current instruction, none of that is written yet.*
