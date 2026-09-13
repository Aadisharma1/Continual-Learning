# Engineering Plan — `conti` Codebase Rebuild for the Subliminal Pivot

**Status:** PLAN ONLY. Zero code written, zero commands executed until explicit go-ahead (§12).
**Scope:** everything that must exist before the 8×A100 DGX cluster can run the experiment grid on day one of access.
**Companion docs:** `IDEATION_subliminal_pivot.md` (science), `SPRINT_local_rtx4050_4days.md` (schedule), `SUPERVISOR_BRIEF_subliminal_pivot.md` (summary).

---

## 0. Ground Rules

1. **No execution without an explicit "go" from you.** That includes: `pytest`, any `python -c`, `pip install`, dataset downloads, file deletions, `git push`. Writing new `.py`/`.yaml`/`.md` files inside the repo counts as *authoring*, not executing — it starts only when you unlock **Gate W** (§12).
2. Local machine = RTX 4050 6 GB → **no 7–8B weights ever load locally**. All local validation uses GPT-2 (124M) / TinyLlama-1.1B / Llama-Guard-3-**1B**, or pure-tensor synthetic fixtures.
3. Every module ships with unit tests written in the same commit; the smoke test (`configs/smoke_gpt2.yaml`) is the Day-4 deliverable that proves end-to-end wiring before DGX.
4. Distribution layer = **Accelerate over `torch.distributed`** (nccl), launched via `torchrun`. Rationale: the whole existing codebase (`SFTTrainer`, `run.py`) already speaks Accelerate; Accelerate composes with FSDP (needed only if we add a full-FT sanity arm) and DeepSpeed is unnecessary for LoRA 8B on 40 GB × 8. Raw DeepSpeed configs would be a rewrite with zero benefit — this decision is deliberate and reversible.

---

## 1. Current-State Audit (what exists, what's broken, what's missing)

Read every file listed below. Verdicts:

| File | Verdict | Notes |
|---|---|---|
| `conti/loop/run.py` | **Rewrite → `distributed_star.py`** | Monolithic; generation single-process on main rank then broadcast (wasteful on 8 GPUs); verifier is the dead-code keyword/toxicity pair; no resume logic despite `context.md` claiming `_find_last_completed_round` exists (not in current source); math eval capped at `min(200, …)` samples (line ~105) — the exact bug that invalidated old runs; EWC setup has a TODO and never computes Fisher. |
| `conti/training/sft.py` | **Keep + extend** | Clean Accelerate+PEFT trainer; already has `set_ewc()` hook. Needs: per-candidate-gradient helper for the filter (or that lives in the filter module), replay-loss mixing (currently replay texts are just appended to `texts` — fine), optional full-FT path for robustness arm. |
| `conti/training/ewc.py` | **Replace → `training/ewc_baseline.py`** | `compute_fisher()` exists but is never called anywhere; Fisher would be self-distillation (labels=input_ids) not safety-specific; no restriction to trained parameterization; no sharding; penalty loops named_parameters every step (slow). |
| `conti/verifier/keyword.py`, `model_scorer.py` | **Retire to legacy** | The dead-code filter. Keep files (history), remove from the active pipeline; replaced by `filters/semantic.py`. |
| `conti/replay/buffer.py` | **Keep as-is** | Simple JSONL buffer, sampling with replacement — adequate. May add stratified sampling later; not blocking. |
| `conti/eval/metrics.py` | **Modify heavily** | Unbatched per-sample generation (slow: context.md claims batch_size=128 but current source generates one-by-one); ASR uses `len(gen) > 120` length heuristic (must die); no XSTest safe/unsafe distinction; `wilson_ci` is correct — keep. |
| `conti/eval/drift.py` | **Keep + wrap** | DriftTracker is fine for behavioral drift; latent drift gets its own telemetry module. |
| `conti/eval/stats.py` | **Keep** | `summarize()` gives multi-seed mean/std/CI — exactly what §8-hygiene needs; underused today. |
| `conti/data/loaders.py` | **Modify** | Defaults to bundled 12–15-sample subsets; no full-dataset HF mapping in this version (context.md says `advbench_full` etc. existed — gone); no XSTest split; no probe-set loading; GSM8K namespace bug already patched once (`openai/gsm8k`) — must pin. |
| `conti/config_schema.py` | **Extend (v2)** | Stale defaults (`Llama-3.1-8B-Instruct`, subset datasets); experiment literals don't cover new conditions; no fields for filters/probes/judge/telemetry/thresholds. |
| `conti/logging.py`, `reproducibility.py` | **Keep** | Manifest + seeding infra solid; add filter-stage and SDS fields to round logs. |
| `scripts/run_experiment.py` | **Keep + siblings** | Thin CLI; add `scripts/eval_checkpoint.py`, `scripts/build_distillation_corpus.py`, `scripts/calibrate_filter.py`, `scripts/launch_dgx.sh`. |
| Root directory | **Surgery** | `(1)`-suffixed duplicates and `-Revenant` variants throughout `conti/`; stray unrelated PDF (`2409.05208v4.pdf` — influence-function attacks paper, not ours). Deletion list in §9. |

**Net:** ~60% of the runtime skeleton is reusable; all *science* modules (filter, telemetry, true EWC, judged eval, distributed loop v2) are net-new.

---

## 2. Target Architecture

```
conti/
├── config_schema.py            [MODIFY] ContiConfig v2 (+FilterConfig, ProbeConfig, JudgeConfig, TelemetryConfig, MediationConfig)
├── distributed_star.py         [NEW]    loop v2: torchrun-ready STaR engine (single entry for ALL conditions)
├── filters/
│   ├── __init__.py             [NEW]
│   ├── base.py                 [NEW]    TrajectoryFilter protocol + FilterLog schema (stage, verdict, score, latency)
│   ├── semantic.py             [NEW]    LlamaGuardFilter (3-8B on DGX / 3-1B local-smoke), batched chat-format calls
│   ├── refusal_direction.py    [NEW]    extract_refusal_direction(), project_hidden_states() (Arditi et al. diff-in-means)
│   └── subliminal_filter.py    [NEW]    GradientAlignmentFilter (headline) + FisherOverlapFilter + calibration harness
├── telemetry/
│   ├── __init__.py             [NEW]
│   └── sds.py                  [NEW]    SubliminalDriftScorer: SDS-act / SDS-grad / SDS-param per round → logs/sds.jsonl
├── training/
│   ├── sft.py                  [MODIFY] +replay loss weighting α, +full-FT flag, +grad-hook export
│   └── ewc_baseline.py         [NEW]    SafetyFisher (diagonal, safety-data-driven, adapter-or-full, sharded-ready) + EWCPenalty v2
├── eval/
│   ├── metrics.py              [MODIFY] batched generation; XSTest split outputs; drop length-heuristic ASR
│   ├── judge.py                [NEW]    LlamaGuardJudge + generic LLM-judge (prompt-versioned) + human-spotcheck exporter
│   ├── round0.py               [NEW]    frozen round-0 baseline producer (greedy, fixed seed, one per model)
│   ├── drift.py                [KEEP]
│   └── stats.py                [KEEP]
├── data/
│   ├── loaders.py              [MODIFY] full-HF benchmark map (pinned namespaces), xstest safe/unsafe split, probe loader
│   ├── build_corpus.py         [NEW]    distillation-corpus builder for E3 arms (self/cross/misaligned/human/shuffled)
│   └── probes/                 [NEW]    versioned contrast sets (harmful/benign pairs) + safety probe eval set
├── legacy/
│   └── (keyword.py, model_scorer.py moved here) [MOVE]
tests/
├── test_filters_semantic.py    [NEW]  mock-backed (no downloads in CI)
├── test_refusal_direction.py   [NEW]  GPT-2 synthetic
├── test_subliminal_filter.py   [NEW]  tiny-LoRA synthetic ranking check
├── test_ewc_baseline.py        [NEW]  toy forgetting task: penalty ↓ drift
├── test_eval_round0.py         [NEW]  greedy determinism, XSTest split accounting
├── test_loaders.py             [NEW]  expected counts/splits (mocked datasets)
└── test_smoke_loop.py          [NEW]  2-round GPT-2 loop, n=50, asserts artifacts exist (the D4 gate)
configs/
├── smoke_gpt2.yaml             [NEW]
├── e1_{naive,semantic,semisub,full}_{llama,qwen}_s{42,7,123}.yaml   [NEW, generated]
├── e3_mediation_*.yaml         [NEW]
├── e5_{ewc,bianchi}.yaml       [NEW]
scripts/
├── run_experiment.py           [KEEP]
├── eval_checkpoint.py          [NEW]
├── calibrate_filter.py         [NEW]
├── build_distillation_corpus.py[NEW]
├── make_grid.py                [NEW]  writes the e1_* YAML grid programmatically (no hand-typed 24 files)
└── launch_dgx.sh               [NEW]  torchrun wrapper(s) + runbook header
```

---

## 3. Module Specifications

### 3.1 `filters/base.py` — contracts first
```python
@dataclass(frozen=True)
class FilterVerdict:
    stage: str            # "semantic" | "subliminal_grad" | "subliminal_fisher" | ...
    accepted: bool
    score: float          # stage-native score; threshold recorded separately
    reason: str           # machine-parseable, e.g. "cos_gsafe=-0.21 < tau=0.05"

class TrajectoryFilter(Protocol):
    def score_batch(self, items: list[TrajectoryItem]) -> list[FilterVerdict]: ...
```
Every filter logs one row per candidate to `logs/filter_log.jsonl` (superset of the old schema so old plotting still works). `TrajectoryItem = {question, generation, gold, meta}` — the unit that flows through the round.

### 3.2 `filters/semantic.py` — Llama-Guard-3 gate
- Wraps `meta-llama/Llama-Guard-3-8B` (DGX) or `-1B` (local smoke) via transformers, batched, bf16, dedicated device slice.
- Input format: official guard chat template `[user] {prompt} [assistant] {response}`; output parsed to safe/unsafe + category.
- API: `LlamaGuardFilter(model_id, device_map, batch_size=16)`; `score_batch(items) -> verdicts`.
- Budget: 2,000 traces × ~700 tok ≈ 5–10 min on one A100; runs on rank 0..k while other ranks pre-tokenize next gen batch (overlap documented in §4).
- Failure modes handled: gated-model missing → hard error with instructions (never silently degrade to keyword-only — that's how the dead-code filter happened).

### 3.3 `filters/refusal_direction.py`
- `extract_direction(model, tokenizer, harmful_prompts, benign_prompts, layer, token_pos="last") -> DirectionVector`
  Difference-in-means of final-hidden-states over the contrast pair set (Arditi et al. 2024 method), L2-normalized; extracted **once per model at round 0**, cached to `probes/{model}/direction.pt` with sha256 of the prompt lists in metadata.
- Contrast set: ~100 AdvBench harmful vs ~100 XSTest-safe benign prompts, versioned under `conti/data/probes/`.
- Used two ways: cheap activation pre-gate (optional) and, primarily, as the readout for SDS-act telemetry.

### 3.4 `subliminal_filter.py` — headline module
Math implemented (symbols identical to ideation §7.4):
```
g_i      = ∇_{θ_adapter} L(y_i)                    # per-candidate gradient, LoRA params only
g_safe   = mean over B_safe-sample of ∇_{θ_adapter} L(b)
accept i ⟺  cos(g_i, g_safe) ≥ τ_g
FisherOverlap variant: score_i = Σ_j F_j g_ij² / ‖g_i‖²  ≤  τ_F
Soft mode: w_i = σ(β·(cos(g_i,g_safe) − τ_g))       # down-weight instead of reject
```
Implementation decisions locked:
- Gradients in **adapter space only** (peft params): r=16 → ~20 M trainable scalars ⇒ g_i ≈ 80 MB fp32; compute with `torch.autograd.grad` on loss of a single padded example, grads accumulated in fp32; micro-batch of 1 candidate per backward (batched per-candidate grads via functorch/vmap is a stretch goal, not a dependency).
- `g_safe` computed **once per round** from 64 sampled replay examples (cached).
- Throughput budget: ~0.5–1.0 s/candidate on one A100 → 2,000 candidates ≈ 17–35 min/round; parallelized 8-way by rank-sharding the candidate list (each rank scores its shard, all-gather verdicts).
- `calibrate_threshold(scores_benign, constraint=0.90)` → picks largest τ keeping ≥90% of human-gold GSM8K traces; emits ROC vs "known-bad" labels (E3c traces or top-drift-inducing E1 traces) for the paper figure. Exposed as `scripts/calibrate_filter.py`.
- Failure mode guarded: if acceptance rate <70% at calibrated τ, loop aborts with a loud config error (over-filtering kills STaR bootstrap; better to fail than to silently gut training).

### 3.5 `telemetry/sds.py`
Per round, on a fixed 200-prompt safety probe set (versioned):
- **SDS-act**: mean over probes of ⟨h_last(x), d̂⟩ (projection on refusal direction) — tracks alignment geometry.
- **SDS-grad**: fraction of this round's accepted candidates with cos(g_i, g_safe) < 0.
- **SDS-param**: cosine(current flat adapter vector, round-0 snapshot).
Writes `logs/sds.jsonl` rows `{round, sds_act, sds_grad, sds_param, n_probes}`; these are the leading-indicator curves (paper claim: SDS rises before ASR).

### 3.6 `training/ewc_baseline.py` — the real EWC
```
SafetyFisher.compute(model, safety_texts, ...)   # diagonal Fisher = E[ĝ²] over SAFETY data
                                                 # restricted to the ACTUAL trained parameterization
                                                 # (adapters when LoRA; all params only in full-FT arm)
EWCPenalty_v2.penalty(model)                     # λ/2 · Σ F_j (θ_j − θ*_j)²  fused in SFTTrainer hook
```
- Fisher stored bf16 on GPU / fp32 CPU-offloaded; adapters-only ⇒ ≤160 MB — trivial. Full-FT arm: 8 B scalars ⇒ sharded across ranks (each rank owns the slice of params FSDP assigns it); documented, behind a flag, never on the 4050.
- λ sweep ∈ {50, 500, 5000} via three configs, not code edits.
- Unit test: on a toy 2-task setup (GPT-2, 10 steps task-A Fisher, then task-B training ± penalty) assert penalty reduces task-A loss increase. This is the anti-ghost-baseline proof artifact.

### 3.7 `distributed_star.py` — loop v2 (one engine, every condition)
Round structure (all ranks):
1. Shard the 2,000-prompt pool across 8 ranks → each generates its shard (batched, temperature from config) — fixes the current design where rank 0 does all generation.
2. Correctness filter (exact-match numeric, unchanged logic, extracted to `eval/scoring.py`).
3. **Filter cascade**: semantic (Llama-Guard) → optional subliminal grad/Fisher gates; per-rank scoring, `all_gather` of verdicts; full log retained.
4. Assemble round training set (+ k=4-per-batch replay mixing with weight α; + EWC penalty object if armed).
5. `SFTTrainer.train_on_texts(...)` (unchanged call signature).
6. Checkpoint adapter → `outputs/<run>/checkpoints/round_r/`.
7. Eval phase (see 3.8) + SDS telemetry; append `round_metrics.jsonl`, `sds.jsonl`.
Resume: scan `checkpoints/round_*` for highest complete round marker (`.done` sentinel file written post-save — the old resume logic described in context.md doesn't exist in current source; rebuilt properly).
Condition dispatch purely by config flags (which filters enabled, replay on/off, ewc on/off, corpus path override for E3 arms) — one engine, no per-experiment forks.

### 3.8 Eval v2 (`metrics.py` mods + `judge.py` + `round0.py`)
- Batched generation (left-padding, batch 32–128, greedy for eval) — replaces the per-sample loop.
- Benchmarks pinned: `openai/gsm8k` (test 1,319) · AdvBench 520 · **XSTest split → `xstest_safe`(200) + `xstest_unsafe`(50)** · DoNotAnswer 939. Loader asserts exact counts and fails loudly on mismatch (this class of silent-subset bug killed the old runs).
- Metrics per benchmark: `refusal_rate` (keywords), `asr_judge` (Llama-Guard-as-judge on (prompt,response)), `frr` (false-refusal, XSTest-safe only). Length heuristic deleted. Wilson CIs kept.
- `round0.py`: one frozen greedy evaluation per model → `round0_baseline.json`; every run copies it; any table whose round-0 differs is a bug by construction.
- Judge prompts stored under `configs/judge/` with version hashes in every log row.

### 3.9 Data layer (`loaders.py` mods + `build_corpus.py`)
- Full-dataset map (namespaces verified against the `openai/gsm8k` fix already committed), subset files demoted to smoke-only.
- Probe sets: `probes/harmful_v1.jsonl` (100), `probes/benign_v1.jsonl` (100), `probes/safety_probe_eval_v1.jsonl` (200) — versioned, sha256'd into manifests.
- `build_corpus.py` (E3): generate correctness-filtered GSM8K traces from a chosen teacher (`--teacher {self,cross_base:<id>,misaligned_ckpts,misaligned_ckpt}`, plus `--human-gold` and `--shuffle` controls), then subsample to **matched size AND matched acceptance-rate/accuracy** across arms (the confound killer), emitting `corpora/e3_<arm>.jsonl` with full provenance in each row.

### 3.10 Config v2 (additive; old keys stay valid)
New literals for `loop.experiment`: `naive_continual`, `semantic_only`, `semantic_subliminal`, `semantic_subliminal_replay`, `ewc_baseline`, `bianchi_static`, plus `mediation_sft` (driven by `mediation.corpus_path`). New sections: `filters` (semantic on/off + model id; subliminal kind/threshold/soft-mode), `judge`, `telemetry`, `probe`, `mediation`. Defaults flipped to the mandated Instruct checkpoints and full datasets.

---

## 4. One Round, End-to-End (v2 data flow)

```
pool(2000 prompts) ──shard──► 8×generate(batched) ──► 2000 candidates
   ├─ correctness filter (exact-match gold)          → rej_correct logged
   ├─ semantic gate  (Llama-Guard-3, batched)        → verdicts + latencies
   ├─ subliminal gate (per-candidate g_i vs g_safe)  → verdicts + scores   [conditional]
   ├─ assemble train set (+replay k=4/batch, α)      
   ├─ SFT (LoRA, AdamW, EWC penalty if armed)
   ├─ save checkpoint + .done marker
   └─ eval: GSM8K ∥ safety benchmarks ∥ SDS telemetry  → round_metrics.jsonl / sds.jsonl
```
Artifacts per run: `run_manifest.json`, `logs/{filter_log,round_metrics,sds,drift_log}.jsonl`, `supervised_counts.json`, `checkpoints/round_r/`, `round0_baseline.json` (copied).

---

## 5. Condition Grid → Config Map (what DGX will run)

| Ideation exp | Config family | Count |
|---|---|---|
| E1 main | `e1_naive / e1_semantic / e1_semisub / e1_full` × {llama-instruct, qwen-instruct} × seeds {42, 7, 123} | 24 runs |
| E3 mediation | `e3_self / e3_cross_l2q / e3_cross_q2l / e3_human / e3_shuffle` (+`e3_misaligned` pending decision) — 3 rounds each | 5–6 short runs |
| E5 baselines | `e5_ewc_l{50,500,5000}` (1 seed, 1 model), `e5_bianchi_static` | 4 runs |
| E6 stretch | 20-round extension of `e1_full` llama s42 | 1 run |
All generated by `scripts/make_grid.py` — no hand-typed YAMLs, no copy-paste drift.

## 6. Test Plan (all runnable on RTX 4050)

| Test | Fixture | Assertion core |
|---|---|---|
| `test_loaders` | mocked HF datasets | exact counts; XSTest split 200/50; loud failure on mismatch |
| `test_filters_semantic` | mocked guard outputs | parse rules, batch handling, hard-error on missing model |
| `test_refusal_direction` | GPT-2 + synthetic harmful/benign | direction unit-norm; projections separate the sets |
| `test_subliminal_filter` | tiny LoRA + crafted examples | candidate known to push against anchor ranks below τ; ≥90% benign-pass calibration works |
| `test_ewc_baseline` | GPT-2 toy 2-task | Fisher finite, positive; penalty reduces task-A forgetting |
| `test_eval_round0` | GPT-2 greedy | determinism across 2 invocations; split accounting adds up |
| `test_smoke_loop` | GPT-2, 2 rounds, n=50 | all artifacts exist; filter log row count == candidates; resume picks up after simulated crash |

CI-style discipline locally: `pytest -q` must pass before any module is considered done. **Nothing here executes until you say so.**

## 7. Dependencies & Access Checklist (Day-1 blockers)

- pyproject: no new runtime deps required (guard + judge run on existing transformers stack). Dev extras already sufficient. Optional: pin `transformers>=4.43,<5` for Llama-Guard-3 chat-template compatibility (verify at install time on DGX).
- HF gated approvals needed **before DGX day**: `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Llama-Guard-3-8B`, `meta-llama/Llama-Guard-3-1B` (Qwen2.5-7B-Instruct is ungated). Valid `HF_TOKEN` in `.env` (dotenv already wired in `conti/__init__.py`).
- Datasets re-verified on DGX network: `openai/gsm8k`, `walledai/AdvBench`, `walledai/XSTest`, `LibrAI/do-not-answer`.

## 8. Repo Surgery (pending your approval — destructive)

1. Delete all `*(1).py` duplicates and `*-Revenant*` variants across `conti/` (canonical non-suffixed files stay).
2. Move `verifier/keyword.py` + `model_scorer.py` → `conti/legacy/` (keep history; out of the hot path).
3. `.gitignore`: `outputs/`, `checkpoints/`, `*.pt`, `__pycache__/`, `.env`.
4. Flag `2409.05208v4.pdf` (unrelated influence-functions paper) for removal from repo root.
5. Security note carried from context.md: rotate the exposed WandB key (your action, not mine).
Git strategy: branch `feat/subliminal-pivot` off `master`; one commit per module with its tests; merge to master only after `test_smoke_loop` passes; tag `dgx-ready`. Pushes happen only with your say-so.

## 9. DGX Runbook (outline — finalized when cluster details arrive)

Env: conda/venv → `pip install -e ".[dev]"` → `HF_TOKEN` → cache dir on fast scratch. Launch: `torchrun --nproc_per_node=8 scripts/run_experiment.py --config configs/e1_....yaml`. Storage layout `outputs/<exp>/<model>/<seed>/…` matching the grid generator. Monitoring: tail `logs/round_metrics.jsonl`; wallclock per round logged by manifest. Resume = relaunch same command (sentinel-file logic). Open question for you: SLURM or bare metal?

## 10. Engineering Risk Register

| Risk | Mitigation in design |
|---|---|
| Per-candidate gradients slower than estimated | Rank-sharded scoring; measured in pilot; fallback = score only survivors of semantic gate (already the plan) + soft-mode on top-k suspicious |
| peft + autograd.grad interaction edge cases | Unit-tested on tiny LoRA before DGX; fallback = manual forward hooks computing loss on adapter-only graph |
| Llama-Guard license/redistribution | Weights used via HF gating, never re-hosted; outputs (labels) published, not model |
| OneDrive path quirks (locking, sync) during heavy local edits | Work in-repo but avoid running anything heavy locally anyway; consider relocating repo off OneDrive for DGX-sync sanity (your call) |
| Silent dataset drift (namespace changes again) | Loader count assertions + manifest sha256 of loaded corpora |
| Scope creep in the loop engine | One engine + config dispatch; no per-experiment code forks allowed |

## 11. Build Order (= sprint days, each ending green tests)

D1: repo surgery → loaders v2 + tests → eval v2 (batched, split, judge scaffold) + round0 script.
D2: refusal_direction + telemetry/sds + probe data files (+ tests).
D3: subliminal_filter + ewc_baseline (+ tests) — the science core.
D4: distributed_star loop v2 + config v2 + make_grid + smoke_gpt2 runbook → tag `dgx-ready` (after your approval to push).

## 12. Execution Gates — what I need from you

- **Gate W (authoring):** say **"go"** → I create the branch and start writing D1 files + tests. Still zero execution.
- **Gate R (running):** a separate, explicit go for `pytest` / smoke test / any command.
- **Gate D (destructive):** explicit yes for §8 deletions/moves.
- Info I need eventually: HF token valid for gated repos? DGX scheduler (SLURM vs bare)? Push rights confirmed on `Aadisharma1/Conti`?
