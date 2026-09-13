# plan.md — DGX Execution Plan (conti · subliminal safety pivot)

**Hardware target:** DGX with 8× NVIDIA A100 40GB (single-node assumed; multi-node notes in §9).
**Deliverable class:** operational runbook — steps, checkpoints, logging, budgets, failure playbooks.
**Companions:** `docs/IDEATION_subliminal_pivot.md` (science) · `docs/ENGINEERING_PLAN_subliminal_code.md` (code) · `docs/SPRINT_local_rtx4050_4days.md` (D1–D4 laptop sprint).
**Status rule:** this plan executes NOTHING locally. It becomes live when the code exists (`dgx-ready` tag) and the cluster turns on.

---

## 0. TL;DR

1. **Days 4→5:** finish laptop sprint → push `dgx-ready` → Day 5 morning: cluster prep (~2 h) → frozen round-0 baselines for both Instruct models.
2. **Gate G1 pilot:** 1 model, naive STaR, 10 rounds, telemetry on. Decision in §7.1: continue / fix / reframe.
3. **Main grid E1:** 2 models × 4 conditions × 3 seeds = 24 runs (~15–16 node-hours each ⇒ ≈ 380 node-hours total ⇒ ≈ 16 days on one 8-GPU node, or 8 days if two nodes).
4. **E3 mediation + E5 baselines** (+EWC λ×3 + Bianchi-static): ≈ 60–80 node-hours.
5. Triage ladder in §8 if the ICLR-2027 clock beats us.

---

## 1. Blockers BEFORE any cluster command (must all be ✅)

| # | Item | Owner / where |
|---|---|---|
| B1 | Code complete at tag `dgx-ready` (per ENGINEERING_PLAN §11 build order) | laptop sprint D1–D4 |
| B2 | HF token accepted for gated repos: `Meta-Llama-3-8B-Instruct`, `Llama-Guard-3-8B`, `Llama-Guard-3-1B` | request **today** (HF account) |
| B3 | Repo `gh`/`https` push pull works; team can clone `Aadisharma1/Conti` on cluster | Day 4 |
| B4 | DGX facts collected: SLURM vs bare-metal, shared scratch path, CUDA ≥ 12.1, internet egress to huggingface.co | Day 5 morning |
| B5 | WandB key rotated (old one was exposed); new key in `.env` (never committed) | you |
| B6 | Frozen probe/contrast sets committed: `probes/harmful_v1.jsonl`, `benign_v1.jsonl`, `safety_probe_eval_v1.jsonl` | Day 2–3 sprint |

---

## 2. Cluster Prep — Day 5 Morning (~2 h), exact steps

```bash
# 2.1 env (shared scratch, NOT $HOME)
cd $SCRATCH && git clone https://github.com/Aadisharma1/Conti.git && cd Conti
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"          # per pyproject (no new heavy deps)
export HF_TOKEN=<gated-token>                          # .env file via dotenv is wired already

# 2.2 dataset pre-cache (fail-fast with loud assertions in loaders)
python scripts/precache_data.py   # downloads: openai/gsm8k, walledai/AdvBench, walledai/XSTest,
                                  # LibrAI/do-not-answer  → asserts counts {1319, 520, 250→(200 safe/50 unsafe), 939}

# 2.3 model weight pre-cache
python scripts/precache_models.py # Meta-Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Llama-Guard-3-8B (+ -1B)
# (weights ≈ 16 GB bf16 × 3 models ≈ 50 GB on scratch — fine)

# 2.4 sanity (2 GPU, 5 min)
torchrun --nproc_per_node=2 scripts/sanity_compute.py   # bf16 alloc, nccl all-gather, flash-attn flag probe
```
**Checkpoint C0:** prep script prints `ENV OK`, datasets table identical to expected counts, models cached with sha256 × manifest entries. Record in `meta/cluster_prep.log`.

---

## 3. Naming, Storage & Checkpoint Conventions

```
$SCRATCH/Conti/outputs/
├── round0/llama3-8b-instruct/round0_baseline.json        # frozen, copied into every llama run dir
├── round0/qwen2.5-7b-instruct/round0_baseline.json
├── pilot_g1/{model}/                                     # gate run
├── e1/{condition}/{model}/{seed}/                        # 24 runs
├── e3/{arm}/  e5/{ewc_l50|ewc_l500|ewc_l5000|bianchi_static}/
└── corpora/e3_{self,cross_l2q,cross_q2l,human,shuffle[,misaligned]}.jsonl
```

Per-run dir: `run_manifest.json` · `round0_baseline.json` (copy) · `logs/{filter_log,round_metrics,sds,drift_log}.jsonl` · `supervised_counts.json` · `checkpoints/round_<r>/` (LoRA adapters only ≈ 150 MB) with `.done` sentinel per finished round · `metrics.json` (final).

**Checkpoint policy:** save adapter + optimizer state per round; sentinel `.done` written LAST (atomic resume signature). Resume = relaunch same command; engine fast-forwards to first round lacking `.done`. Retention: keep all rounds (adapters are tiny); delete optimizer states after run completes unless `--keep-opt`.

---

## 4. Logging Spec (what each file guarantees)

| File | Row schema keys (minimum) | Written by |
|---|---|---|
| `round_metrics.jsonl` | `round, model, condition, seed, math_pass1{acc,n,ci}, safety{advbench,xstest_safe_frr,xstest_unsafe_asr,donotanswer,aggregate}{rate,n,ci, judge_version}, accepted, rej_correct, rej_semantic, rej_subliminal, wallclock_s, gpu` | loop engine, per round |
| `filter_log.jsonl` | `round, item_id, stage, accepted, score, threshold, reason, latency_ms` | filter cascade |
| `sds.jsonl` | `round, sds_act, sds_grad, sds_param, n_probes, probe_version` | telemetry |
| `drift_log.jsonl` | `round, benchmark, baseline, current, abs_drift, rel_drift` | DriftTracker |
| `run_manifest.json` | config sha256, git commit, package versions, seeds, env | reproducibility.py (existing) |
| WandB project `conti-safety` (optional mirror) | same metrics, online | logging.py |

Invariants enforced in code: **round-0 row identical across all tables of one model**; every metric row carries `judge_version` + `probe_version` + `eval_mode ∈ {full, probe}`; filter-log row count == candidate count, every round (assertion).

---

## 5. Monitoring SOP

**Every ~2 h while a run is active (first 2 runs: hourly):**
- `tail -1 logs/round_metrics.jsonl` per active run; verify round index advancing (expected pace §6.4).
- GPU util `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` — generation <80% or train <90% sustained ⇒ file a perf note, don't restart.
- NaN/inf guard: engine aborts if `loss` non-finite twice in a row; log line `FATAL step` ⇒ see §10.4.
- Filter sanity (per round): `rej_semantic` expected ≈ 0 (that IS the finding — log, don't panic); `rej_subliminal` expected 2–10% after τ calibration; `accepted` ≥ 25 (below → capability-collapse risk; engine warns).

**Daily close-out (15 min):** copy `logs/` snapshots to `daily/YYYYMMDD/`; run `scripts/stats_rollup.py`; update the run-sheet (§6 table) with rounds/day actual vs target.

---

## 6. Phases — ordered, with per-run cards

### P0 — Frozen round-0 baselines (Day 5, ~2 h)
`scripts/eval_round0.py --model {llama,qwen} --mode full --greedy` → `outputs/round0/<model>/round0_baseline.json`.
**Checkpoint C1:** GSM8K ≈ published-class accuracy (Llama-3-8B-Instruct ~75–80 %, Qwen2.5-7B-Instruct ~80 %+); safety tables sane. If wildly off ⇒ stop, debug prompts/tokenization before anything else.

### G1 — Pilot gate (Days 5–7, ~16 node-hours incl. buffer)
One run: `e1_naive` × llama-instruct × s42, telemetry + semantic gate ON (we need its zero-rejection count here).
**G1 pass criteria (pre-registered):**
- aggregate judged-ASR drift by round 10 > +3 pp beyond seed-noise band, **or** SDS-act monotonic rise ≥ 6 rounds with ASR rise in final 3 rounds;
- math pass@1 gain ≥ +5 pp (loop actually self-improves);
- semantic rejections ≈ 0 (expected; confirms "blind filter" premise);
- per-round wallclock measured → re-derive §6.4 budget constants.
**Outcomes:** PASS → §6.2. DRIFT-FAIL → fix levers (LR ×2, 20 rounds, or check eval) before rerun (one retry budget). NO-DRIFT after retry → switch to the "honest reframe" paper track (ideation §4 contingency) — decide with instructor.

### E1 — Main grid (24 runs)  _(condition definitions live in configs generated by `scripts/make_grid.py`)_

| Condition | Filters active | Replay | Purpose |
|---|---|---|---|
| `e1_naive` | — | off | pathology |
| `e1_semantic` | Llama-Guard-3-8B | off | "blind filter" evidence |
| `e1_semisub` | Guard + gradient-alignment subliminal filter (τ from calib) | off | headline fix |
| `e1_full` | Guard + subliminal | k=4/batch (<1%) | full stack |

× models {Llama-3-8B-Instruct, Qwen2.5-7B-Instruct} × seeds {42, 7, 123} ⇒ **24 runs**.
**Checkpoint C2 (mid-grid):** after first model × condition × seed block (8 runs): audit SDS-vs-ASR lead relationship; confirm acceptance rates healthy; only then launch the rest.

### E3 — Mediation arms (~3-round short runs, 1 model = llama first)

| Arm | Training corpus | Test of |
|---|---|---|
| `e3_self` | student's own correct traces | baseline loop |
| `e3_cross_q2l` | same GSM8K prompts, traces generated by Qwen-Instruct | shared-base requirement |
| `e3_human` | gold GSM8K solutions (matched N + acceptance) | forgetting-only control |
| `e3_shuffle` | self-traces, within-trace shuffled | sequence-level carrier (Cloud et al. control) |
| `e3_misaligned`* | correct traces from insecure-code-FT teacher | theoretical worst-case channel |

*needs your earlier sign-off (instructor OK'd per decisions list). Corpora built by `scripts/build_corpus.py --arm …` one-time (gen ≈ same cost as 1 round each).

### E5 — Ghost-baseline exorcism (1 model, 1 seed)
`e5_ewc_l{50,500,5000}` (true safety-Fisher EWC; over-refusal on XSTest-safe is the metric of interest) + `e5_bianchi_static` (single-shot ~3% safety mix, NO per-round re-anchor → does the static recipe survive 10 rounds?).
**Checkpoint C3:** EWC numbers written into manuscript ONLY from these runs.

### E4 — Filter ablations (1 model, 1 seed, reuse pilot compute)
τ sweep {low, calibrated, high}, soft-weighting variant, Fisher-overlap variant, activation pre-gate variant. Each = one short run on 5 rounds.

### E6 — Stretch
20-round `e1_full` llama s42; second dataset (MATH-500 subset) sanity arm.

### 6.4 Per-round wallclock budget (to be re-derived by G1, current estimates)

| Stage | est. | notes |
|---|---|---|
| Generate 2,000 (sharded ×8) | 15–25 min | batch 128/GPU, 512 max-new |
| Correctness filter | <1 min | regex |
| Semantic gate (Guard-8B) | 8–15 min | on ~50–100 correctness survivors + spot-sample for rejection log |
| Subliminal gate | 2–6 min | ~100 per-candidate LoRA-space backwards, sharded |
| Train (1 epoch, LoRA) | 10–20 min | |
| Eval: probe mode (n=300) | 15–20 min | rounds 1–4, 6–9 |
| Eval: full mode (GSM8K 1319 + safety 1909 + judge) | 60–80 min | rounds 0, 5, 10 |
| **Round total** | **~1.2 h probe / ~2.2 h full** | ⇒ **≈ 15–16 node-hours per 10-round run** |

---

## 7. Compute Budget & Day Plan

| Block | runs | node-hours (est) |
|---|---|---|
| P0 + G1 | 1 + setup | ~20 |
| E1 grid | 24 | ~380 |
| E3 arms (+corpora) | 5–6 | ~40 |
| E5 baselines | 4 | ~60 |
| E4 ablations | 4–6 short | ~25 |
| E6 stretch | 1–2 | ~30 |
| **Total** | | **≈ 550 node-hours** |

On **one** 8-GPU node ≈ 23 days; ICLR-2027 full deadline is ~late-Sept ⇒ tight-but-possible only with zero slipped days + triage ready. **4-GPU allocation (current standing limit):** per-run wallclock roughly doubles (≈ 25–30 h for a 10-round run) ⇒ grid ≈ 26–30 days sequential — the triage ladder below is therefore *likely needed*, in order: (1) seeds 3→2 on one model (−64 h); (2) candidates/round 2000→1200 (−15% gen+filter); (3) E6 cut entirely; (4) E4 on 1 model short-form; (5) E3-shrink to {self, human, cross_q2l}; (6) if a second node exists, split models across nodes (halves wall-clock).

## 8. Day-by-day calendar (anchor: cluster live morning of "Day 5")

| Day | Target |
|---|---|
| 5 | Prep + P0 baselines; launch G1 overnight |
| 6–7 | G1 completes; **gate decision meeting**; re-budget §6.4 |
| 7–13 | E1 grid block A (llama × 4 cond × 3 seeds) — C2 audit after run 8 |
| 13–19 | E1 block B (qwen) — stats rollup each evening |
| 19–21 | E5 + E3 corpora |
| 21–24 | E3 runs + E4 ablations |
| 24–26 | buffer /reruns/lost days; E6 if on schedule |
| 26–31 | analysis, figures, manuscript, AC self-audit, submit |

## 9. Failure Playbook

| Symptom | Action |
|---|---|
| OOM mid-train | halve `train_micro_batch_size` (config), relaunch (resume skips done rounds); NEVER patch code on the fly mid-grid |
| NaN twice → engine abort | lower LR 2e-5→1e-5 for that run only, tag `lr1e5` in name, note in run-sheet |
| `accepted` < 25 for 3 consecutive rounds | condition-specific alert → decide LR/rounds fix at grid level, pause that condition |
| Dataset count drift (assertion fires) | STOP that run; verify HF snapshot; pin `revision=` in loader; re-run affected |
| Guard model REST/download flake | pre-cached weights only, network egress only for datasets once |
| Node loss mid-run | relaunch same command (resume via `.done` sentinels); log incident in run-sheet |
| Multi-node available | replicate node image; run one model per node; shared scratch only for results/logs |

## 10. Hard "NEVER" list (paper-integrity guards)

1. Never run base (non-Instruct) checkpoints for paper numbers.
2. Never cap math eval below 1,319 (`min(200,…)` bug is dead forever).
3. Never pool XSTest safe+unsafe; never length-heuristic ASR.
4. Never hand-edit a YAML per run — `make_grid.py` output only.
5. Never train on E3 corpus without its provenance header (teacher id, acceptance-match stats, sha256).
6. No harmful generations in repo/WandB; judged labels only.
7. If τ calibration fails the 90%-gold-pass constraint → run without subliminal gate and report honestly; do NOT silently retune mid-grid.

## 11. Phase-exit checklists

- [ ] P0 done ⇒ two `round0_baseline.json` files pass sanity vs known-good ranges
- [ ] G1 ⇒ decision recorded in `meta/g1_decision.md` with numbers
- [ ] E1 ⇒ 24 `metrics.json` + complete logs; rollup table reproducible by `scripts/stats_rollup.py`
- [ ] E3/E5/E4 ⇒ per-arm summaries + τ-calibration ROC artifacts
- [ ] Handoff ⇒ `results_rollup/` (csv + figs), seeds manifest, then manuscript numbers ONLY from rollup

---
*Authored as plan-only. Nothing above has been executed; execution begins on your explicit go.*
