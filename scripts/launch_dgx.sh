#!/usr/bin/env bash
# launch_dgx.sh — Day-5 cluster prep + G1 pilot launcher for the 8x A100 DGX.
#
# Usage:
#   bash scripts/launch_dgx.sh prep          # env check + precache + sanity + round-0 baselines
#   bash scripts/launch_dgx.sh pilot         # G1 pilot (naive STaR, llama, seed 42, 10 rounds)
#   bash scripts/launch_dgx.sh grid          # launch the full generated E1 grid sequentially
#   bash scripts/launch_dgx.sh CONFIG.yaml   # launch one specific generated config
#
# Requirements (docs/plan.md section 2): HF_TOKEN exported, repo on $SCRATCH,
# torchrun available. Every step fails fast (set -euo pipefail).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-4}"  # 4 GPUs is the standing allocation; export NPROC=8 when the full node is granted

fail() { echo "[launch_dgx][FAIL] $*" >&2; exit 1; }

maybe_sync() {
  local label="$1"
  if [ -n "${CONTI_GITHUB_TOKEN:-}" ]; then
    echo "[launch_dgx] syncing results to GitHub (label: ${label})"
    bash scripts/sync_results_github.sh "${label}" \
      || echo "[launch_dgx][warn] results sync failed; cluster work continues, sync manually later"
  else
    echo "[launch_dgx] CONTI_GITHUB_TOKEN not set; skipping results sync"
  fi
}

require_env() {
  command -v python >/dev/null || fail "python not on PATH"
  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found (not the DGX?)"
  [ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not exported (gated models will not load)"
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
  echo "GPUs visible: ${GPU_COUNT}"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable inside python"
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| gpus:", torch.cuda.device_count())
PY
}

step_prep() {
  require_env
  python -m pip install -e ".[dev]" -q
  python scripts/precache_data.py
  python scripts/precache_models.py
  torchrun --nproc_per_node="${NPROC}" scripts/sanity_compute.py
  python scripts/make_grid.py
  python scripts/eval_round0.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output outputs/round0/llama3-8b-instruct
  python scripts/eval_round0.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output outputs/round0/qwen2.5-7b-instruct
  echo "[prep] DONE — checkpoint C0/C1 reached. Inspect round0_baseline.json before launching the pilot."
  maybe_sync "prep"
}

step_pilot() {
  require_env
  CONFIG="${REPO_ROOT}/configs/generated/e1_star_only_llama3_s42.yaml"
  [ -f "${CONFIG}" ] || fail "missing ${CONFIG}; run 'launch_dgx.sh prep' first"
  torchrun --nproc_per_node="${NPROC}" scripts/run_experiment.py --config "${CONFIG}"
  echo "[pilot] DONE — Gate G1 decision: inspect outputs/e1/star_only/llama3/42/metrics.json and logs/latent_filter.jsonl"
  maybe_sync "pilot"
}

step_grid() {
  require_env
  for CONFIG in "${REPO_ROOT}"/configs/generated/e1_*.yaml; do
    echo "[grid] launching ${CONFIG}"
    torchrun --nproc_per_node="${NPROC}" scripts/run_experiment.py --config "${CONFIG}"
    maybe_sync "grid_$(basename "${CONFIG}" .yaml)"
  done
  python scripts/stats_rollup.py --outputs-root outputs --rollup-dir results_rollup
  maybe_sync "grid_rollup"
  echo "[grid] DONE — results_rollup/summary.md has the multi-seed table"
}

step_config() {
  local CONFIG="$1"
  [ -f "${CONFIG}" ] || fail "config not found: ${CONFIG}"
  torchrun --nproc_per_node="${NPROC}" scripts/run_experiment.py --config "${CONFIG}"
  maybe_sync "run_$(basename "${CONFIG}" .yaml)"
}

step_sync() {
  maybe_sync "${2:-manual}"
}

case "${1:-}" in
  prep)   step_prep ;;
  pilot)  step_pilot ;;
  grid)   step_grid ;;
  sync)   step_sync "$@" ;;
  *)      step_config "$1" ;;
esac
