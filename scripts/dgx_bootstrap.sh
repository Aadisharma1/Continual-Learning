#!/usr/bin/env bash
# dgx_bootstrap.sh — ONE-command unattended pipeline: prep -> pilot -> gate check -> grid.
#
# Set the secrets ONCE in your shell, then fire and walk away:
#
#   cd $SCRATCH/Conti
#   export HF_TOKEN=hf_xxx
#   export CONTI_GITHUB_TOKEN=github_pat_xxx
#   export NPROC=4                          # optional, default 4
#   nohup bash scripts/dgx_bootstrap.sh > outputs/bootstrap_console.log 2>&1 &
#
# Behavior:
#   * prep   : env checks, dataset/weight caches, NCCL sanity, grid generation,
#              round-0 baselines for both Instruct models (syncs to GitHub).
#   * pilot  : G1 gate run (naive STaR x 10 rounds, telemetry) under the nohup
#              watchdog (30-min heartbeats + crash labeling + GPU reaping).
#   * GATE   : auto-checks aggregate judged/proxy ASR drift (final vs round-0).
#              drift >= G1_MIN_DRIFT (default 0.03) -> continue to grid.
#              otherwise STOP and push a 'gate_hold' sync (set AUTO_CONTINUE=1
#              to bypass and always run the grid).
#   * grid   : full E1 grid (sync after every run) + final stats rollup.
#
# Everything inherits the exported tokens; you never type them again.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

G1_MIN_DRIFT="${G1_MIN_DRIFT:-0.03}"
AUTO_CONTINUE="${AUTO_CONTINUE:-0}"
PILOT_DIR="outputs/e1/star_only/llama3/42"

fail() { echo "[bootstrap][FAIL] $*" >&2; exit 1; }
[ -n "${CONTI_GITHUB_TOKEN:-}" ] || fail "CONTI_GITHUB_TOKEN not exported"
[ -n "${HF_TOKEN:-}" ]           || fail "HF_TOKEN not exported"

echo "[bootstrap] phase 1/3: prep"
bash scripts/launch_dgx.sh prep

echo "[bootstrap] phase 2/3: pilot (G1 gate run)"
if ! bash scripts/launch_dgx.sh pilot; then
  echo "[bootstrap][FAIL] pilot crashed — crash state already synced; inspect GitHub artifacts" >&2
  exit 1
fi

echo "[bootstrap] gate G1: measuring aggregate ASR drift"
DRIFT="$(python - "$PILOT_DIR" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "logs" / "round_metrics.jsonl"
if not path.exists():
    print("nan"); raise SystemExit
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
baseline = [r for r in rows if r.get("round") == -1]
finals = [r for r in rows if r.get("eval_mode") in ("full", "final") and r.get("round", -1) >= 0]
if not baseline or not finals:
    print("nan"); raise SystemExit
base = baseline[-1]["safety"]["aggregate_unsafe"]["asr"]
final = finals[-1]["safety"]["aggregate_unsafe"]["asr"]
print(f"{final - base:.4f}")
PY
)"
echo "[bootstrap] gate G1: aggregate ASR drift = ${DRIFT} (threshold ${G1_MIN_DRIFT})"

if [ "${DRIFT}" = "nan" ] || python -c "import sys; sys.exit(0 if float('${DRIFT}') >= float('${G1_MIN_DRIFT}')" 2>/dev/null; then
  :
else
  if [ "${AUTO_CONTINUE}" != "1" ]; then
    echo "[bootstrap] GATE HOLD: drift below threshold — grid NOT launched."
    echo "  inspect outputs/e1/star_only/llama3/42, then either re-run manually:"
    echo "    bash scripts/dgx_run_nohup.sh grid   (with tokens exported)"
    echo "  or export AUTO_CONTINUE=1 to skip this gate in future."
    bash scripts/sync_results_github.sh "gate_hold" || true
    exit 42
  fi
  echo "[bootstrap] AUTO_CONTINUE=1 — continuing despite gate result"
fi

echo "[bootstrap] phase 3/3: full E1 grid"
bash scripts/launch_dgx.sh grid

echo "[bootstrap] ALL PHASES COMPLETE"
