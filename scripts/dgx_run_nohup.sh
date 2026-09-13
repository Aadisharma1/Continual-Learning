#!/usr/bin/env bash
# dgx_run_nohup.sh — unattended nohup supervisor for long DGX runs.
#
# Usage (from the repo root on the DGX):
#   export HF_TOKEN=hf_xxx
#   CONTI_GITHUB_TOKEN=github_pat_xxx nohup bash scripts/dgx_run_nohup.sh pilot \
#       > outputs/nohup_pilot_console.log 2>&1 &
#
# Modes:
#   pilot            -> launch_dgx.sh pilot   (G1 gate run)
#   grid             -> launch_dgx.sh grid    (full E1 grid, sync after every run)
#   <config path>    -> launch_dgx.sh <config>
#
# What the watchdog does:
#   * starts the training step in its OWN PROCESS GROUP (setsid) so it can be
#     reaped cleanly — no dead runs squatting on shared GPUs;
#   * appends a heartbeat (GPU util/mem, aliveness) to
#     outputs/nohup/<tag>/heartbeat.jsonl every SYNC_INTERVAL_SECONDS
#     (default 1800) and pushes ALL current artifacts to GitHub — partial logs
#     land remotely long before the run ends, so mishaps are visible from the
#     commit history alone;
#   * when training exits: final push labeled  done_<mode>  or
#     crashed_<mode>_exit<code>, then SIGTERM/SIGKILL the whole process group
#     and log post-kill nvidia-smi to prove the GPUs are actually free.
#
# GPU count: NPROC (default 4; export NPROC=8 if you get the whole node).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-4}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-1800}"
MODE="${1:?usage: bash scripts/dgx_run_nohup.sh <pilot|grid|config.yaml>}"

if [ -z "${CONTI_GITHUB_TOKEN:-}" ]; then
  echo "[nohup][FAIL] CONTI_GITHUB_TOKEN is not set — unattended runs must sync." >&2
  exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[nohup][FAIL] HF_TOKEN is not set — gated models will not load." >&2
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_TAG="${STAMP}_${MODE}"
RUN_DIR="outputs/nohup/${RUN_TAG}"
HEARTBEAT="${RUN_DIR}/heartbeat.jsonl"
mkdir -p "${RUN_DIR}"

echo "[nohup] ${RUN_TAG} starting: mode=${MODE} nproc=${NPROC} sync_every=${SYNC_INTERVAL_SECONDS}s"

# --- launch training in its own process group (PGID == child PID) ----------- #
setsid env NPROC="${NPROC}" CONTI_GITHUB_TOKEN="${CONTI_GITHUB_TOKEN}" \
  bash scripts/launch_dgx.sh "${MODE}" \
  > "${RUN_DIR}/train.log" 2>&1 &
TRAIN_PID=$!

gpu_snapshot() {
  nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}

append_heartbeat() {
  local phase="$1" exit_code="${2:-}"
  local ts child_alive group_alive
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then child_alive=true; else child_alive=false; fi
  if kill -0 -- -"${TRAIN_PID}" 2>/dev/null; then group_alive=true; else group_alive=false; fi
  {
    printf '{"ts":"%s","phase":"%s","exit_code":"%s","mode":"%s","nproc":"%s","child_alive":%s,"group_alive":%s,"gpu":"' \
      "${ts}" "${phase}" "${exit_code}" "${MODE}" "${NPROC}" "${child_alive}" "${group_alive}"
    gpu_snapshot | tr '\n' ';' | sed 's/"/\\"/g'
    printf '"}\n'
  } >> "${HEARTBEAT}"
}

kill_process_group() {
  # Reap EVERYTHING in the training group: torchrun, its worker ranks, any
  # accelerate/nccl daemons. Shared GPUs must not keep a corpse warm.
  kill -TERM -- -"${TRAIN_PID}" 2>/dev/null || true
  sleep 10
  if kill -0 -- -"${TRAIN_PID}" 2>/dev/null; then
    echo "[nohup] process group still alive after SIGTERM; sending SIGKILL" >> "${RUN_DIR}/train.log"
    kill -KILL -- -"${TRAIN_PID}" 2>/dev/null || true
    sleep 3
  fi
  append_heartbeat "post_kill_cleanup"
}

# --- initial sync so the run's existence is on GitHub from minute one ------- #
bash scripts/sync_results_github.sh "start_${MODE}" \
  || echo "[nohup][warn] initial sync failed; continuing"

# --- watchdog loop ----------------------------------------------------------- #
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep "${SYNC_INTERVAL_SECONDS}"
  if kill -0 "${TRAIN_PID}" 2>/dev/null; then
    append_heartbeat "running"
    bash scripts/sync_results_github.sh "hb_${MODE}" \
      || echo "[nohup][warn] heartbeat sync failed; will retry next interval"
  fi
done

# --- training exited: reap, label, final push -------------------------------- #
set +e
wait "${TRAIN_PID}"
EXIT_CODE=$?
set -e

kill_process_group

if [ "${EXIT_CODE}" -eq 0 ]; then
  echo "[nohup] training completed cleanly; final sync"
  FINAL_LABEL="done_${MODE}"
else
  echo "[nohup][FAIL] training exited with code ${EXIT_CODE}; pushing crash state" >&2
  FINAL_LABEL="crashed_${MODE}_exit${EXIT_CODE}"
fi
append_heartbeat "final" "${EXIT_CODE}"
bash scripts/sync_results_github.sh "${FINAL_LABEL}" \
  || echo "[nohup][warn] FINAL sync failed — run locally: CONTI_GITHUB_TOKEN=... bash scripts/sync_results_github.sh ${FINAL_LABEL}"

echo "[nohup] ${RUN_TAG} finished with exit code ${EXIT_CODE}; GPUs reaped; artifacts on GitHub (${FINAL_LABEL})"
exit "${EXIT_CODE}"
