#!/usr/bin/env bash
# sync_results_github.sh — push DGX logs/results artifacts to github.com/Aadisharma1/Conti.
#
# Usage:
#   CONTI_GITHUB_TOKEN=<pat> bash scripts/sync_results_github.sh <label>
#   e.g. CONTI_GITHUB_TOKEN=github_pat_xxxx bash scripts/sync_results_github.sh pilot
#
# What gets pushed (lightweight provenance artifacts ONLY, stamped by UTC time):
#   outputs/**/*.json, *.jsonl   (metrics, filter/latent/drift logs, manifests, baselines)
#   results_rollup/*             (rollup.csv, summary.md, if present)
# What NEVER gets pushed: checkpoints, model weights (*.safetensors/*.bin/*.pt),
# .env files, or anything > 5 MB. Checkpoints stay on cluster scratch.
#
# The token is used transiently in the remote URL and scrubbed from .git/config
# when the sync finishes (success or failure). Provide it via env var, not argv,
# when possible so it stays out of the shell history.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GITHUB_REPO_URL="https://github.com/Aadisharma1/Conti.git"
LABEL="${1:-manual}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ART_DIR="results/artifacts/${STAMP}_${LABEL}"

if [ -z "${CONTI_GITHUB_TOKEN:-}" ]; then
  echo "[sync][FAIL] CONTI_GITHUB_TOKEN is not set." >&2
  echo "  usage: CONTI_GITHUB_TOKEN=<pat> bash scripts/sync_results_github.sh <label>" >&2
  exit 1
fi

cleanup_remote() {
  git remote set-url origin "${GITHUB_REPO_URL}" 2>/dev/null || true
}
trap cleanup_remote EXIT

if [ ! -d "outputs" ] && [ ! -d "results_rollup" ]; then
  echo "[sync] nothing to sync (no outputs/ or results_rollup/ found); skipping"
  exit 0
fi

mkdir -p "${ART_DIR}"

COPIED=0
if [ -d "outputs" ]; then
  while IFS= read -r -d '' file; do
    cp --parents "${file}" "${ART_DIR}/"
    COPIED=$((COPIED + 1))
  done < <(
    find outputs -type f \
      \( -name '*.json' -o -name '*.jsonl' \) \
      ! -path '*checkpoints*' \
      ! -name '*.safetensors' ! -name '*.bin' ! -name '*.pt' \
      ! -name '.env' \
      -size -5M -print0
  )
fi

if [ -d "results_rollup" ]; then
  mkdir -p "${ART_DIR}/results_rollup"
  cp -r results_rollup/. "${ART_DIR}/results_rollup/"
  COPIED=$((COPIED + $(find results_rollup -type f | wc -l)))
fi

if [ "${COPIED}" -eq 0 ]; then
  echo "[sync] no artifact files found under outputs/ or results_rollup/; nothing to commit"
  exit 0
fi

echo "[sync] staged ${COPIED} artifact files into ${ART_DIR}"

git config user.name  "${GIT_AUTHOR_NAME:-conti-dgx}"
git config user.email "${GIT_AUTHOR_EMAIL:-conti-dgx@users.noreply.github.com}"

git remote set-url origin "https://x-access-token:${CONTI_GITHUB_TOKEN}@github.com/Aadisharma1/Conti.git"

git add "${ART_DIR}"
if git diff --cached --quiet; then
  echo "[sync] no changes to commit (artifacts identical to last sync)"
  exit 0
fi

git commit -m "results: ${LABEL} @ ${STAMP} (${COPIED} files from DGX)"

git pull --rebase origin "${BRANCH}" || echo "[sync][warn] rebase against origin/${BRANCH} failed; attempting direct push"

git push origin "${BRANCH}"

echo "[sync] DONE -> github.com/Aadisharma1/Conti (${BRANCH}) ${ART_DIR}"
