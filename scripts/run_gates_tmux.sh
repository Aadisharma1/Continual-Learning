#!/usr/bin/env bash
# Launch the G7/G8 gate pipeline inside a detached tmux session so it survives
# closing the SSH connection.
#
#   ./scripts/run_gates_tmux.sh            # launch (default session: gates)
#   ./scripts/run_gates_tmux.sh mysession  # launch under a different name
#
# Idempotent: refuses to start a second run against the same repo, because two
# concurrent runs would fight over the same outputs/ and results/artifacts/
# trees and corrupt both.
#
# Designed for a real interactive SSH session. A detached tmux pane has a TTY
# that nobody is watching, so a secret prompt there would hang forever. This
# script therefore runs the secret capture in the FOREGROUND first, verifies it,
# and only then detaches.

set -euo pipefail

SESSION="${1:-gates}"
CONFIG="${GATES_CONFIG:-configs/gates/g7g8_llama31_8b.json}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say()  { printf '\033[1;34m[gates]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gates]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[gates]\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------ preconditions
command -v tmux >/dev/null 2>&1 || die "tmux is not installed: sudo apt-get install -y tmux"
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
[ -f gates/runall.py ] || die "gates/runall.py not found; wrong directory?"
git rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository"
[ -f .env ] || die "no .env yet -- run:  python -m gates.runall --config $CONFIG --secrets-only"

# A previous run still attached? Refusing is safer than two writers on one tree.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  die "tmux session '$SESSION' already exists.
  attach:  tmux attach -t $SESSION
  or kill: tmux kill-session -t $SESSION"
fi

# Another session under a different name would still share outputs/ on disk.
for s in $(tmux list-sessions -F '#{session_name}' 2>/dev/null || true); do
  case "$s" in
    *gates*) die "a gates session is already running as '$s'. Stop it first:
  tmux attach -t $s     # inspect
  tmux kill-session -t $s"
  esac
done

mkdir -p outputs/runner/logs

# ------------------------------------------------------------ secrets check
# Runs in the foreground on purpose: this is the only moment a human can
# actually answer the prompt.
say "checking secrets (foreground, this may prompt once)"
if ! python -m gates.runall --config "$CONFIG" --secrets-only; then
  die "secrets incomplete. Fix the keys above, then re-run this script."
fi

# ------------------------------------------------------------ GPU + env check
say "preflight"
if ! python -m gates.runall --config "$CONFIG" --dry-run --yes; then
  die "preflight failed. Fix the environment before launching."
fi

# ------------------------------------------------------------ launch
RUNNER_LOG="outputs/runner/logs/tmux_${SESSION}.log"
say "launching detached session '$SESSION'"
say "console -> $RUNNER_LOG"

# --yes: never prompt inside the detached pane. Anything missing was already
# caught by the two checks above.
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
  "exec python -m gates.runall --config '$CONFIG' --yes 2>&1 | tee -a '$RUNNER_LOG'"

sleep 3
if tmux has-session -t "$SESSION" 2>/dev/null; then
  say "running. you can close this ssh connection now."
else
  die "session died immediately. Check: tail -n 60 '$RUNNER_LOG'"
fi

cat <<EOF

  attach      tmux attach -t $SESSION          (detach: Ctrl-b, then d)
  watch log   tail -f $RUNNER_LOG
  health      cat outputs/runner/health.json
  gpu         watch -n5 nvidia-smi
  results     git log --oneline -5
  stop        tmux kill-session -t $SESSION

EOF
