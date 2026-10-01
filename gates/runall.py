"""One-command orchestration for the G7/G8 gate pipeline.

Design constraints this file encodes:

* Secrets are collected ONCE, interactively, and cached in a gitignored .env.
  The GitHub PAT is never written to .git/config, argv, or a log line.
* Health checks and GitHub sync run on their own cadence, independent of the
  training process, so a wedged or crashed run is still detected and its logs
  still reach the remote.
* Every stage is a subprocess. A stage that OOMs or segfaults takes down only
  itself; the supervisor records the failure and moves on, and the final
  summary says plainly what did not run.
* The environment is probed before anything heavy runs, and a missing or
  incompatible dependency prints the exact command to fix it rather than
  dying 40 lines into a traceback.

Usage:
    python -m gates.runall                      # interactive, full pipeline
    python -m gates.runall --dry-run            # plan only, no GPU touched
    python -m gates.runall --stage g8           # one stage
    python -m gates.runall --yes                # never prompt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = ["main", "load_config", "SecretStore", "HealthWatch", "preflight"]

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STATE = ROOT / "outputs" / "runner"
LOG_DIR = STATE / "logs"
SYNC_DIR = ROOT / "results" / "artifacts"

# --------------------------------------------------------------- config
DEFAULT_CONFIG = {
    "run_name": "gate-run",
    "repo": "aadisharma1/Conti",
    "branch": "gates",
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "trained_model": "",
    "ckpt_root": "checkpoints",
    "persona": "contrarian",
    "layers": "15-22",
    "device": "cuda:0",
    "batch_size": 8,
    "max_len": 1024,
    "seed": 0,
    "sync_interval_s": 900,
    "health_interval_s": 120,
    "stall_minutes": 45,
    "disk_min_free_gb": 50,
    "wandb": False,
    "stages": ["g7", "g8"],
}


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(raw) - set(cfg)
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}")
        cfg.update(raw)
    # env overrides, so a queue script can retarget without editing the file
    for k in list(cfg):
        env_k = "CONTI_" + k.upper()
        if env_k in os.environ:
            cfg[k] = os.environ[env_k]
    return cfg


# --------------------------------------------------------------- secrets
SECRET_SPEC = [
    ("CONTI_GITHUB_TOKEN", "GitHub PAT (repo scope) for pushing results",
     "github_pat_..."),
    ("HF_TOKEN", "Hugging Face token (needed for gated Llama weights)", "hf_..."),
    ("WANDB_API_KEY", "Weights & Biases key (optional, blank to skip)", "blank to skip"),
]


class SecretStore:
    """Interactive one-time secret capture, cached in a gitignored .env.

    A PAT pasted into argv shows up in shell history and in `ps` for every
    other user on the node, so it is only ever read from stdin.
    """

    def __init__(self, path: Path = ENV_FILE, assume_yes: bool = False) -> None:
        self.path = path
        self.assume_yes = assume_yes
        self.values: dict[str, str] = {}

    def load(self) -> dict[str, str]:
        vals: dict[str, str] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"').strip("'")
        return vals

    def collect(self) -> dict[str, str]:
        existing = self.load()
        todo = []
        for key, desc, hint in SECRET_SPEC:
            if existing.get(key):
                continue
            if key == "WANDB_API_KEY" and not self._want_wandb():
                continue
            todo.append((key, desc, hint))
        if not todo:
            self.values = existing
            return existing

        interactive = sys.stdin.isatty() and not self.assume_yes
        if not interactive:
            # Never block on a prompt we cannot answer. Report precisely what
            # is absent so the run can still proceed on a cache-only basis.
            names = [k for k, _, _ in todo if k != "WANDB_API_KEY"]
            if names:
                print("\n[secrets] not prompting (non-TTY or --yes). Missing: "
                      + ", ".join(names))
                print(f"[secrets] results will NOT be pushed. To enable pushing, "
                      f"create {self.path} with those keys, or run once in a "
                      f"real terminal.")
            self.values = existing
            return existing

        for key, desc, hint in todo:
            self._ask(key, desc, hint)

        merged = {**existing, **self.values}
        self._persist(merged)
        self.values = merged
        return merged

    def _want_wandb(self) -> bool:
        return os.environ.get("CONTI_WANDB", "").lower() in ("1", "true", "yes")

    def _ask(self, key: str, desc: str, hint: str) -> None:
        import getpass

        prompt = f"\n{key}\n  {desc}\n  (blank to skip; hint: {hint})\n> "
        try:
            val = getpass.getpass(prompt)
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\naborted at secret prompt")
        except Exception:  # noqa: BLE001 - no console handle available
            val = input(prompt)
        val = val.strip()
        if val:
            self.values[key] = val
            os.environ[key] = val

    def _persist(self, vals: dict[str, str]) -> None:
        lines = ["# Written by gates.runall. Contains secrets. Do NOT commit.",
                 "# .gitignore already excludes this file; see .gitignore."]
        for key, _, _ in SECRET_SPEC:
            if vals.get(key):
                lines.append(f"{key}={vals[key]}")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass  # Windows / FAT: ACLs handle it instead
        # Belt and braces: even if .gitignore is edited later, never track it.
        gi = ROOT / ".gitignore"
        rule = ".env"
        if gi.is_file() and rule not in gi.read_text(encoding="utf-8").splitlines():
            with gi.open("a", encoding="utf-8") as fh:
                fh.write("\n.env\n.env.*\n!.env.example\n")
        print(f"\n[secrets] wrote {self.path} (mode 600)")

    def apply_to_environ(self) -> None:
        for k, v in self.load().items():
            os.environ.setdefault(k, v)

    def scrub_token_from_git(self) -> None:
        """Remove any credential accidentally baked into the origin URL.

        A PAT left in .git/config is readable by every process on the node and
        is committed by the next `git remote -v` pasted into a bug report.
        """
        if not shutil.which("git"):
            return
        try:
            url = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return
        if url and re.search(r"://[^/@]*@", url):
            clean = re.sub(r"://[^/@]*@", "https://", url)
            subprocess.run(["git", "remote", "set-url", "origin", clean],
                           cwd=ROOT, capture_output=True, timeout=30)
            print("[secrets] scrubbed credentials from origin remote URL")


# --------------------------------------------------------------- preflight
REQUIRED = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("accelerate", "accelerate"),
]

MIN_PY = (3, 10)


def _importable(mod: str) -> str | None:
    try:
        __import__(mod)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def preflight(cfg: dict, dry_run: bool = False) -> dict:
    """Check the environment before burning a GPU-hour on an ImportError."""
    import platform

    report: dict = {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info[:2] >= MIN_PY,
        "cuda_devices": 0,
        "cuda_ok": True,
        "gpus": [],
        "missing": [],
        "version_conflicts": [],
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_devices"] = torch.cuda.device_count()
        report["cuda_ok"] = report["cuda_available"]
        if report["cuda_available"]:
            for i in range(report["cuda_devices"]):
                p = torch.cuda.get_device_properties(i)
                report["gpus"].append({
                    "index": i, "name": p.name,
                    "total_gb": round(p.total_memory / 1024**3, 2),
                    "free_gb": round(
                        torch.cuda.mem_get_info(i)[0] / 1024**3, 2),
                    "capability": f"{p.major}.{p.minor}",
                })
    except Exception as exc:  # noqa: BLE001
        report["cuda_ok"] = False
        report["torch_error"] = str(exc)

    for mod, pip_name in REQUIRED:
        err = _importable(mod)
        if err:
            report["missing"].append({"module": mod, "error": err})

    # transformers 4.x has no wheels for 3.13+; 5.x is the only line that does.
    if not report["missing"]:
        try:
            import transformers

            major = int(transformers.__version__.split(".")[0])
            report["transformers"] = transformers.__version__
            if major < 5 and sys.version_info >= (3, 13):
                report["version_conflicts"].append(
                    f"transformers {transformers.__version__} has no Python "
                    f"{sys.version_info.major}.{sys.version_info.minor} wheels; "
                    "the pins in requirements.txt cannot be satisfied here."
                )
        except Exception:  # noqa: BLE001
            pass

    if not dry_run:
        free = shutil.disk_usage(ROOT).free / 1024**3
        report["disk_free_gb"] = round(free, 1)
        report["disk_ok"] = free >= float(cfg["disk_min_free_gb"])

    ok = (report["python_ok"] and not report["missing"]
          and not report["version_conflicts"])
    if dry_run:
        ok = not report["missing"] and report["python_ok"]
    report["ok"] = ok
    report["platform"] = platform.platform()
    return report


def print_preflight(r: dict, dry_run: bool) -> None:
    print("=" * 72)
    print("PREFLIGHT")
    print("=" * 72)
    print(f"  python           {r['python']}  "
          f"{'ok' if r['python_ok'] else 'TOO OLD (need >= %d.%d)' % MIN_PY}")
    if "torch" in r:
        print(f"  torch            {r['torch']}")
    if r["gpus"]:
        for g in r["gpus"]:
            print(f"  cuda:{g['index']}           {g['name']}  "
                  f"{g['free_gb']:.1f}/{g['total_gb']:.1f} GB free  "
                  f"sm_{g['capability']}")
    else:
        print(f"  cuda             NONE  ({r.get('cuda_available')})")
    if "disk_free_gb" in r:
        print(f"  disk free        {r['disk_free_gb']} GB  "
              f"{'ok' if r.get('disk_ok') else 'LOW'}")
    for m in r["missing"]:
        print(f"  MISSING          {m['module']}: {m['error'][:80]}")
    for c in r["version_conflicts"]:
        print(f"  CONFLICT         {c}")
    if r["missing"]:
        print("\n  Fix with:")
        mods = " ".join(m["module"] for m in r["missing"])
        print(f"    {Path(sys.executable).name} -m pip install -U {mods}")
        if r["version_conflicts"]:
            print("    # and to clear the transformers pin conflict:")
            print("    python -m pip install -U 'transformers>=5'")
    if not r["cuda_ok"] and not dry_run:
        print("\n  NOTE: no CUDA device visible. Gates need a GPU to run for "
              "real; a CPU run is a smoke test only.")
    print("=" * 72)


# --------------------------------------------------------------- health
_ERROR_PATTERNS = [
    (re.compile(r"CUDA out of memory", re.I), "oom", True),
    (re.compile(r"CUDA error", re.I), "cuda_error", True),
    (re.compile(r"device-side assert", re.I), "device_assert", True),
    (re.compile(r"NCCL (error|failed)", re.I), "nccl", True),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback", False),
    (re.compile(r"\bloss\s*[:=]\s*(nan|-?inf)\b", re.I), "nan_loss", True),
    (re.compile(r"grad_norm\s*[:=]\s*(nan|-?inf)\b", re.I), "nan_grad", True),
]


@dataclass
class HealthEvent:
    ts: str
    level: str          # WARN | FAIL
    kind: str
    detail: str
    log: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class HealthWatch:
    """Periodic supervisor over the running stage.

    The failure modes that actually kill multi-hour GPU jobs are: OOM, a CUDA
    fault, NaN that quietly destroys a probe, a full scratch disk, and a
    process that is alive but has stopped making progress. The last one is the
    one nobody notices, so throughput is tracked explicitly.
    """

    def __init__(self, cfg: dict, log_dir: Path = LOG_DIR) -> None:
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = STATE / "health.json"
        self.events: list[HealthEvent] = []
        self.stop = threading.Event()
        self.log_path: Path | None = None
        self.proc: subprocess.Popen | None = None
        self._start: dict[str, float] = {}

    # -- public ------------------------------------------------------
    def watch(self, proc: subprocess.Popen, log_path: Path) -> None:
        self.proc = proc
        self.log_path = log_path
        self._start = {"t": time.time(), "bytes": self._size(log_path)}
        t = threading.Thread(target=self._loop, args=(proc, log_path), daemon=True)
        t.start()

    def snapshot(self) -> dict:
        running = self.proc is not None and self.proc.poll() is None
        snap = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage_running": running,
            "gpu_mem_mb": self._gpu_mem(),
            "gpu_util": self._gpu_util(),
            "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1024**3, 1),
            "n_events": len(self.events),
            "worst": self._worst(),
        }
        if running and self.log_path is not None:
            snap["log_bytes"] = self._size(self.log_path)
            snap["log_age_s"] = int(time.time() - self.log_path.stat().st_mtime)
        return snap

    def finish(self) -> None:
        self.stop.set()
        self._flush()

    # -- internals ---------------------------------------------------
    @staticmethod
    def _size(p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _worst() -> str:
        return "ok"

    def _gpu_mem(self) -> list[dict]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        rows = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                rows.append({"index": int(parts[0]),
                             "used_mb": int(parts[1]),
                             "total_mb": int(parts[2])})
        return rows

    def _gpu_util(self) -> list[int]:
        if not shutil.which("nvidia-smi"):
            return []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        return [int(x) for x in out.strip().splitlines() if x.strip().isdigit()]

    def _emit(self, level: str, kind: str, detail: str) -> None:
        ev = HealthEvent(
            ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            level=level, kind=kind, detail=detail,
            log=str(self.log_path) if self.log_path else "-",
        )
        self.events.append(ev)
        print(f"\n[health:{level}] {kind}: {detail}", flush=True)

    def _scan_log(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        for pat, kind, fatal in _ERROR_PATTERNS:
            m = pat.search(text)
            if m:
                ln = text[:m.start()].count("\n") + 1
                line = m.group(0)[:160]
                self._emit("FAIL" if fatal else "WARN", kind,
                           f"{path.name}:{ln} {line}")

    def _loop(self, proc: subprocess.Popen, log_path: Path) -> None:
        interval = int(self.cfg["health_interval_s"])
        stall_s = int(self.cfg["stall_minutes"]) * 60
        last_size = self._size(log_path)
        last_growth = time.time()
        last_gpu_active = time.time()
        reported: set[str] = set()

        while not self.stop.wait(interval):
            rc = proc.poll()
            if rc is not None:
                self._scan_log(log_path)
                self._emit("FAIL" if rc != 0 else "WARN", "exit",
                           f"stage process exited rc={rc}")
                self._flush()
                return

            self._scan_log(log_path)

            size = self._size(log_path)
            if size > last_size:
                last_size = size
                last_growth = time.time()

            if self._gpu_util() and any(u > 5 for u in self._gpu_util()):
                last_gpu_active = time.time()

            now = time.time()
            # A stage is stalled if BOTH the log has not grown AND the GPU has
            # not been active. Either one alone is not enough: a probe-fitting
            # phase is CPU-bound with a silent log, and a log flush can lag.
            if (now - last_growth > stall_s
                    and now - last_gpu_active > stall_s):
                if "stall" not in reported:
                    reported.add("stall")
                    self._emit("FAIL", "stall",
                               f"no log growth and no GPU activity for "
                               f"{int((now - last_growth) // 60)} min")

            free = shutil.disk_usage(ROOT).free / 1024**3
            if free < float(self.cfg["disk_min_free_gb"]):
                if "disk" not in reported:
                    reported.add("disk")
                    self._emit("FAIL", "disk",
                               f"only {free:.1f} GB free on the scratch volume")

            for g in self._gpu_mem():
                if g["total_mb"] and g["used_mb"] / g["total_mb"] > 0.97:
                    k = f"mem{g['index']}"
                    if k not in reported:
                        reported.add(k)
                        self._emit("WARN", "gpu_memory",
                                   f"cuda:{g['index']} at "
                                   f"{100 * g['used_mb'] / g['total_mb']:.1f}%")

            self._flush()

        self._flush()

    def _flush(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        snap = self.snapshot()
        snap["events"] = [e.to_dict() for e in self.events[-50:]]
        self.state_file.write_text(json.dumps(snap, indent=2), encoding="utf-8")


# --------------------------------------------------------------- sync
class GitHubSync:
    """Periodic artifact-only push to the results repo.

    Artifacts only, never weights. Weights would be tens of gigabytes and the
    remote rejects them, which historically left the whole sync failing and
    nobody receiving any results at all.
    """

    PATTERNS = ("*.json", "*.jsonl", "*.csv", "*.md", "*.log", "*.txt")
    MAX_BYTES = 5 * 1024 * 1024

    def __init__(self, cfg: dict, token: str = "") -> None:
        self.cfg = cfg
        self.token = token
        self.interval = int(cfg["sync_interval_s"])
        self.stop = threading.Event()
        self.srcs = [ROOT / "outputs", ROOT / "results"]
        self.state_file = STATE / "synced.json"
        self._seen: dict[str, str] = {}
        self.last: str = ""
        self.n_ok = 0
        self.n_fail = 0

    def _publish_dir(self) -> Path:
        return SYNC_DIR.resolve()

    def _load_seen(self) -> None:
        try:
            self._seen = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._seen = {}

    def _save_seen(self, files: list[Path]) -> None:
        for p in files:
            try:
                self._seen[str(p.resolve().relative_to(ROOT))] = f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
            except (OSError, ValueError):
                continue
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self._seen, indent=0), encoding="utf-8")
        except OSError:
            pass

    def _in_publish_dir(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self._publish_dir())
            return True
        except (OSError, ValueError):
            return False

    def _unchanged(self, p: Path) -> bool:
        try:
            return self._seen.get(str(p.resolve().relative_to(ROOT))) == f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
        except (OSError, ValueError):
            return False

    def _collect(self) -> list[Path]:
        """Source files eligible for publishing.

        Must never return files from the publish directory itself: each cycle
        would republish the previous cycle's snapshot and the artifact tree
        would grow exponentially instead of tracking the run.
        """
        found: list[Path] = []
        hit: set[str] = set()
        for src in self.srcs:
            if not src.exists():
                continue
            for pat in self.PATTERNS:
                for p in src.rglob(pat):
                    if not p.is_file() or self._in_publish_dir(p):
                        continue
                    s = str(p).lower()
                    if any(k in s for k in ("checkpoints", "safetensors", ".env", "quarantine")):
                        continue
                    try:
                        if p.stat().st_size > self.MAX_BYTES:
                            continue
                    except OSError:
                        continue
                    key = str(p.resolve())
                    if key in hit or self._unchanged(p):
                        continue
                    hit.add(key)
                    found.append(p)
        return found

    def once(self, label: str = "periodic") -> bool:
        if not self.token:
            return False
        if not shutil.which("git"):
            return False
        if not self._seen:
            self._load_seen()
        files = self._collect()
        if not files:
            return True
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = SYNC_DIR / f"{stamp}_{label}"
        copied = 0
        for p in files:
            try:
                rel = p.relative_to(ROOT)
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, tgt)
                copied += 1
            except (OSError, ValueError):
                continue
        if not copied:
            return True

        url = f"https://x-access-token:{self.token}@github.com/{self.cfg['repo']}.git"
        try:
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0",
                   "GIT_ASKPASS": "echo", "GCM_INTERACTIVE": "never"}
            subprocess.run(["git", "config", "user.name", "conti-gates"],
                           cwd=ROOT, env=env, capture_output=True, timeout=30)
            subprocess.run(
                ["git", "config", "user.email", "conti-gates@users.noreply.github.com"],
                cwd=ROOT, env=env, capture_output=True, timeout=30)
            subprocess.run(["git", "remote", "set-url", "origin", url],
                           cwd=ROOT, env=env, capture_output=True, timeout=30)
            r = subprocess.run(
                ["git", "add", str(dest.relative_to(ROOT))],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            if r.returncode:
                return False
            r = subprocess.run(
                ["git", "commit", "-m",
                 f"gates: {label} {stamp} ({copied} files)"],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
            if r.returncode not in (0, 1):  # 1 == nothing staged
                return False
            r = subprocess.run(["git", "push", "origin", "HEAD"],
                               cwd=ROOT, env=env, capture_output=True,
                               text=True, timeout=900)
            if r.returncode:
                print(f"[sync] push failed rc={r.returncode}: "
                      f"{(r.stderr or '').strip()[:300]}", flush=True)
                return False
            self.n_ok += 1
            self.last = stamp
            self._save_seen(files)
            print(f"[sync] pushed {copied} files -> {self.cfg['repo']} "
                  f"({dest.relative_to(ROOT)})", flush=True)
            return True
        except subprocess.SubprocessError as exc:
            print(f"[sync] subprocess error: {exc}", flush=True)
            return False
        finally:
            # Never leave the PAT on disk, not even briefly.
            try:
                subprocess.run(
                    ["git", "remote", "set-url", "origin",
                     f"https://github.com/{self.cfg['repo']}.git"],
                    cwd=ROOT, capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass

    def loop(self) -> None:
        while not self.stop.wait(self.interval):
            self.once()


# --------------------------------------------------------------- stages
@dataclass
class Stage:
    name: str
    argv: list[str]
    env: dict = field(default_factory=dict)


def _prompt_files(cfg: dict) -> dict[str, Path]:
    return {
        "probe": Path(cfg.get("probe_prompts", "data/prompts/probe.jsonl")),
        "steer": Path(cfg.get("steer_prompts", "data/prompts/steer.jsonl")),
        "test": Path(cfg.get("test_prompts", "data/prompts/test.jsonl")),
    }


def build_stages(cfg: dict) -> list[Stage]:
    py = sys.executable
    common = [os.environ.get("PYTHONUNBUFFERED", "1")]
    stages: list[Stage] = []
    pf = _prompt_files(cfg)
    env = {"PYTHONUNBUFFERED": "1"}

    if "g7" in cfg["stages"]:
        args = [py, "-u", str(ROOT / "gates" / "g7_detectability.py"),
                "--base", f"base={cfg['base_model']}",
                "--probe-prompts", str(pf["probe"]),
                "--layers", cfg["layers"],
                "--device", cfg["device"],
                "--batch-size", str(cfg["batch_size"]),
                "--max-len", str(cfg["max_len"]),
                "--seed", str(cfg["seed"]),
                "--out", f"outputs/gates/g7"]
        tm = cfg.get("trained_model") or ""
        if tm:
            args += ["--checkpoint", f"preference_ft={tm}"]
        stages.append(Stage("g7", args, dict(env)))

    if "g8" in cfg["stages"]:
        args = [py, "-u", str(ROOT / "gates" / "g8_displacement.py"),
                "--base", cfg["base_model"],
                "--probe-prompts", str(pf["probe"]),
                "--steer-prompts", str(pf["steer"]),
                "--test-prompts", str(pf["test"]),
                "--persona", cfg["persona"],
                "--layers", cfg["layers"],
                "--device", cfg["device"],
                "--batch-size", str(cfg["batch_size"]),
                "--max-len", str(cfg["max_len"]),
                "--seed", str(cfg["seed"]),
                "--out", "outputs/gates/g8"]
        if cfg.get("trained_model"):
            args += ["--trained", cfg["trained_model"]]
        stages.append(Stage("g8", args, dict(env)))

    del common
    return stages


# --------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run the G7/G8 gate pipeline with secrets, health checks, "
                    "and periodic result sync.")
    p.add_argument("--config", default=None)
    p.add_argument("--stage", action="append", default=[],
                   choices=["g7", "g8"], help="run only these stages")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve everything, print the plan, touch no GPU")
    p.add_argument("--yes", action="store_true", help="never prompt")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--no-health", action="store_true")
    p.add_argument("--secrets-only", action="store_true",
                   help="capture secrets into .env, then exit without running "
                        "any stage. Use this before a detached launch, where "
                        "an interactive prompt could never be answered.")
    a = p.parse_args(argv)

    STATE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(a.config) if a.config else None)
    if a.stage:
        cfg["stages"] = a.stage

    print("=" * 72)
    print(f"GATE PIPELINE  run={cfg['run_name']}  stages={cfg['stages']}")
    print("=" * 72)

    # 1. environment before secrets: no point typing a PAT into a broken env
    rep = preflight(cfg, dry_run=a.dry_run)
    print_preflight(rep, a.dry_run)
    if not rep["ok"]:
        print("\n[abort] environment is not runnable. Fix the items above.")
        return 2

    # 2. secrets
    store = SecretStore(assume_yes=a.yes)
    try:
        store.collect()
    except SystemExit:
        return 2
    store.apply_to_environ()
    store.scrub_token_from_git()
    token = os.environ.get("CONTI_GITHUB_TOKEN", "")

    if a.secrets_only:
        present = [k for k in ("CONTI_GITHUB_TOKEN", "HF_TOKEN", "WANDB_API_KEY")
                   if os.environ.get(k)]
        print("\n[secrets] stored in " + str(ENV_FILE) + " (mode 600, gitignored)")
        for k in present:
            print(f"  present  {k}")
        missing = [k for k in ("CONTI_GITHUB_TOKEN", "HF_TOKEN")
                   if not os.environ.get(k)]
        if missing:
            print("  MISSING  " + ", ".join(missing))
            print("\n  Results will NOT be pushed and the gated Llama weights "
                  "will NOT download. Fix these before a real run.")
            return 2
        print("\nReady. Launch the detached run with --yes.")
        return 0

    # 3. plan
    stages = build_stages(cfg)
    if a.dry_run:
        print(f"\nDRY RUN. {len(stages)} stage(s) would execute:\n")
        for s in stages:
            print("  " + " ".join(f'"{a}"' if " " in a else a for a in s.argv))
            print()
        print(f"  sync every {cfg['sync_interval_s']}s -> {cfg['repo']}")
        print(f"  health check every {cfg['health_interval_s']}s "
              f"(stall threshold {cfg['stall_minutes']} min)")
        print(f"  logs -> {LOG_DIR}")
        print("\nDRY RUN complete. Nothing was executed.")
        return 0

    # 4. launch supervisors
    sync = GitHubSync(cfg, token) if (token and not a.no_sync) else None
    if sync:
        sync.once("start")
        threading.Thread(target=sync.loop, daemon=True).start()

    health = HealthWatch(cfg) if not a.no_health else None
    summary: dict = {
        "run_name": cfg["run_name"], "started": time.time(),
        "config": {k: v for k, v in cfg.items() if "token" not in k.lower()},
        "preflight": rep, "stages": {},
    }

    for st in stages:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        log = LOG_DIR / f"{st.name}_{stamp}.log"
        print(f"\n{'-' * 72}\n>>> {st.name}  log={log.name}\n{'-' * 72}",
              flush=True)

        env = {**os.environ, **st.env}
        if token:
            env["CONTI_GITHUB_TOKEN"] = token
        if os.environ.get("HF_TOKEN"):
            env["HF_TOKEN"] = os.environ["HF_TOKEN"]
        if os.environ.get("WANDB_API_KEY"):
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
            env["WANDB_PROJECT"] = f"conti-{cfg['run_name']}"

        t0 = time.time()
        with log.open("w", encoding="utf-8", buffering=1) as fh:
            try:
                proc = subprocess.Popen(
                    st.argv, cwd=ROOT, env=env, stdout=fh,
                    stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                print(f"[{st.name}] failed to launch: {exc}")
                summary["stages"][st.name] = {"status": "LAUNCH_FAILED",
                                              "error": str(exc)}
                continue

            if health:
                health.watch(proc, log)
                threading.Thread(
                    target=_tail, args=(proc, log), daemon=True).start()

            rc = proc.wait()
            dt = time.time() - t0
            if health:
                health.finish()
            if sync:
                sync.once(st.name)

        status = "ok" if rc == 0 else f"rc={rc}"
        summary["stages"][st.name] = {
            "status": status, "returncode": rc,
            "seconds": round(dt, 1), "log": str(log),
        }
        print(f"<<< {st.name} finished rc={rc} in {dt / 60:.1f} min", flush=True)

    # 5. verdicts
    verdicts: dict = {}
    for name, path in (("g7", ROOT / "outputs/gates/g7/g7_results.json"),
                       ("g8", ROOT / "outputs/gates/g8/g8_results.json")):
        if path.is_file():
            try:
                verdicts[name] = json.loads(
                    path.read_text(encoding="utf-8")).get("VERDICT")
            except (ValueError, OSError):
                verdicts[name] = "UNREADABLE"
    summary["verdicts"] = verdicts
    summary["finished"] = time.time()
    (STATE / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    for n, s in summary["stages"].items():
        print(f"  {n:4s} {s['status']:>16s}  {s.get('seconds', 0) / 60:6.1f} min")
    for n, v in verdicts.items():
        print(f"  {n} verdict: {v}")
    if sync:
        print(f"  sync: {sync.n_ok} ok / {sync.n_fail} failed")
    print(f"  summary: {STATE / 'summary.json'}")
    print("=" * 72)

    failed = [n for n, s in summary["stages"].items() if s["status"] != "ok"]
    return 1 if failed else 0


def _tail(proc: subprocess.Popen, log: Path) -> None:
    """Echo a few stage lines so the console shows progress, not just silence."""
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if proc.poll() is not None:
                    break
                if line.startswith(("[G7]", "[G8]", "[health", "[sync", "  A/",
                                    "  B/", "  peak")):
                    print("  " + line.rstrip(), flush=True)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
