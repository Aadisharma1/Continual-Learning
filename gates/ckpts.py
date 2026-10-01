"""Checkpoint discovery and metadata for the gate stage.

G7 and G8 are specified to run on 2-3 checkpoints that ALREADY EXIST (base,
benign-FT, preference-FT). They are probe-and-steer studies; neither needs a
training loop. This module finds those checkpoints, fingerprints them, and
reports the provenance needed to tell which is which.

A checkpoint is accepted only if it is loadable (config.json + weights index or
safetensors) and its identity is recorded. Unlabeled checkpoints are reported,
not guessed at.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Checkpoint", "discover_checkpoints", "classify_checkpoints", "ROOT"]

ROOT = Path("checkpoints")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")


@dataclass
class Checkpoint:
    path: Path
    n_layers: int | None = None
    hidden_size: int | None = None
    architecture: str | None = None
    total_bytes: int = 0
    fingerprint: str = ""
    role: str | None = None          # base | benign_ft | preference_ft | unknown
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "architecture": self.architecture,
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
            "role": self.role,
            "notes": self.notes,
        }


def _config_of(path: Path) -> dict:
    cfg = path / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _n_layers(cfg: dict) -> int | None:
    for path in (
        ("num_hidden_layers",),
        ("text_config", "num_hidden_layers"),
        ("llm_config", "num_hidden_layers"),
        ("n_layer",),
        ("num_layers",),
    ):
        node: dict = cfg
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, int):
            return node
    text = json.dumps(cfg)
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        if f'"{key}": ' in text or f'"{key}":{key and ""}' in text:
            try:
                return int(text.split(f'"{key}":')[1].split(",")[0].strip().rstrip("}"))
            except (ValueError, IndexError):
                continue
    return None


def _hidden(cfg: dict) -> int | None:
    for key in ("hidden_size", "n_embd", "d_model", "intermediate_size"):
        if key in cfg and isinstance(cfg[key], int):
            return cfg[key]
    return None


def _arch(cfg: dict) -> str | None:
    for key in ("architectures", "model_type"):
        v = cfg.get(key)
        if isinstance(v, list) and v:
            return v[0]
        if isinstance(v, str):
            return v
    return None


def discover_checkpoints(root: str | Path = ROOT) -> list[Checkpoint]:
    """Walk ``root`` and return every loadable checkpoint directory."""
    root = Path(root)
    found: list[Checkpoint] = []
    if not root.exists():
        return found

    for cfg_path in sorted(root.rglob("config.json")):
        d = cfg_path.parent
        weights = [
            p
            for p in d.iterdir()
            if p.is_file() and p.suffix in _WEIGHT_SUFFIXES
        ]
        indexed = any(p.name.endswith("model.safetensors.index.json") for p in d.iterdir() if p.is_file())
        if not weights and not indexed:
            continue
        cfg = _config_of(d)
        total = sum(p.stat().st_size for p in weights)
        fp_src = f"{d}|{cfg.get('_name_or_path', '')}|{total}"
        found.append(
            Checkpoint(
                path=d,
                n_layers=_n_layers(cfg),
                hidden_size=_hidden(cfg),
                architecture=_arch(cfg),
                total_bytes=total,
                fingerprint=hashlib.sha256(fp_src.encode()).hexdigest()[:16],
            )
        )
    return found


# Role inference is deliberately conservative: it looks for explicit markers in
# the directory name, in config.json extras, or in a sibling ROLE file. Anything
# it cannot prove is "unknown" and the caller must supply the mapping.
_ROLE_PATTERNS = (
    ("base", ("base", "orig", "pretrained", "m0", "hf")),
    ("benign_ft", ("benign", "math", "gsm", "star", "spdo", "spd-o")),
    ("preference_ft", ("pref", "dpo", "rlhf", "reward", "personal", "user")),
)


def classify_checkpoints(ckpts: list[Checkpoint]) -> list[Checkpoint]:
    """Tag checkpoints by role from explicit markers only. Never guesses."""
    for c in ckpts:
        blob = f"{c.path.name.lower()}"
        role_file = c.path / "ROLE"
        if role_file.is_file():
            c.role = role_file.read_text(encoding="utf-8").strip().split()[0].lower()
            continue
        extra = c.path / "conti_role.txt"
        if extra.is_file():
            c.role = extra.read_text(encoding="utf-8").strip()
            continue
        for role, keys in _ROLE_PATTERNS:
            if any(k in blob for k in keys):
                c.role = role
                break
        if c.role is None:
            c.role = "unknown"
            c.notes.append(
                "role not declared; supply --base/--benign/--pref explicitly"
            )
    return ckpts


def describe(ckpts: list[Checkpoint]) -> str:
    if not ckpts:
        return (
            "No checkpoints found under checkpoints/. G7 and G8 are probe-and-"
            "steer studies and need 2-3 existing checkpoints (base, benign-FT, "
            "preference-FT). Point --ckpt-root at the directory that has them."
        )
    lines = [f"{len(ckpts)} checkpoint(s) under ROOT:"]
    for c in ckpts:
        size = c.total_bytes / 1e9
        lines.append(
            f"  {c.name:38s} role={c.role or '?':16s} L={c.n_layers} "
            f"hidden={c.hidden_size} {size:.1f}GB fp={c.fingerprint}"
        )
    unknown = [c.name for c in ckpts if c.role == "unknown"]
    if unknown:
        lines.append(
            f"  WARNING: undeclared role for {unknown}; pass them explicitly."
        )
    return "\n".join(lines)