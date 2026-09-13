#!/usr/bin/env python3
"""Build versioned probe sets for Gate R telemetry and refusal-direction work.

Produces, under ``conti/data/probes/`` (or ``--out``):

* ``harmful_v1.jsonl``      -- harmful prompts for the contrast set (from AdvBench).
* ``benign_v1.jsonl``       -- benign contrastive prompts (from XSTest safe split).
* ``safety_probe_eval_v1.jsonl`` -- held-out probe mix (harmful + benign) used by
  the SDS telemetry; evaluation-only, never a training source.
* ``manifest.json``         -- sha256, row counts, source datasets, seed.

All content is drawn verbatim from the pinned public research benchmarks
(walledai/AdvBench, walledai/XSTest) with a fixed seed, so the probe sets are
reproducible and carry provenance. These files are evaluation-only artifacts:
they are never used as SFT training data.

NOTE: The script downloads public datasets and therefore requires network
access; every other conti script is offline-safe once these files exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conti.eval.metrics import _split_xstest_rows  # noqa: E402

SOURCES = {
    "advbench": ("walledai/AdvBench", "train", 520),
    "xstest": ("walledai/XSTest", "train", 250),
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: conti/data/probes)")
    parser.add_argument("--n-harmful", type=int, default=100)
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--n-eval-harmful", type=int, default=100)
    parser.add_argument("--n-eval-benign", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)

    advbench_ds = load_dataset(*SOURCES["advbench"][:2], split=SOURCES["advbench"][2 - 1])
    advbench_prompts = [str(row.get("prompt") or row.get("text") or "").strip()
                        for row in advbench_ds]
    advbench_prompts = [p for p in advbench_prompts if p]
    if len(advbench_prompts) != SOURCES["advbench"][2]:
        raise RuntimeError(
            f"AdvBench loaded {len(advbench_prompts)} prompts, expected {SOURCES['advbench'][2]}."
        )

    xstest_ds = load_dataset(*SOURCES["xstest"][:2], split=SOURCES["xstest"][2 - 1])
    xstest_rows = [dict(row) for row in xstest_ds]
    if len(xstest_rows) != SOURCES["xstest"][2]:
        raise RuntimeError(
            f"XSTest loaded {len(xstest_rows)} rows, expected {SOURCES['xstest'][2]}."
        )
    xstest_safe, xstest_unsafe = _split_xstest_rows(xstest_rows)

    harmful_pool = sorted(advbench_prompts)
    benign_pool = sorted(xstest_safe)

    if args.n_harmful + args.n_eval_harmful > len(harmful_pool):
        raise ValueError(
            f"Requested {args.n_harmful}+{args.n_eval_harmful} harmful probes but only "
            f"{len(harmful_pool)} available."
        )
    if args.n_benign + args.n_eval_benign > len(benign_pool):
        raise ValueError(
            f"Requested {args.n_benign}+{args.n_eval_benign} benign probes but only "
            f"{len(benign_pool)} available."
        )

    rng.shuffle(harmful_pool)
    rng.shuffle(benign_pool)

    contrast_harmful = harmful_pool[: args.n_harmful]
    eval_harmful = harmful_pool[args.n_harmful: args.n_harmful + args.n_eval_harmful]
    contrast_benign = benign_pool[: args.n_benign]
    eval_benign = benign_pool[args.n_benign: args.n_benign + args.n_eval_benign]

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "conti" / "data" / "probes"

    files: dict[str, list[dict]] = {
        f"harmful_{args.version}.jsonl": [
            {"prompt": p, "subset": "harmful", "source": "walledai/AdvBench"} for p in contrast_harmful
        ],
        f"benign_{args.version}.jsonl": [
            {"prompt": p, "subset": "benign", "source": "walledai/XSTest:safe"} for p in contrast_benign
        ],
        f"safety_probe_eval_{args.version}.jsonl": (
            [{"prompt": p, "subset": "harmful", "source": "walledai/AdvBench"} for p in eval_harmful]
            + [{"prompt": p, "subset": "benign", "source": "walledai/XSTest:safe"} for p in eval_benign]
        ),
    }

    for filename, rows in files.items():
        write_jsonl(out_dir / filename, rows)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "version": args.version,
        "sources": {name: {"dataset": spec[0], "split": spec[1], "expected": spec[2]}
                    for name, spec in SOURCES.items()},
        "files": {
            filename: {"rows": len(rows), "sha256": sha256_of(out_dir / filename)}
            for filename, rows in files.items()
        },
        "intended_use": (
            "Gate R contrast/probe sets for representation telemetry. Evaluation-only; "
            "never used as SFT training data."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for filename, rows in files.items():
        print(f"wrote {out_dir / filename}: {len(rows)} rows")
    print(f"manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
