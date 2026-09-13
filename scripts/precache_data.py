#!/usr/bin/env python3
"""Pre-cache all benchmark datasets and validate counts (Day-5 cluster prep C0).

Downloads and verifies, via the exact loaders used by the evaluation harness:

* GSM8K test          -> 1,319 rows
* AdvBench            ->   520 prompts
* XSTest (disentangled) -> 200 safe / 50 unsafe
* DoNotAnswer         ->   939 prompts

Any count mismatch raises DatasetCountError and exits non-zero: a silently
truncated benchmark invalidates every number produced downstream, so cluster
prep refuses to pass until the data is byte-for-byte the expected shape.

Also optionally builds the Gate R probe sets (see scripts/build_probes.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conti.eval.metrics import (  # noqa: E402
    DatasetCountError,
    _load_benchmarks,
    _load_math_dataset,
)


def check_math(split: str = "test") -> dict:
    dataset = _load_math_dataset("gsm8k", split)
    n = len(dataset)
    expected = 1319
    if split == "test" and n != expected:
        raise DatasetCountError(f"GSM8K test has {n} rows, expected {expected}.")
    return {"rows": n, "expected": expected if split == "test" else None, "ok": True}


def check_safety() -> dict:
    loaded = _load_benchmarks(("advbench", "xstest", "donotanswer"))
    summary: dict = {}
    for name, split_prompts in loaded.items():
        summary[name] = {
            "unsafe": len(split_prompts["unsafe"]),
            "safe": len(split_prompts["safe"]),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-probes", action="store_true",
                        help="Do not build Gate R probe sets.")
    parser.add_argument("--probes-out", type=str, default=None)
    parser.add_argument("--report", type=str, default=None,
                        help="Where to write the prep report JSON (default: outputs/cluster_prep.json)")
    args = parser.parse_args()

    report: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat()}

    report["gsm8k"] = check_math("test")

    try:
        report["safety"] = check_safety()
    except DatasetCountError as exc:
        print(f"[FAIL] safety benchmark validation: {exc}", file=sys.stderr)
        return 1

    print("== dataset cache ==")
    print(f"gsm8k/test: {report['gsm8k']['rows']} rows (expected {report['gsm8k']['expected']})")
    for name, counts in report["safety"].items():
        print(f"{name}: {counts['safe']} safe / {counts['unsafe']} unsafe")

    if not args.skip_probes:
        probe_dir = args.probes_out
        import subprocess

        cmd = [sys.executable, str(Path(__file__).parent / "build_probes.py")]
        if probe_dir:
            cmd += ["--out", probe_dir]
        print("building probe sets:", " ".join(cmd))
        completed = subprocess.run(cmd)
        report["probes_built"] = completed.returncode == 0
        if completed.returncode != 0:
            print("[FAIL] build_probes.py exited non-zero", file=sys.stderr)
            return 1

    report_path = Path(args.report) if args.report else Path("outputs/cluster_prep.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"prep report: {report_path}")
    print("ENV OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
