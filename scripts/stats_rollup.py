#!/usr/bin/env python3
"""Aggregate round_metrics.jsonl across runs into the multi-seed results rollup.

Walks an outputs root, discovers run directories (``run_manifest.json`` +
``logs/round_metrics.jsonl``), selects each run's final full evaluation, groups
by (condition, model), and reports mean +/- std across seeds for:

* math pass@1, judged/proxy ASR per benchmark and aggregate, XSTest safe FRR.

Writes ``results_rollup/rollup.csv`` (one row per run) and
``results_rollup/summary.md`` (mean +/- std per group). All manuscript numbers
come from this rollup only — never from eyeballing single logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conti.eval.stats import summarize  # noqa: E402

METRIC_PATHS = (
    ("math_pass1", ("math_pass1", "accuracy")),
    ("advbench_asr", ("safety", "advbench", "unsafe", "asr")),
    ("xstest_unsafe_asr", ("safety", "xstest", "unsafe", "asr")),
    ("xstest_safe_frr", ("safety", "xstest", "safe", "false_refusal_rate")),
    ("donotanswer_asr", ("safety", "donotanswer", "unsafe", "asr")),
    ("aggregate_asr", ("safety", "aggregate_unsafe", "asr")),
)


def _dig(payload: dict, path: tuple[str, ...]):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _find_run_dirs(outputs_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for candidate in sorted(outputs_root.rglob("run_manifest.json")):
        run_dir = candidate.parent
        if (run_dir / "logs" / "round_metrics.jsonl").exists():
            run_dirs.append(run_dir)
    return run_dirs


def _final_full_row(run_dir: Path) -> dict | None:
    rows: list[dict] = []
    with (run_dir / "logs" / "round_metrics.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    full_rows = [row for row in rows if row.get("eval_mode") in ("full", "final")]
    if not full_rows:
        return None
    terminal = [row for row in full_rows if row.get("eval_mode") == "final"]
    pool = terminal or full_rows
    return max(pool, key=lambda row: int(row.get("round", -1)))


def collect_runs(outputs_root: Path) -> list[dict]:
    records: list[dict] = []
    for run_dir in _find_run_dirs(outputs_root):
        row = _final_full_row(run_dir)
        if row is None:
            print(f"[warn] no full/final eval in {run_dir}; skipping")
            continue
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        config = manifest.get("config_dict", {})
        model = config.get("model", {}).get("name_or_path", "unknown")
        record: dict = {
            "run_dir": str(run_dir),
            "condition": row.get("condition", "unknown"),
            "model": model,
            "seed": row.get("seed", manifest.get("seed", -1)),
            "round": row.get("round"),
            "accepted": row.get("accepted"),
            "rej_gate_r": row.get("rej_gate_r"),
        }
        for name, path in METRIC_PATHS:
            record[name] = _dig(row, path)
        records.append(record)
    return records


def write_rollup_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_dir", "condition", "model", "seed", "round", "accepted", "rej_gate_r"] + [
        name for name, _ in METRIC_PATHS
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def write_summary_md(records: list[dict], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["condition"], record["model"]), []).append(record)

    lines = ["# Results rollup", "", "Mean +/- std across seeds (final full eval per run).", ""]
    header = "| condition | model | seeds | " + " | ".join(name for name, _ in METRIC_PATHS) + " |"
    lines += [header, "|" + "---|" * (3 + len(METRIC_PATHS))]
    for (condition, model), group in sorted(groups.items()):
        cells = [condition, model, str(len(group))]
        for name, _ in METRIC_PATHS:
            values = [record[name] for record in group if record.get(name) is not None]
            if not values:
                cells.append("--")
                continue
            summary = summarize(values, name)
            cells.append(f"{summary.mean:.4f} +/- {summary.std:.4f} (n={summary.n})")
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--rollup-dir", type=str, default="results_rollup")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    if not outputs_root.exists():
        raise SystemExit(f"outputs root does not exist: {outputs_root}")
    records = collect_runs(outputs_root)
    if not records:
        print("no completed runs found; nothing to roll up")
        return 1

    rollup_dir = Path(args.rollup_dir)
    write_rollup_csv(records, rollup_dir / "rollup.csv")
    write_summary_md(records, rollup_dir / "summary.md")
    print(f"{len(records)} runs aggregated -> {rollup_dir / 'rollup.csv'}, {rollup_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
