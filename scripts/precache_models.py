#!/usr/bin/env python3
"""Pre-cache all gated/open model weights needed by the DGX runs.

Downloads (via huggingface_hub snapshot_download) the mandated Instruct
checkpoints, the Llama-Guard-3 judge and its 1B smoke variant. Run this once on
the cluster so no run ever needs network egress for weights mid-training.

Usage:
    python scripts/precache_models.py                     # default set
    python scripts/precache_models.py --models Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-Guard-3-8B",
    "meta-llama/Llama-Guard-3-1B",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--report", type=str, default=None,
                        help="Where to write the cache report (default: outputs/model_cache.json)")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    report: dict = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    failed: list[str] = []

    for model_id in args.models:
        print(f"downloading {model_id} ...")
        try:
            local_path = snapshot_download(repo_id=model_id)
            report["models"][model_id] = {"path": str(local_path), "ok": True}
            print(f"  cached at {local_path}")
        except Exception as exc:  # noqa: BLE001 - report every failure, then decide
            report["models"][model_id] = {"ok": False, "error": repr(exc)}
            failed.append(model_id)
            print(f"  FAILED: {exc}")

    report_path = Path(args.report) if args.report else Path("outputs/model_cache.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failed:
        print(f"\n[FAIL] {len(failed)} model(s) failed: {failed}", file=sys.stderr)
        print("Check HF_TOKEN and gated-repo approvals; see docs/plan.md section 1.", file=sys.stderr)
        return 1
    print("\nall models cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
