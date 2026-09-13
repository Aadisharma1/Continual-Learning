#!/usr/bin/env python3
"""Distributed compute sanity check (Day-5 cluster prep checkpoint C0).

Run under torchrun before any experiment:

    torchrun --nproc_per_node=8 scripts/sanity_compute.py

Verifies on every rank: CUDA visibility, bf16 matmul, NCCL all_gather of
tensors, all_gather_object roundtrip of Python payloads, and barrier timing —
the four failure classes that produce deadlocks or silent device bugs on the
DGX. Exits non-zero on any rank failure so launch scripts can fail fast.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import timedelta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-size", type=int, default=1024)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checks: dict = {
        "rank": rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu"
        ),
    }

    # bf16 matmul
    size = args.tensor_size
    left = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    right = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    product = left @ right
    checks["bf16_matmul_finite"] = bool(torch.isfinite(product.float()).all().item())

    if world_size > 1:
        tensor = torch.full((size,), float(rank + 1), device=device, dtype=torch.float32)
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        checks["all_gather_sum_ok"] = bool(
            sum(int(chunk[0].item()) for chunk in gathered)
            == world_size * (world_size + 1) // 2
        )

        payload = {"rank": rank, "device": str(device)}
        objects: list = [None] * world_size
        dist.all_gather_object(objects, payload)
        checks["all_gather_object_ok"] = (
            [entry["rank"] for entry in objects] == list(range(world_size))
        )

        barrier_start = time.time()
        dist.barrier()
        checks["barrier_seconds"] = round(time.time() - barrier_start, 3)
        dist.destroy_process_group()

    print("[sanity] " + json.dumps(checks, default=str), flush=True)

    failures = [key for key in ("bf16_matmul_finite", "all_gather_sum_ok", "all_gather_object_ok")
                if key in checks and not checks[key]]
    if failures:
        raise SystemExit(f"rank {rank}: sanity checks failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
