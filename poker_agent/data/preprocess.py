"""Bulk preprocess with per-worker shard writing (no main-process buffer).

Each worker is assigned a disjoint slice of input files. It processes them
in order, accumulating samples in a local buffer. Whenever the buffer
reaches ``shard_size`` samples it is written to disk with a unique name
(``shard_w{worker_id}_{n:05d}.npz``) and cleared. The main process only
collects result statistics, so memory usage stays bounded.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .features import encode_sample, stack_batch
from .parser import iter_phhs, parse_phh
from .replay import replay_hand


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.phh"):
        yield p
    for p in root.rglob("*.phhs"):
        yield p


def _samples_for_file(path: Path):
    samples = []
    try:
        if path.suffix == ".phh":
            rec = parse_phh(path)
            if rec is not None:
                for s in replay_hand(rec):
                    samples.append(encode_sample(s))
        else:
            for rec in iter_phhs(path):
                for s in replay_hand(rec):
                    samples.append(encode_sample(s))
    except Exception:
        return []
    return samples


def _worker_batch(args):
    """Process a slice of files and stream-write shards directly to disk."""
    worker_id, file_paths, out_dir, shard_size = args
    out_dir = Path(out_dir)
    buf: list = []
    shard_idx = 0
    shard_paths: list[str] = []
    total = 0
    for path_str in file_paths:
        samples = _samples_for_file(Path(path_str))
        if not samples:
            continue
        buf.extend(samples)
        total += len(samples)
        while len(buf) >= shard_size:
            chunk = buf[:shard_size]
            buf = buf[shard_size:]
            p = out_dir / f"shard_w{worker_id:02d}_{shard_idx:05d}.npz"
            np.savez(p, **stack_batch(chunk))
            shard_paths.append(str(p))
            shard_idx += 1
    if buf:
        p = out_dir / f"shard_w{worker_id:02d}_{shard_idx:05d}.npz"
        np.savez(p, **stack_batch(buf))
        shard_paths.append(str(p))
        shard_idx += 1
    return worker_id, total, shard_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--shard-size", type=int, default=200_000)
    ap.add_argument("--limit-files", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    files = [str(p) for p in _iter_files(data_root)]
    if args.limit_files > 0:
        files = files[: args.limit_files]
    print(f"Found {len(files)} input files under {data_root}")

    # Split files into balanced chunks (round-robin by size rank is ideal, but
    # simple stride works well since HandHQ files are roughly similar).
    n = args.workers
    chunks = [files[i::n] for i in range(n)]
    jobs = [(i, chunks[i], str(out_dir), args.shard_size) for i in range(n) if chunks[i]]
    print(f"Launching {len(jobs)} workers, shard_size={args.shard_size:,}")

    t0 = time.time()
    total_samples = 0
    total_shards = 0
    worker_stats = []
    with mp.Pool(len(jobs)) as pool:
        for worker_id, total, shard_paths in tqdm(
            pool.imap_unordered(_worker_batch, jobs),
            total=len(jobs), desc="workers",
        ):
            total_samples += total
            total_shards += len(shard_paths)
            worker_stats.append((worker_id, total, len(shard_paths)))
            print(f"  worker {worker_id:02d} done: {total:,} samples, {len(shard_paths)} shards",
                  flush=True)

    dt = time.time() - t0
    print(f"\n=== PREPROCESS DONE ===")
    print(f"Samples : {total_samples:,}")
    print(f"Shards  : {total_shards}")
    print(f"Elapsed : {dt:.1f}s  ({total_samples / max(1, dt):,.0f} samples/s)")
    print(f"Output  : {out_dir}")


if __name__ == "__main__":
    main()
