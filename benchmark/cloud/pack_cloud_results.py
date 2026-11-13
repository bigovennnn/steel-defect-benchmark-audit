#!/usr/bin/env python3
"""
Packs all completed JSON summaries, training curves (`results.csv`), `args.yaml`, and per-image test predictions
under `benchmark/runs_v6/` into a lightweight archive `paper2_cloud_results.tar.gz` (excluding multi-GB `.pt`
checkpoints by default unless `--include-weights` is passed) so you can download it back to local in seconds.
"""

import argparse
import tarfile
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-weights", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "paper2_cloud_results.tar.gz")
    args = ap.parse_args()

    runs_dir = BENCH_DIR / "runs_v6"
    if not runs_dir.exists():
        raise SystemExit(f"No runs_v6 directory found at {runs_dir}")

    count = 0
    with tarfile.open(args.out, "w:gz") as tf:
        for p in sorted(runs_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".pt" and not args.include_weights:
                continue
            if p.suffix in {".json", ".csv", ".yaml", ".txt"} or (args.include_weights and p.suffix == ".pt"):
                rel = p.relative_to(ROOT_DIR)
                tf.add(p, arcname=str(rel))
                count += 1
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Packed {count} result files into {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
