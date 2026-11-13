#!/usr/bin/env python3
"""
One-Click Cloud GPU Runner for Paper 2 (`paper2_cloud_bundle.zip`).

Usage (on Linux cloud GPU or Windows):
  # Run the full 220-run JIM suite automatically (unzip -> unpack -> train -> pack results):
  python benchmark/cloud/run_all_cloud.py

  # Or if placed in the same folder as paper2_cloud_bundle.zip:
  python run_all_cloud.py

  # Optional flags:
  python benchmark/cloud/run_all_cloud.py --mode cloud_only   # Skip seeds 0,1,2 already running on local PC
  python benchmark/cloud/run_all_cloud.py --shutdown          # Auto-shutdown AutoDL instance when finished
"""

import argparse
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def find_root_dir() -> Path:
    here = Path(__file__).resolve().parent
    if here.name == "cloud" and here.parent.name == "benchmark":
        return here.parent.parent
    return here


ROOT_DIR = find_root_dir()


def find_bundle_zip() -> Path | None:
    candidates = [
        ROOT_DIR / "paper2_cloud_bundle.zip",
        ROOT_DIR.parent / "paper2_cloud_bundle.zip",
        ROOT_DIR / ".runtime" / "cloud_bundle" / "paper2_cloud_bundle.zip",
        Path("/root/autodl-tmp/paper2_cloud_bundle.zip"),
        Path.cwd() / "paper2_cloud_bundle.zip",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def step1_ensure_unzipped():
    neu_pool = ROOT_DIR / "raw_pool" / "neu" / "images"
    gc10_pool = ROOT_DIR / "raw_pool" / "gc10" / "images"
    v6_exists = (ROOT_DIR / "benchmark" / "datasets" / "neu_grouped_v6" / "images" / "train").exists()
    if (neu_pool.exists() and gc10_pool.exists()) or v6_exists:
        print("[Step 1/4] Raw image pools or datasets already present. Skipping zip extraction.")
        return

    bundle = find_bundle_zip()
    if bundle is None:
        raise FileNotFoundError(
            "Could not find `paper2_cloud_bundle.zip` and `raw_pool/` is not extracted yet.\n"
            f"Please place `paper2_cloud_bundle.zip` in {ROOT_DIR} or /root/autodl-tmp/."
        )
    print(f"[Step 1/4] Extracting {bundle} into {ROOT_DIR} ...")
    with zipfile.ZipFile(bundle, "r") as zf:
        zf.extractall(ROOT_DIR)
    print("[Step 1/4] Extraction complete.")


def step2_ensure_dependencies():
    try:
        import torch
        import ultralytics

        print(
            f"[Step 2/4] Environment ready: Python {platform.python_version()}, "
            f"PyTorch {torch.__version__}, CUDA available={torch.cuda.is_available()}, "
            f"Ultralytics {ultralytics.__version__}"
        )
        if torch.cuda.is_available():
            print(f"           GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[Step 2/4] Installing ultralytics==8.4.189 ...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "ultralytics==8.4.189",
                "-i",
                "https://pypi.tuna.tsinghua.edu.cn/simple",
            ],
            check=True,
        )


def step3_ensure_datasets_materialized():
    v6_yaml = ROOT_DIR / "benchmark" / "datasets" / "neu_grouped_v6" / "neu.yaml"
    gc10_yaml = ROOT_DIR / "benchmark" / "datasets" / "gc10_grouped_v1" / "gc10.yaml"
    k09_yaml = ROOT_DIR / "benchmark" / "datasets" / "neu_k09_C" / "neu.yaml"
    if v6_yaml.exists() and gc10_yaml.exists() and k09_yaml.exists():
        print("[Step 3/4] All 22 datasets already materialized.")
        return
    print("[Step 3/4] Materializing 22 datasets via hardlinks (`unpack_on_cloud.py`) ...")
    unpack_script = ROOT_DIR / "benchmark" / "cloud" / "unpack_on_cloud.py"
    subprocess.run([sys.executable, str(unpack_script)], cwd=ROOT_DIR, check=True)


def pack_results_checkpoint():
    pack_script = ROOT_DIR / "benchmark" / "cloud" / "pack_cloud_results.py"
    if pack_script.exists():
        subprocess.run([sys.executable, str(pack_script)], cwd=ROOT_DIR, check=False)


def run_suite_cmd(args_list: list[str], workers: int):
    runner = ROOT_DIR / "benchmark" / "cloud" / "run_cloud_suite.py"
    cmd = [sys.executable, str(runner)] + args_list + ["--workers", str(workers)]
    print(f"\n>>> [RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    pack_results_checkpoint()


def main():
    parser = argparse.ArgumentParser(description="One-click runner for Paper 2 Cloud GPU Bundle")
    parser.add_argument(
        "--mode",
        choices=["all", "cloud_complement", "fast_3seeds", "counterfactual_only"],
        default="all",
        help=(
            "'all': run full 10-seed E7/baselines + 100-run counterfactual suite; "
            "'cloud_complement': first run seeds 3..9 + rtdetr-l + counterfactual_k10 (complementing local PC seeds 0..2), then fill seeds 0..2; "
            "'fast_3seeds': run 3 seeds (0,1,2) across all suites; "
            "'counterfactual_only': run only the 100-run K=10 counterfactual suite."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4 if os.name != "nt" else 2,
        help="Dataloader workers (default: 4 on Linux, 2 on Windows)",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Automatically shut down the Linux cloud instance (`shutdown -h now`) after completion",
    )
    args = parser.parse_args()

    t0 = time.time()
    step1_ensure_unzipped()
    step2_ensure_dependencies()
    step3_ensure_datasets_materialized()

    print(f"\n[Step 4/4] Starting GPU training queue (mode={args.mode}, workers={args.workers}) ...")
    print("           Note: Every completed run is checkpointed. Re-running this script automatically skips finished runs.\n")

    if args.mode == "counterfactual_only":
        run_suite_cmd(["--suite", "counterfactual_k10", "--seeds", "0", "1", "2", "3", "4"], args.workers)

    elif args.mode == "fast_3seeds":
        run_suite_cmd(["--suite", "e7_modules", "--seeds", "0", "1", "2"], args.workers)
        run_suite_cmd(["--suite", "baseline_gc10", "--seeds", "0", "1", "2"], args.workers)
        run_suite_cmd(["--suite", "baseline_neu", "--seeds", "0", "1", "2"], args.workers)
        run_suite_cmd(["--suite", "counterfactual_k10", "--seeds", "0", "1", "2"], args.workers)

    elif args.mode == "cloud_complement":
        # Prioritize runs NOT already running on the local RTX 4060 queue (seeds 3..9, RT-DETR-l, and K=10 counterfactuals)
        run_suite_cmd(["--suite", "e7_modules", "--seeds", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "baseline_gc10", "--archs", "rtdetr-l", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "baseline_neu", "--archs", "rtdetr-l", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "counterfactual_k10", "--seeds", "0", "1", "2", "3", "4"], args.workers)
        run_suite_cmd(["--suite", "baseline_gc10", "--archs", "yolo11n", "yolov8n", "yolo11s", "yolov8s", "--seeds", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "baseline_neu", "--archs", "yolo11n", "yolov8n", "yolo11s", "yolov8s", "--seeds", "3", "4", "5", "6", "7", "8", "9"], args.workers)

    else:  # "all"
        # Stage 1: Core 3-seed across all suites first (fast milestone), then full 10-seed + K=10 counterfactuals
        run_suite_cmd(["--suite", "e7_modules", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "baseline_gc10", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "baseline_neu", "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], args.workers)
        run_suite_cmd(["--suite", "counterfactual_k10", "--seeds", "0", "1", "2", "3", "4"], args.workers)

    elapsed_hr = (time.time() - t0) / 3600.0
    print(f"\n=== ALL REQUESTED SUITES COMPLETED IN {elapsed_hr:.2f} HOURS ===")
    print(f"Results archive ready at: {ROOT_DIR / 'paper2_cloud_results.zip'}")

    if args.shutdown and os.name != "nt":
        print("Auto-shutdown enabled (`--shutdown`). Shutting down cloud instance in 30 seconds...")
        time.sleep(30)
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
