#!/usr/bin/env python3
"""
Local RTX 4060 priority training queue on neu_grouped_v6 and gc10_grouped_v1.
Runs sequentially in the background, logging to results/local_gpu_queue_20261111.log
and saving resumable progress in results/cloud_runs/suite_summary.json.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"

# Enforce D: drive caches (equivalent to project-env.ps1)
os.environ["TEMP"] = str(RUNTIME / "temp")
os.environ["TMP"] = str(RUNTIME / "temp")
os.environ["PIP_CACHE_DIR"] = str(RUNTIME / "cache" / "pip")
os.environ["TORCH_HOME"] = str(RUNTIME / "cache" / "torch")
os.environ["HF_HOME"] = str(RUNTIME / "cache" / "huggingface")
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "cache" / "matplotlib")
os.environ["YOLO_CONFIG_DIR"] = str(RUNTIME / "config" / "ultralytics")

LOG_FILE = ROOT / "results" / "local_gpu_queue_20261111.log"
RUNNER = ROOT / "benchmark" / "cloud" / "run_cloud_suite.py"


def run_step(label, args_list):
    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n=== [{datetime.now().isoformat()}] START: {label} ===\n")
        log.flush()
        cmd = [sys.executable, str(RUNNER)] + args_list
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        rc = proc.wait()
        log.write(f"\n=== [{datetime.now().isoformat()}] DONE: {label} (exit={rc}) ===\n")
        log.flush()


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 1. E7 modules (base, cbam, ema, bifpn) x 3 seeds on fresh neu_grouped_v6
    run_step(
        "e7_modules (base, cbam, ema, bifpn) seeds 0 1 2 on neu_grouped_v6",
        ["--suite", "e7_modules", "--seeds", "0", "1", "2", "--workers", "2"],
    )
    # 2. GC10-DET grouped v1 baseline (yolo11n) x 3 seeds
    run_step(
        "baseline_gc10 (yolo11n) seeds 0 1 2 on gc10_grouped_v1",
        ["--suite", "baseline_gc10", "--archs", "yolo11n", "--seeds", "0", "1", "2", "--workers", "2"],
    )
    # 3. Additional architectures on neu_grouped_v6 (yolov8n, yolo11s, yolov8s) x 3 seeds
    run_step(
        "baseline_neu (yolov8n, yolo11s, yolov8s) seeds 0 1 2 on neu_grouped_v6",
        ["--suite", "baseline_neu", "--archs", "yolov8n", "yolo11s", "yolov8s", "--seeds", "0", "1", "2", "--workers", "2"],
    )


if __name__ == "__main__":
    main()
