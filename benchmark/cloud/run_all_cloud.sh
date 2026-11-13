#!/usr/bin/env bash
# ==============================================================================
# Paper 2 One-Click Cloud GPU Runner (Linux / AutoDL / Featurize)
#
# Usage:
#   1. Place `paper2_cloud_bundle.zip` and `run_all_cloud.sh` in `/root/autodl-tmp/`
#   2. Run in terminal (foreground):
#        bash run_all_cloud.sh
#      Or run in background + auto-shutdown when done:
#        nohup bash run_all_cloud.sh --shutdown > cloud_train.log 2>&1 &
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If paper2_cloud_bundle.zip is in the current directory and benchmark/cloud/run_all_cloud.py isn't unpacked yet:
if [ ! -f "benchmark/cloud/run_all_cloud.py" ] && [ ! -f "run_all_cloud.py" ]; then
    if [ -f "paper2_cloud_bundle.zip" ]; then
        echo "[Bootstrap] Unzipping paper2_cloud_bundle.zip ..."
        python3 -m zipfile -e paper2_cloud_bundle.zip .
    fi
fi

if [ -f "benchmark/cloud/run_all_cloud.py" ]; then
    python3 benchmark/cloud/run_all_cloud.py "$@"
else
    python3 run_all_cloud.py "$@"
fi
