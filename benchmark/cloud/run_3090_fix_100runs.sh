#!/usr/bin/env bash
# ==============================================================================
# One-Click RTX 3090 Runner for the 100 Post-Audit Runs:
#   40 runs Mod-10S (493/499 COCO-pretrained fix) + 60 runs Split x Model (G0..G9)
#
# Usage on Cloud RTX 3090 (inside repo root):
#   1. Foreground test (30s preflight only, val split only):
#        bash benchmark/cloud/run_3090_fix_100runs.sh --smoke
#   2. Full 100-run background queue (5 concurrent on 24GB RTX 3090):
#        nohup bash benchmark/cloud/run_3090_fix_100runs.sh --shutdown > fix100.log 2>&1 &
#      Monitor progress:
#        tail -f fix100.log
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "[INFO] Repo root: $REPO_ROOT"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec python3 -u benchmark/cloud/run_3090_fix_100runs.py --parallel 5 --workers 2 "$@"
