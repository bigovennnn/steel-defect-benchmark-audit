#!/usr/bin/env python3
"""
Creates a compact, self-contained upload archive `.runtime/cloud_bundle/paper2_cloud_bundle.zip`
for running the cloud GPU suites on AutoDL / Featurize.

Instead of duplicating NEU-DET images 22 times across `neu_grouped_v6` and `neu_k00..09_{L,C}`,
it packs each unique NEU-DET image/label once (`raw_pool/neu/`) and each unique GC10-DET image/label once
(`raw_pool/gc10/`), plus `results/step2_v6_splits_manifest.json`, `split.json` files, weights, and scripts.
On the cloud server, running `python benchmark/cloud/unpack_on_cloud.py` reconstructs all 22 datasets
in ~2 seconds using hardlinks!
"""

import json
import os
import zipfile
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent
OUT_DIR = ROOT_DIR / ".runtime" / "cloud_bundle"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "paper2_cloud_bundle.zip"

    neu_v6 = BENCH_DIR / "datasets" / "neu_grouped_v6"
    gc10_v1 = BENCH_DIR / "datasets" / "gc10_grouped_v1"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Code & configs
        for rel in [
            "benchmark/cloud/custom_modules.py",
            "benchmark/cloud/run_cloud_suite.py",
            "benchmark/cloud/run_all_cloud.py",
            "benchmark/cloud/run_all_cloud.sh",
            "benchmark/cloud/run_3090_fix_100runs.py",
            "benchmark/cloud/run_3090_fix_100runs.sh",
            "benchmark/cloud/run_3090_stage2_expansions.py",
            "benchmark/cloud/unpack_on_cloud.py",
            "benchmark/cloud/pack_cloud_results.py",
            "benchmark/cloud/models/yolo11n-base.yaml",
            "benchmark/cloud/models/yolo11n-cbam.yaml",
            "benchmark/cloud/models/yolo11n-ema.yaml",
            "benchmark/cloud/models/yolo11n-bifpn.yaml",
            "benchmark/audits/neu_candidate_edges_223_20261101.csv",
            "benchmark/near_duplicate_20261101/candidate_pairs_hs_v1.json",
            "benchmark/split_designs/neu_grouped_v6_split.json",
            # G1..G9 are required by run_3090_fix_100runs.py (split_interaction); missing them broke the 2026-11-13 smoke run
            *[f"benchmark/split_designs/neu_grouped_g{g:02d}_split.json" for g in range(1, 10)],
            "benchmark/split_designs/gc10_grouped_v1_split.json",
            "benchmark/split_designs/neu_counterfactual_k10_splits.json",
            "benchmark/split_designs/neu_naive_random_split_20261103.json",
            "results/step2_v6_splits_manifest.json",
            "results/step3_counterfactual_splits_manifest.json",
            "results/counterfactual_leaked_partners_k10.csv",
            "results/counterfactual_leaked_partners_m3_single_split.csv",
            "benchmark/datasets/neu_grouped_v6/split.json",
            "benchmark/datasets/gc10_grouped_v1/split.json",
            "benchmark/weights/yolo11n_coco_official.pt",
        ]:
            p = ROOT_DIR / rel
            if p.exists():
                zf.write(p, arcname=rel)

        # Also place one-click entrypoints at the archive root for convenience
        for root_entry in [
            "run_all_cloud.py",
            "run_all_cloud.sh",
            "run_3090_fix_100runs.py",
            "run_3090_fix_100runs.sh",
        ]:
            p = CLOUD_DIR / root_entry
            if p.exists():
                zf.write(p, arcname=root_entry)

        # 2. Unique NEU-DET images & labels (1,797)
        for part in ["train", "val", "test"]:
            for p in (neu_v6 / "images" / part).glob("*.jpg"):
                zf.write(p, arcname=f"raw_pool/neu/images/{p.name}")
            for p in (neu_v6 / "labels" / part).glob("*.txt"):
                zf.write(p, arcname=f"raw_pool/neu/labels/{p.name}")

        # 3. Unique GC10-DET images & labels (2,291)
        for part in ["train", "val", "test"]:
            for p in (gc10_v1 / "images" / part).glob("*.jpg"):
                zf.write(p, arcname=f"raw_pool/gc10/images/{p.name}")
            for p in (gc10_v1 / "labels" / part).glob("*.txt"):
                zf.write(p, arcname=f"raw_pool/gc10/labels/{p.name}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created cloud upload bundle: {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
