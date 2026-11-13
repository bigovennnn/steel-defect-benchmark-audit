# How Large Must a Gain Be? Seed Noise, Split Choice and Near-Duplicate Leakage in Steel Surface Defect Detection Benchmarks

Official open-science companion repository for the manuscript:

> **How Large Must a Gain Be? Seed Noise, Split Choice and Near-Duplicate Leakage in Steel Surface Defect Detection Benchmarks**  
> Wenzhuo Liu and Shiling Yu (2026)

---

## Overview

This repository provides all code, data-hygiene audit artifacts, exact split manifests, custom attention/neck modules, and statistical reproduction scripts supporting our empirical audit and controlled benchmark study on **NEU-DET** and **GC10-DET**:

1. **Literature Reporting Completeness Audit (`audit_data/`)**:
   - 433-record PRISMA screening table (`audit_data/europepmc_combined_screening.csv`), PRISMA reconciliation (`results/step4_prisma_flow_reconciliation.json`), and first-pass model-assisted extraction records (`audit_data/first_pass_coding_v2/` across 135 screened full texts + `audit_data/rollup_20261103.json` for **131 included studies / 182 dataset evaluations**).
   - Claimed mAP@0.5 deltas (`results/step4_claimed_deltas_182.csv`, `.json`; raw extraction pending full manual cleaning and dual-author core-6 coding in `audit_data/human_dual_coding/core6_182_blank_coder{1,2}.csv`).
2. **Two-Stage Near-Duplicate Detection & Human Verification (`benchmark/near_duplicate_20261101/`, `results/human_audit/`)**:
   - FFT translation screening + 64x64 sub-pixel phase correlation / 200x200 crop overlap check + dual pixel (NCC >= 0.95) and gradient (NCC >= 0.90) SE(2) verification (`223` registration edges across `1,797` NEU-DET images).
   - Two-author blind human verification of **349 candidate image pairs** (`results/human_audit/pair_labels_coder1.csv`, `pair_labels_coder2.csv`, `pair_labels_consensus.csv`, `internal_pool_mapping.json`; on the 223 registration edges: `83.9%` agreement, Gwet's AC1 = `0.805`; plus `64` additional human-confirmed overlapping pairs outside the graph component analyzed as lower-bound caveats on `G0`–`G9` and `K0`–`K9`).
3. **Registration-Grouped & Coil-Grouped Benchmark Splits (`benchmark/split_designs/`, `benchmark/datasets/*/split.json`, `results/step2_v6_splits_manifest.json`)**:
   - **NEU-DET (`NEU-Grouped`)**: 1,437 train / 180 val / 180 test images across 1,593 grouping units (147 multi-image registration components + 1,446 singletons) (`benchmark/split_designs/neu_grouped_v6_split.json`, `benchmark/datasets/neu_grouped_v6/split.json`; 0 cross-split registration edges; SHA-256: `1a635ccfc6a61b3bdeb8f198f4812dc7693ef93954bc138ff9294f01ec11ce85`).
   - **GC10-DET (`GC10-CoilGrouped`)**: 1,831 train / 230 val / 230 test images across 467 coil prefixes (`benchmark/split_designs/gc10_grouped_v1_split.json`, `benchmark/datasets/gc10_grouped_v1/split.json`; `2,291` valid images from `2,294` raw files; SHA-256: `ead84d9fe29c2678a2e30ccd0b2b99d2af1ad8e91452f777b1094cb0a191d7d1`).
4. **Multi-Seed Benchmarks, Split-ANOVA, Counterfactual Partner-Removal, Fixed-FP Escape Audit & Architectural Modules (`benchmark/cloud/`, `scripts/`)**:
   - **297-run Cloud GPU Suite (`results/cloud_297_runs_manifest.csv`, `results/cloud_297_comprehensive_stats.json`, 297 files in `results/cloud_297_runs/`)**:
     - `Baseline-10S` (100 A100 runs across Colab A & B): 5 detectors (`yolo11n`, `yolov8n`, `yolo11s`, `yolov8s`, `rtdetr-l`) x 10 seeds (`s0`–`s9`) on `NEU-Grouped` and `GC10-CoilGrouped` (single-run 95% threshold $\Delta_{\min} = 1.96\sqrt{2}\,\sigma_{\text{seed}}$: 2.2–3.5 pp on NEU-DET, 4.0–5.5 pp on GC10-DET).
     - `Split-ANOVA` (30 RTX 3090 runs): 10 registration-grouped splits (`G0`–`G9`) x 3 seeds (`7.9` pp cross-split $\Delta_{\min}$; split variance 82.8% vs. seed variance 17.1%, $F(9,20)=15.50$).
     - `CF-K10` matched counterfactual partner-removal (100 RTX 3090 runs; +4.08 pp leaked-image mAP@0.5, +0.38 pp full-test mAP@0.5 [-0.29, +1.06], -2.6 to -3.0 pp absolute fixed-FP leaked-image escape drop) & 5-level `CF-Dose` scaling (45 RTX 3090 runs; Page's $L=438.0, p=0.028$).
     - `Mod-10S` + `Split×Model` Interaction Queue (`benchmark/cloud/run_3090_fix_100runs.{py,sh}`): fixes the Ultralytics YAML `model.ckpt=None` loader bug so all 4 variants (`base`, `+CBAM`, `+EMA`, `+BiFPN`) verify `493/499` COCO weight transfer (`Transferred ... items from pretrained weights`) before rerunning the 40 paired `Mod-10S` runs and adding 60 `G0`–`G9 × {yolo11n-base, yolo11n-bifpn} × 3 seeds` interaction runs on RTX 3090.

---

## Repository Structure

```text
├── audit_data/                  # 433-record screening CSV, 135 first-pass JSONs & blank core-6 dual-coding sheets
│   ├── first_pass_coding_v2/    # 135 paper-level first-pass model-assisted JSON records
│   └── human_dual_coding/       # Blank dual-author core-6 sheets (182 evals), 26-paper IRR manifest & report
├── benchmark/
│   ├── cloud/                   # Multi-seed / multi-architecture / module replication suite
│   │   ├── custom_modules.py    # CBAM, EMA, BiFPNFusion & verified 493/499 COCO checkpoint loader
│   │   ├── run_cloud_suite.py   # Unified training & evaluation runner
│   │   ├── run_3090_fix_100runs.py # Preflight-gated 100-run RTX 3090 queue (40 Mod-10S + 60 Split×Model)
│   │   └── models/              # YOLO11n-base, CBAM, EMA, BiFPN YAML architectures
│   ├── datasets/                # Frozen split.json manifests (neu_grouped_v6, gc10_grouped_v1)
│   ├── near_duplicate_20261101/ # Two-stage FFT + SE(2) NCC near-duplicate detection & MIP solver
│   └── split_designs/           # Frozen JSON split manifests (incl. neu_grouped_v6_split, gc10_grouped_v1_split, K10, G1-G9)
├── manuscript/
│   ├── main.tex                 # Springer Nature (`sn-jnl`, `sn-apa` APA 7th) LaTeX manuscript source
│   ├── main.pdf                 # Compiled manuscript PDF
│   ├── sn-jnl.cls               # Springer Nature journal class file
│   ├── sn-apacite.bst           # Springer Nature APA 7th BibTeX style
│   ├── refs.bib                 # Verified BibTeX bibliography
│   ├── figures/                 # Vector PDF figures (`figA`–`figF`, `fig02`–`fig06b`)
│   └── figures_src/             # Matplotlib reproduction scripts (`make_stats_jim.py`, `figA`–`figF`)
├── results/                     # JSON/CSV statistical outputs, human audit records & 297+24 run summaries
└── scripts/                     # End-to-end reproduction scripts (splits, 297-run stats, verification)
```

---

## Quick Start & Reproduction

### 1. Generate Registration-Grouped & Coil-Grouped Splits
```bash
python scripts/step2_generate_new_splits_v6.py
python scripts/step2_generate_gc10_grouped_v1.py
```

### 2. Re-run Statistical Analysis & Manuscript Verification
```bash
python scripts/step3_analyze_all_297_cloud_runs.py
python scripts/step5_verify_manuscript_numbers.py
```

### 3. Run Verified 100-Run RTX 3090 Module + Interaction Queue
```bash
# Runs CPU preflight (493/499 COCO transfer check), then 40 Mod-10S + 60 Split×Model runs
nohup bash benchmark/cloud/run_3090_fix_100runs.sh --shutdown > fix100.log 2>&1 &
```

### 4. Regenerate Publication Figures (`figA`–`figF`)
```bash
python manuscript/figures_src/make_stats_jim.py
for f in manuscript/figures_src/fig*.py; do python "$f"; done
```

---

## License

This code and benchmark manifest package is released under the [MIT License](LICENSE).
