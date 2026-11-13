#!/usr/bin/env python3
"""
Step 5.2: Build DMLR Single-Blind Reproducibility Package & SHA-256 Manifest
Part of AGENT EXECUTION PLAYBOOK (Phase 5).
"""

import os
import sys
import glob
import json
import shutil
import hashlib

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_DIR = os.path.join(ROOT_DIR, "reproducibility_package")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
AUDITS_DIR = os.path.join(ROOT_DIR, "benchmark", "audits")
DATASETS_DIR = os.path.join(ROOT_DIR, "benchmark", "datasets")

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Step 5.2: Building Reproducibility Package for DMLR Submission ===")
    if os.path.exists(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    
    os.makedirs(os.path.join(PKG_DIR, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(PKG_DIR, "data_audit"), exist_ok=True)
    os.makedirs(os.path.join(PKG_DIR, "scripts"), exist_ok=True)

    # 1. Copy Manifests
    frozen_split = os.path.join(DATASETS_DIR, "neu_v5_frozen", "split.json")
    naive_split = os.path.join(DATASETS_DIR, "neu_naive_random", "split.json")
    edges_csv = os.path.join(AUDITS_DIR, "neu_candidate_edges_223_20261101.csv")

    if os.path.exists(frozen_split):
        shutil.copy2(frozen_split, os.path.join(PKG_DIR, "manifests", "neu_v5_frozen.json"))
    if os.path.exists(naive_split):
        shutil.copy2(naive_split, os.path.join(PKG_DIR, "manifests", "neu_naive_random.json"))
    if os.path.exists(edges_csv):
        shutil.copy2(edges_csv, os.path.join(PKG_DIR, "manifests", "neu_candidate_edges_223.csv"))

    # 2. Copy Data Audit and Results
    result_files = [
        "step1_our_neudet_stats.json",
        "step1_competitor_alignment.json",
        "step2_geometric_registration_def.json",
        "step2_method_comparison.json",
        "step2_mip_solution_report.json",
        "step3_counterfactual_splits_manifest.json",
        "step3_counterfactual_ablation.json",
        "step4_f04_adjudication.json",
        "step4_prisma_flow_reconciliation.json",
        "step4_comparable_subset_analysis.json",
        "step5_verification_report.json"
    ]
    for rf in result_files:
        src = os.path.join(RESULTS_DIR, rf)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PKG_DIR, "data_audit", rf))

    # 3. Copy Execution Scripts
    for sf in glob.glob(os.path.join(SCRIPTS_DIR, "step*.py")):
        shutil.copy2(sf, os.path.join(PKG_DIR, "scripts", os.path.basename(sf)))

    # 4. Generate Reproducibility Package README
    readme_content = """# Reproducibility Artifact Package: Steel Surface Defect Detection Benchmark & Literature Audit
Target Journal: Journal of Data-centric Machine Learning Research (DMLR)

## Contents Overview
This local snapshot preserves selected manifests, provisional reports and execution scripts. It is not a complete standalone reproduction environment or a published submission artifact. Copied scripts retain repository-relative dependencies; execute the maintained originals from the repository, with the original datasets and environment available.

## Evidence and execution status (2026-11-05 revision)
- Human visual labels and independent literature coding are pending. The sole active literature coding package is `../docs/coding_v1/human_dual_coding/`; the old 2,828-row package is retired.
- `step2_method_comparison.json` and its PR curve are pool-membership proxy diagnostics with circular NCC labels. They do not establish comparative detection performance. The candidate pool omits thumbnail-method top candidates and random controls, so human labeling alone would not resolve pool selection bias.
- The reported +11.00 pp is a difference between two test subsets, with SD over three training seeds. Image sampling uncertainty, subset difficulty and validation contamination prevent a causal interpretation.
- Counterfactual runs remain `awaiting_gpu_runs`; setup files contain no measured causal effect.
- The retrieval ledger is provisional; absent PMC XML is not equivalent to a paywall. Source-layer verification, version deduplication, reference checks and publication review are pending.
- Text-pattern checks do not establish source authenticity or publication readiness. No public release URL is claimed by this local snapshot.

### Directory Structure
```
reproducibility_package/
├── README.md                           # This datasheet and reproducibility guide
├── SHA256SUMS.txt                      # Cryptographic checksums of all package artifacts
├── manifests/                          # Canonical dataset split and candidate edge manifests
│   ├── neu_v5_frozen.json              # Frozen leak-mitigated partition (1,437 / 180 / 180)
│   ├── neu_naive_random.json           # Unconstrained naive stratified random partition
│   └── neu_candidate_edges_223.csv     # 223 algorithmic candidate edges, not human gold labels
├── data_audit/                         # Machine-readable audit reports and empirical matrices
│   ├── step1_our_neudet_stats.json     # NEU-DET 1,797 image & 4,180 box instance statistics
│   ├── step1_competitor_alignment.json # Cross-alignment with Zehao Li (2026 Figshare v2)
│   ├── step2_geometric_registration_def.json # Formal SE(2) 2D rigid planar registration spec
│   ├── step2_method_comparison.json    # Circular proxy-label diagnostic, not method superiority
│   ├── step2_mip_solution_report.json  # HiGHS mixed-integer programming solver solution report
│   ├── step3_counterfactual_splits_manifest.json # M=3 draws matched-size ablation setup
│   ├── step3_counterfactual_ablation.json        # Pending-run status; no measured effect sizes
│   ├── step4_f04_adjudication.json     # Verbatim quotes & resolution of F04 reporting
│   ├── step4_prisma_flow_reconciliation.json # Full PRISMA 2020 flow & 45 external records
│   ├── step4_comparable_subset_analysis.json # 122 comparable NEU-DET study delta analysis
│   └── step5_verification_report.json  # Automated LaTeX consistency verification report
└── scripts/                            # Code snapshots requiring the original repository
    ├── step1_audit_dataset_version.py
    ├── step2_build_verification_pool.py
    ├── step2_compare_similarity_methods.py
    ├── step2_solve_mip_partition.py
    ├── step3_run_counterfactual_ablation.py
    ├── step3_summarize_counterfactual.py
    ├── step4_audit_prisma_consistency.py
    ├── step4_prepare_dual_coding.py
    ├── step4_compute_irr_metrics.py
    └── step5_verify_manuscript_numbers.py
```

## Quickstart & Replication
1. **Environment Setup**:
   ```bash
   pip install torch torchvision opencv-python scipy scikit-learn pandas matplotlib
   ```
2. **Verify Dataset Version & Competitor Alignment**:
   ```bash
   python scripts/step1_audit_dataset_version.py
   ```
3. **Inspect the exploratory proxy PR diagnostic (no superiority claim)**:
   ```bash
   python scripts/step2_compare_similarity_methods.py
   ```
4. **Solve Integer Linear Programming Partition**:
   ```bash
   python scripts/step2_solve_mip_partition.py
   ```
5. **Verify Full Manuscript Consistency**:
   ```bash
   python scripts/step5_verify_manuscript_numbers.py
   ```
"""
    readme_path = os.path.join(PKG_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 5. Generate SHA256SUMS.txt
    checksum_lines = []
    for root, _, files in os.walk(PKG_DIR):
        for fname in sorted(files):
            if fname == "SHA256SUMS.txt":
                continue
            full_p = os.path.join(root, fname)
            rel_p = os.path.relpath(full_p, PKG_DIR).replace("\\", "/")
            digest = compute_sha256(full_p)
            checksum_lines.append(f"{digest}  {rel_p}")

    sums_path = os.path.join(PKG_DIR, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    print(f"Reproducibility package built successfully at: {PKG_DIR}")
    print(f"Total packaged artifacts: {len(checksum_lines)}")

if __name__ == "__main__":
    main()
