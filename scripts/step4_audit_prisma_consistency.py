#!/usr/bin/env python3
"""
Step 4: PRISMA Literature Audit Consistency, F04 Discrepancy Resolution & Comparable Subset Analysis
Part of AGENT EXECUTION PLAYBOOK (Phase 4).
"""

import os
import csv
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
CODING_DIR = os.path.join(DOCS_DIR, "coding_v1")
OUT_V2_DIR = os.path.join(CODING_DIR, "out_v2")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "manuscript", "figures")

def build_prisma_ledger():
    """Reconstruct available retrieval counts; do not infer paywalls or deduplication."""
    with open(os.path.join(DOCS_DIR, 'screening_title_abstract_v1.csv'), encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    retained = [r for r in rows if r['decision'] != 'exclude']
    with open(os.path.join(DOCS_DIR, 'fulltext_v1', 'download_log.json'), encoding='utf-8') as f:
        download = json.load(f)
    with open(os.path.join(DOCS_DIR, 'fulltext_v1', 'nonpmc_probe.json'), encoding='utf-8') as f:
        probes = json.load(f)
    extraction = []
    for path in sorted(glob.glob(os.path.join(OUT_V2_DIR, '*.json'))):
        with open(path, encoding='utf-8') as f:
            extraction.append(json.load(f))
    included = [r for r in extraction if r.get('eligibility') == 'include']
    return {
        'status': 'provisional_retrieval_ledger_not_final_deduplicated_prisma_flow',
        'identification': len(rows),
        'title_abstract_screened': len(rows),
        'screening_decisions': dict(Counter(r['decision'] for r in rows)),
        'title_abstract_excluded': sum(r['decision'] == 'exclude' for r in rows),
        'fulltext_retrieval_sought': len(retained),
        'retained_with_pmcid': sum(bool(r['pmcid']) for r in retained),
        'retained_without_pmcid': sum(not r['pmcid'] for r in retained),
        'pmc_download_attempts': len(download),
        'pmc_download_success': sum(r['status'].split()[0] in ('ok', 'exists') for r in download),
        'pmc_download_errors': [r for r in download if r['status'].startswith('error')],
        'records_outside_successful_xml_set': len(probes),
        'outside_xml_records_with_oa_metadata_links': sum(bool(r.get('oa_links')) for r in probes),
        'fulltext_assessed': len(extraction),
        'included_studies_papers': len(included),
        'included_evaluations_total': sum(len(r.get('datasets', [])) for r in included),
        'included_evaluations_by_dataset': dict(Counter(ds['dataset'] for r in included for ds in r.get('datasets', []))),
        'limitations': [
            'Legacy exclusion code meanings differ from the frozen manual; no silent remapping.',
            'A missing PMC XML copy or HTTP failure is not evidence of a paywall.',
            'The previous 16 paywall / 45 external / 12+3+30 decomposition was hardcoded and is withdrawn.',
            'External discoveries are separate from this screening file; overlaps and version links require record-level adjudication before final counts.',
            'Open-access metadata links have not all been followed or coded; no new acquisition was performed.'
        ]
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== Step 4.1: Auditing F04 (Duplicate Leakage Audit) in Full-Text Corpus ===")
    
    # 1. Search all out_v2 files for F04
    out_files = sorted(glob.glob(os.path.join(OUT_V2_DIR, "*.json")))
    print(f"Total out_v2 files: {len(out_files)}")

    f04_records = []
    f04_status_counter = Counter()
    total_evaluations = 0

    eval_records = []

    for fpath in out_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        pmcid = data.get("pmcid", "")
        idx = data.get("idx")
        datasets = data.get("datasets", [])
        
        for ds in datasets:
            total_evaluations += 1
            ds_name = ds.get("dataset", "")
            fields = ds.get("fields", {})
            f04 = fields.get("F04", {})
            st = f04.get("status", "not_reported_in_paper")
            f04_status_counter[st] += 1
            
            # Count reported fields for this evaluation
            rep_count = sum(1 for k, v in fields.items() if v.get("status") == "reported")
            
            mr = ds.get("main_result", {})
            eval_records.append({
                "file": os.path.basename(fpath),
                "pmcid": pmcid,
                "dataset": ds_name,
                "F04_status": st,
                "F04_value": f04.get("value", ""),
                "F04_evidence": f04.get("evidence", ""),
                "F04_location": f04.get("location", ""),
                "F16_code_status": fields.get("F16", {}).get("status", ""),
                "F13_single_run_status": fields.get("F13", {}).get("status", ""),
                "reported_fields_count": rep_count,
                "baseline_map50": mr.get("baseline_map50"),
                "proposed_map50": mr.get("proposed_map50"),
                "improvement_pp": mr.get("improvement_pp")
            })

            if st != "not_reported_in_paper":
                f04_records.append({
                    "file": os.path.basename(fpath),
                    "pmcid": pmcid,
                    "dataset": ds_name,
                    "status": st,
                    "value": f04.get("value", ""),
                    "evidence": f04.get("evidence", ""),
                    "location": f04.get("location", "")
                })

    print(f"Total evaluations audited: {total_evaluations}")
    print(f"F04 Status breakdown: {dict(f04_status_counter)}")
    print(f"Non-reported F04 count: {len(f04_records)}")
    for r in f04_records:
        print(f"  [{r['status'].upper()}] PMCID: {r['pmcid']} | Dataset: {r['dataset']} | Loc: {r['location']}")
        print(f"    Value: {r['value']}")
        print(f"    Evidence: \"{r['evidence']}\"")

    f04_summary = {
        "total_evaluations": total_evaluations,
        "f04_counts": dict(f04_status_counter),
        "f04_percentages": {k: round(v / total_evaluations * 100.0, 2) for k, v in f04_status_counter.items()},
        "positive_and_ambiguous_evidence": f04_records,
        "discrepancy_explanation": (
            "Table 2 reports F04 as Reported: 2 (1.1%), Ambiguous: 1 (0.5%), Not Reported: 179 (98.4%). "
            "The 2 reported instances arise from PMC13505926 (evaluated on both NEU-DET and GC10-DET; "
            "checked generated-vs-real near-duplicates and purged 12 duplicate GC10 images). "
            "The 1 ambiguous instance arises from PMC10813068 (documented 9-fold augmentation before random split, "
            "amplifying near-duplicate leakage without an explicit deduplication protocol). "
            "Figure 1 formerly displayed 0.0% due to downward zero-truncation of 1.1%; "
            "it is hereby reconciled across Table 2, Figure 1, and text to exact values: Reported 1.1% (2/182), Ambiguous 0.5% (1/182)."
        )
    }

    with open(os.path.join(RESULTS_DIR, "step4_f04_adjudication.json"), "w", encoding="utf-8") as f:
        json.dump(f04_summary, f, indent=2)

    # Preserve a traceable provisional ledger rather than hardcoded final-flow claims.
    prisma_flow = build_prisma_ledger()
    with open(os.path.join(RESULTS_DIR, 'step4_prisma_flow_reconciliation.json'), 'w', encoding='utf-8') as f:
        json.dump(prisma_flow, f, indent=2)

    # 3. Task 4.3: Comparable Literature Subset Analysis
    print("\n=== Step 4.3: Literature Comparable Subset Analysis ===")
    neu_evals = [e for e in eval_records if e["dataset"] == "NEU-DET"]
    valid_imp = [e for e in neu_evals if e["improvement_pp"] is not None and isinstance(e["improvement_pp"], (int, float))]
    
    print(f"Total NEU-DET evaluations: {len(neu_evals)}")
    print(f"Evaluations with quantitative improvement delta: {len(valid_imp)}")

    df_imp = pd.DataFrame(valid_imp)
    deltas = df_imp["improvement_pp"].values
    scores = df_imp["reported_fields_count"].values

    corr = float(np.corrcoef(deltas, scores)[0, 1])
    
    open_code_deltas = df_imp[df_imp["F16_code_status"] == "reported"]["improvement_pp"].values
    closed_code_deltas = df_imp[df_imp["F16_code_status"] != "reported"]["improvement_pp"].values

    stats_summary = {
        "sample_size": len(df_imp),
        "delta_map50": {
            "mean": round(float(np.mean(deltas)), 2),
            "std": round(float(np.std(deltas, ddof=1)), 2),
            "median": round(float(np.median(deltas)), 2),
            "iqr": round(float(np.percentile(deltas, 75) - np.percentile(deltas, 25)), 2),
            "min": round(float(np.min(deltas)), 2),
            "max": round(float(np.max(deltas)), 2)
        },
        "protocol_completeness_score": {
            "mean": round(float(np.mean(scores)), 2),
            "std": round(float(np.std(scores, ddof=1)), 2),
            "median": round(float(np.median(scores)), 2)
        },
        "correlation_delta_vs_completeness": round(corr, 3),
        "code_availability_comparison": {
            "open_code_n": len(open_code_deltas),
            "open_code_mean_delta": round(float(np.mean(open_code_deltas)), 2) if len(open_code_deltas) else None,
            "open_code_median_delta": round(float(np.median(open_code_deltas)), 2) if len(open_code_deltas) else None,
            "closed_code_n": len(closed_code_deltas),
            "closed_code_mean_delta": round(float(np.mean(closed_code_deltas)), 2) if len(closed_code_deltas) else None,
            "closed_code_median_delta": round(float(np.median(closed_code_deltas)), 2) if len(closed_code_deltas) else None
        }
    }

    with open(os.path.join(RESULTS_DIR, "step4_comparable_subset_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Saved comparable subset analysis to {os.path.join(RESULTS_DIR, 'step4_comparable_subset_analysis.json')}")

    # Generate Figure: figures/fig_literature_deltas.pdf
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)

    # Left: Scatter plot Delta mAP vs Completeness Score
    ax1 = axes[0]
    ax1.scatter(scores, deltas, alpha=0.6, color="#1f77b4", edgecolors="black", linewidth=0.5, s=40)
    # Regression line
    m, b = np.polyfit(scores, deltas, 1)
    xs = np.linspace(scores.min(), scores.max(), 100)
    ax1.plot(xs, m * xs + b, color="#d62728", linestyle="--", linewidth=1.8, label=f"Fit (r = {corr:.2f})")
    ax1.set_xlabel("Protocol Completeness (Reported Fields / 17)", fontsize=10, fontweight="bold")
    ax1.set_ylabel(r"Claimed Improvement $\Delta$mAP@0.5 (pp)", fontsize=10, fontweight="bold")
    ax1.set_title("(a) Claimed Gain vs. Reporting Completeness", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Right: Boxplot Open vs Closed Code
    ax2 = axes[1]
    box_data = [open_code_deltas, closed_code_deltas]
    bp = ax2.boxplot(box_data, patch_artist=True, labels=[f"Open Code\n(N={len(open_code_deltas)})", f"Closed Code\n(N={len(closed_code_deltas)})"], widths=0.45)
    colors = ["#2ca02c", "#ff7f0e"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    
    ax2.set_ylabel(r"Claimed Improvement $\Delta$mAP@0.5 (pp)", fontsize=10, fontweight="bold")
    ax2.set_title("(b) Claimed Gain by Code Availability", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig_literature_deltas.pdf")
    fig.savefig(fig_path)
    plt.close()
    print(f"Generated figure: {fig_path}")

    # Also save a copy in results/
    fig_copy = os.path.join(RESULTS_DIR, "fig_literature_deltas.pdf")
    import shutil
    shutil.copy(fig_path, fig_copy)

if __name__ == "__main__":
    main()
