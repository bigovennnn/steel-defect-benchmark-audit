#!/usr/bin/env python3
"""
Compute Inter-Rater Reliability (Cohen's kappa, Gwet's AC1, and exact agreement Po)
and generate an adjudication worksheet for the 182 Core-6 + Claimed Delta dual-coding exports.

Usage:
  python scripts/step4_compute_core6_irr.py --coder-a path/to/Wenzhuo_Liu.csv --coder-b path/to/Shiling_Yu.csv
  python scripts/step4_compute_core6_irr.py --self-test
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["F03", "F04", "F07", "F08", "F12", "F13"]
CATEGORIES = ["Yes", "Partial", "No", "N/A"]


def compute_kappa_and_ac1(labels_a, labels_b, categories=CATEGORIES):
    assert len(labels_a) == len(labels_b) and len(labels_a) > 0
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    counts_a = {c: 0 for c in categories}
    counts_b = {c: 0 for c in categories}
    for a, b in zip(labels_a, labels_b):
        counts_a[a] = counts_a.get(a, 0) + 1
        counts_b[b] = counts_b.get(b, 0) + 1

    pe_cohen = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in categories)
    kappa = 1.0 if abs(1.0 - pe_cohen) < 1e-12 else (po - pe_cohen) / (1.0 - pe_cohen)

    q = len(categories)
    pi_k = {c: (counts_a.get(c, 0) + counts_b.get(c, 0)) / (2.0 * n) for c in categories}
    pe_gwet = (1.0 / (q - 1.0)) * sum(p * (1.0 - p) for p in pi_k.values())
    ac1 = 1.0 if abs(1.0 - pe_gwet) < 1e-12 else (po - pe_gwet) / (1.0 - pe_gwet)

    return {
        "n": n,
        "agree": agree,
        "po": round(po, 4),
        "pe_cohen": round(pe_cohen, 4),
        "kappa": round(kappa, 4),
        "pe_gwet": round(pe_gwet, 4),
        "ac1": round(ac1, 4),
    }


def load_coder_csv(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["eval_id"]] = r
    return rows


def evaluate_pair(path_a: Path, path_b: Path, out_json: Path, out_disagreements_csv: Path):
    rows_a = load_coder_csv(path_a)
    rows_b = load_coder_csv(path_b)
    common_ids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if not common_ids:
        raise ValueError("No overlapping eval_id rows between Coder A and Coder B CSVs.")

    report = {"n_evaluations": len(common_ids), "fields": {}, "delta_audit": {}}
    disagreements = []

    for f in FIELDS:
        la = [rows_a[eid].get(f"{f}_label", "").strip() for eid in common_ids]
        lb = [rows_b[eid].get(f"{f}_label", "").strip() for eid in common_ids]
        valid_pairs = [(eid, a, b) for eid, a, b in zip(common_ids, la, lb) if a and b]
        if valid_pairs:
            res = compute_kappa_and_ac1([x[1] for x in valid_pairs], [x[2] for x in valid_pairs])
            report["fields"][f] = res
            for eid, a, b in valid_pairs:
                if a != b:
                    disagreements.append({
                        "eval_id": eid,
                        "pmcid": rows_a[eid].get("pmcid", ""),
                        "dataset": rows_a[eid].get("dataset", ""),
                        "field": f,
                        "coder_a_label": a,
                        "coder_a_quote": rows_a[eid].get(f"{f}_quote", ""),
                        "coder_b_label": b,
                        "coder_b_quote": rows_b[eid].get(f"{f}_quote", ""),
                        "consensus_label": "",
                        "consensus_rationale": "",
                    })

    # Delta verification agreement
    da = [rows_a[eid].get("delta_verified", "").strip() for eid in common_ids]
    db = [rows_b[eid].get("delta_verified", "").strip() for eid in common_ids]
    valid_d = [(eid, a, b) for eid, a, b in zip(common_ids, da, db) if a and b]
    if valid_d:
        report["delta_audit"] = compute_kappa_and_ac1(
            [x[1] for x in valid_d],
            [x[2] for x in valid_d],
            categories=["Verified", "Corrected", "N/A_No_Baseline"],
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_disagreements_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_id", "pmcid", "dataset", "field",
                "coder_a_label", "coder_a_quote",
                "coder_b_label", "coder_b_quote",
                "consensus_label", "consensus_rationale",
            ],
        )
        writer.writeheader()
        writer.writerows(disagreements)

    return report, len(disagreements)


def run_self_test():
    a = ["No"] * 175 + ["Yes"] * 7
    b = ["No"] * 174 + ["Yes"] * 1 + ["No"] * 1 + ["Yes"] * 6
    stats = compute_kappa_and_ac1(a, b)
    assert stats["n"] == 182
    assert stats["po"] > 0.98
    assert stats["ac1"] > 0.98
    assert stats["kappa"] > 0.80
    print("[SELF-TEST PASS] step4_compute_core6_irr.py verified:", stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coder-a", type=Path, help="Coder A exported CSV")
    parser.add_argument("--coder-b", type=Path, help="Coder B exported CSV")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "core6_human_irr_182.json",
    )
    parser.add_argument(
        "--out-disagreements",
        type=Path,
        default=ROOT / "docs" / "coding_v1" / "human_dual_coding" / "adjudication_disagreements_182.csv",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.coder_a or not args.coder_b:
        parser.error("Provide --coder-a and --coder-b (or --self-test).")

    report, n_dis = evaluate_pair(args.coder_a, args.coder_b, args.out_json, args.out_disagreements)
    print(f"Saved IRR report to {args.out_json} ({n_dis} disagreements exported to {args.out_disagreements})")


if __name__ == "__main__":
    main()
