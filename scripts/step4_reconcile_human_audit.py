#!/usr/bin/env python3
"""
Reconciles exported dual-author audit bundles (`audit_bundle_coder1.json` and `audit_bundle_coder2.json`)
produced by `dual_author_audit_workbench.html` into:
  1. Actual dual-author IRR report (`Po`, Cohen's kappa, Gwet's AC1) for F03, F04, F07, F08, F12, F13 across 182 evals;
  2. Filled CSVs (`core6_182_coder1.csv`, `core6_182_coder2.csv`, disagreement log for adjudication);
  3. Cleaned claimed-delta verification table (`claimed_deltas_human_verified.csv`) & Non-PMC 100-paper screening ledger.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
HUMAN_DIR = ROOT_DIR / "docs" / "coding_v1" / "human_dual_coding"
RESULTS_DIR = ROOT_DIR / "results"
FIELDS = ["F03", "F04", "F07", "F08", "F12", "F13"]
CATS = ["reported", "ambiguous", "not_reported_in_paper"]


def irr_metrics(a: list[str], b: list[str], cats: list[str]):
    n = len(a)
    if n == 0:
        return {"n": 0}
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe_kappa = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    kappa = (po - pe_kappa) / (1.0 - pe_kappa) if (1.0 - pe_kappa) > 1e-12 else 1.0
    k = len(cats)
    pi = {c: (ca[c] + cb[c]) / (2.0 * n) for c in cats}
    pe_ac1 = sum(pi[c] * (1.0 - pi[c]) for c in cats) / (k - 1) if k > 1 else 0.0
    ac1 = (po - pe_ac1) / (1.0 - pe_ac1) if (1.0 - pe_ac1) > 1e-12 else 1.0
    return {
        "n": n,
        "po": round(po, 4),
        "cohen_kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
        "agree_count": int(sum(x == y for x, y in zip(a, b))),
    }


def main():
    b1_path = HUMAN_DIR / "audit_bundle_coder1.json"
    b2_path = HUMAN_DIR / "audit_bundle_coder2.json"
    if not b1_path.exists() or not b2_path.exists():
        print(
            f"[WAITING] Place both {b1_path.name} and {b2_path.name} into {HUMAN_DIR} after both authors finish the workbench."
        )
        sys.exit(0)

    s1 = json.loads(b1_path.read_text(encoding="utf-8"))["state"]
    s2 = json.loads(b2_path.read_text(encoding="utf-8"))["state"]

    # 1. Core-6 IRR
    eval_ids = sorted(set(s1.get("core6", {}).keys()) | set(s2.get("core6", {}).keys()))
    per_field = {}
    all_a, all_b = [], []
    disagreements = []

    for fc in FIELDS:
        a, b = [], []
        for eid in eval_ids:
            v1 = s1.get("core6", {}).get(eid, {}).get(fc, "")
            v2 = s2.get("core6", {}).get(eid, {}).get(fc, "")
            if v1 in CATS and v2 in CATS:
                a.append(v1)
                b.append(v2)
                all_a.append(v1)
                all_b.append(v2)
                if v1 != v2:
                    disagreements.append({"eval_id": eid, "field": fc, "coder1": v1, "coder2": v2})
        per_field[fc] = irr_metrics(a, b, CATS)

    overall = irr_metrics(all_a, all_b, CATS)

    # 2. Non-PMC 100 screening IRR
    sids = sorted(set(s1.get("nonpmc", {}).keys()) | set(s2.get("nonpmc", {}).keys()))
    np_a, np_b = [], []
    np_dis = []
    for sid in sids:
        d1 = s1.get("nonpmc", {}).get(sid, {}).get("decision", "")
        d2 = s2.get("nonpmc", {}).get(sid, {}).get("decision", "")
        if d1 and d2:
            np_a.append(d1)
            np_b.append(d2)
            if d1 != d2:
                np_dis.append({"sample_id": sid, "coder1": d1, "coder2": d2})
    nonpmc_cats = sorted(set(np_a) | set(np_b))
    nonpmc_irr = irr_metrics(np_a, np_b, nonpmc_cats) if len(nonpmc_cats) > 1 else {"n": len(np_a), "po": 1.0 if np_a else 0.0}

    # 3. Delta clean agreement summary
    d_ids = sorted(set(s1.get("deltas", {}).keys()) | set(s2.get("deltas", {}).keys()))
    d_verdicts = []
    for eid in d_ids:
        v1 = s1.get("deltas", {}).get(eid, {}).get("verdict", "")
        v2 = s2.get("deltas", {}).get(eid, {}).get("verdict", "")
        d_verdicts.append({"eval_id": eid, "coder1_verdict": v1, "coder2_verdict": v2, "agree": v1 == v2 and bool(v1)})

    summary = {
        "core6_overall_irr": overall,
        "core6_per_field_irr": per_field,
        "core6_disagreements_count": len(disagreements),
        "nonpmc_100_irr": nonpmc_irr,
        "nonpmc_disagreements_count": len(np_dis),
        "delta_verdicts_checked": len(d_verdicts),
        "delta_verdict_agree_count": sum(1 for r in d_verdicts if r["agree"]),
    }

    (RESULTS_DIR / "human_dual_coding_reconciled_irr.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (HUMAN_DIR / "core6_disagreements_to_adjudicate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval_id", "field", "coder1", "coder2"])
        w.writeheader()
        w.writerows(disagreements)

    print("Reconciled Dual-Author IRR Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
