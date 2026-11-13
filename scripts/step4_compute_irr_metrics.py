#!/usr/bin/env python3
"""Compare raw, source-matched independent labels in the single active package.

No historical-retest fallback, no adjudicated labels, and no row-wise bootstrap.
An undefined kappa stays null, including in bootstrap replicates.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT_DIR / "docs/coding_v1/human_dual_coding"
SHEET_PATH = PACKAGE_DIR / "blank_dual_coding_sheet.csv"
KEY_PATH = PACKAGE_DIR / "coder1_key.csv"
REPORT_PATH = PACKAGE_DIR / "irr_report.json"
FIELDS = [f"F{i:02d}" for i in range(1, 18)]
# Fixed categories, including genuine non-applicability. Missing evidence is not
# an agreement category. Final cross-source 'not_reported' is a separate layer.
CATEGORIES = ["reported", "ambiguous", "not_reported_in_paper", "not_applicable"]
ALLOWED = set(CATEGORIES) | {"not_assessed", ""}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def point_metrics(counts):
    n = sum(counts.values())
    matrix = {a: {b: counts[a, b] for b in CATEGORIES} for a in CATEGORIES}
    if not n:
        return dict(n_samples=0, observed_agreement_Po=None, cohen_expected_Pe=None,
                    cohen_kappa=None, cohen_kappa_reason="no_assessed_pairs",
                    gwet_expected_Pe=None, gwet_ac1=None, confusion_matrix=matrix)
    first, second = Counter(), Counter()
    for (a, b), count in counts.items():
        first[a] += count
        second[b] += count
    po = sum(counts[a, a] for a in CATEGORIES) / n
    pe = sum(first[a] * second[a] / n**2 for a in CATEGORIES)
    kappa = (po - pe) / (1 - pe) if 1 - pe > 1e-12 else None
    pooled = {a: (first[a] + second[a]) / (2 * n) for a in CATEGORIES}
    pe_ac1 = sum(p * (1 - p) for p in pooled.values()) / (len(CATEGORIES) - 1)
    ac1 = (po - pe_ac1) / (1 - pe_ac1) if 1 - pe_ac1 > 1e-12 else None
    return dict(n_samples=n, observed_agreement_Po=po, cohen_expected_Pe=pe,
                cohen_kappa=kappa,
                cohen_kappa_reason="Pe_equals_one" if kappa is None else None,
                gwet_expected_Pe=pe_ac1, gwet_ac1=ac1, confusion_matrix=matrix)


def percentile(values, probability):
    values = sorted(values)
    at = (len(values) - 1) * probability
    low = int(at)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def compute_irr(rater1, rater2, paper_ids, n_bootstrap=1000, seed=42):
    if not (len(rater1) == len(rater2) == len(paper_ids)):
        raise ValueError("Both label lists and paper IDs must have equal lengths")
    if n_bootstrap < 0:
        raise ValueError("Bootstrap count must be nonnegative")
    if any(x not in CATEGORIES for x in list(rater1) + list(rater2)):
        raise ValueError("IRR accepts assessed paper-layer labels only")
    if any(not str(x).strip() for x in paper_ids):
        raise ValueError("Paper IDs are required for clustered bootstrap")
    counts = Counter(zip(rater1, rater2))
    result = point_metrics(counts)
    clusters = defaultdict(Counter)
    for a, b, paper in zip(rater1, rater2, paper_ids):
        clusters[paper][a, b] += 1
    result["n_papers"] = len(clusters)
    result["categories"] = CATEGORIES
    targets = {"observed_agreement_Po": "observed_agreement_Po_ci95",
               "cohen_kappa": "cohen_kappa_ci95", "gwet_ac1": "gwet_ac1_ci95"}
    result.update({name: None for name in targets.values()})
    bootstrap = dict(unit="paper", requested_replicates=n_bootstrap,
                     completed_replicates=0, seed=seed,
                     undefined_replicates={name: 0 for name in targets})
    result["bootstrap"] = bootstrap
    if not n_bootstrap or len(clusters) < 2:
        bootstrap["status"] = "not_requested" if not n_bootstrap else "fewer_than_two_papers"
        return result
    rng = random.Random(seed)
    papers = sorted(clusters)
    draws = {name: [] for name in targets}
    for _ in range(n_bootstrap):
        sampled = Counter()
        # Draw papers with replacement; keep every dataset and field in a paper.
        for _ in papers:
            sampled.update(clusters[rng.choice(papers)])
        values = point_metrics(sampled)
        for name in targets:
            if values[name] is None:
                bootstrap["undefined_replicates"][name] += 1
            else:
                draws[name].append(values[name])
    bootstrap["completed_replicates"] = n_bootstrap
    bootstrap["status"] = "computed"
    bootstrap["interval_rule"] = "95% percentile; null if any replicate is undefined; no conditional deletion"
    for name, destination in targets.items():
        if not bootstrap["undefined_replicates"][name]:
            result[destination] = [percentile(draws[name], 0.025), percentile(draws[name], 0.975)]
    return result


def join_labels(blind, key):
    def index(rows, source):
        ids = [r["item_id"] for r in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Missing or duplicate items in {source}")
        return {row["item_id"]: row for row in rows}
    blind_index = index(blind, "blind sheet")
    key_index = index(key, "primary key")
    if set(blind_index) != set(key_index):
        raise ValueError("Blind sheet and primary key item sets differ")
    joined = []
    for item_id, raw in blind_index.items():
        first = key_index[item_id]
        for col in ("eval_id", "paper_id", "pmcid", "dataset", "field_id"):
            if raw[col] != first[col] or not raw[col].strip():
                raise ValueError(f"Identity mismatch: {item_id}, {col}")
        if raw["field_id"] not in FIELDS:
            raise ValueError(f"Unknown field: {item_id}")
        second_status = raw["coder2_independent_status"].strip()
        first_status = first["coder1_paper_status"].strip()
        if second_status not in ALLOWED or first_status not in ALLOWED:
            raise ValueError(f"Invalid paper-layer status: {item_id}")
        verified = first["coder1_paper_layer_checked"].strip().lower()
        if verified not in ("", "yes", "no"):
            raise ValueError(f"Invalid primary verification marker: {item_id}")
        first_verified = verified == "yes" and bool(first_status) and all(
            first[c].strip() for c in ("coder1_verifier", "coder1_verification_date", "coder1_checked_scope"))
        second_supported = bool(raw["coder2_checked_scope"].strip())
        first_supported = first_verified
        if second_status in ("reported", "ambiguous"):
            second_supported &= bool(raw["coder2_independent_quote"].strip()
                                     and raw["coder2_independent_location"].strip())
        if first_status in ("reported", "ambiguous"):
            first_supported &= bool(first["coder1_paper_quote"].strip()
                                    and first["coder1_paper_location"].strip())
        if second_status == "not_applicable":
            second_supported &= bool(raw["coder2_independent_value"].strip() or raw["coder2_notes"].strip())
        if first_status == "not_applicable":
            first_supported &= bool(first["coder1_paper_value"].strip())
        joined.append({**raw, "audit_part": first["audit_part"],
                       "rater1": first_status, "rater2": second_status,
                       "first_verified": first_verified,
                       "first_supported": first_supported,
                       "second_supported": second_supported})
    return joined


def summarize(rows, n_bootstrap, seed):
    eligible = [r for r in rows if r["rater1"] in CATEGORIES and r["rater2"] in CATEGORIES
                and r["first_supported"] and r["second_supported"]]
    result = compute_irr([r["rater1"] for r in eligible], [r["rater2"] for r in eligible],
                         [r["paper_id"] for r in eligible], n_bootstrap, seed)
    result["total_planned_items"] = len(rows)
    result["coder2_status_counts"] = dict(Counter(r["rater2"] or "blank" for r in rows))
    result["primary_paper_status_counts"] = dict(Counter(r["rater1"] or "blank" for r in rows))
    result["primary_source_verified_labels"] = sum(r["first_verified"] for r in rows)
    result["coder2_labels_missing_evidence"] = sum(
        r["rater2"] in CATEGORIES and not r["second_supported"] for r in rows)
    result["primary_labels_missing_evidence"] = sum(
        r["rater1"] in CATEGORIES and not r["first_supported"] for r in rows)
    result["excluded_from_irr"] = len(rows) - len(eligible)
    return result


def build_report(blind, key, n_bootstrap=1000, seed=42):
    rows = join_labels(blind, key)
    filled = sum(bool(r["rater2"]) for r in rows)
    overall = summarize(rows, n_bootstrap, seed)
    if not filled:
        status = "awaiting_independent_labels"
    elif filled < len(rows):
        status = "partial_independent_labels"
    elif overall["n_samples"] < len(rows):
        status = "labels_complete_awaiting_evidence_or_primary_verification"
    else:
        status = "paired_paper_layer_labels_complete"
    by_part = defaultdict(list)
    for row in rows:
        by_part[row["audit_part"]].append(row)
    return {
        "status": status, "total_planned_items": len(rows),
        "total_coder2_labels": filled, "total_filled_pairs": overall["n_samples"],
        "comparison": "source-verified primary paper-layer raw status vs independent coder2 paper-layer raw status",
        "legacy_first_codes_used_for_irr": False, "adjudicated_labels_used": False,
        "historical_retest_fallback_used": False,
        "independence_and_human_identity_verified_by_script": False,
        "limitations": [
            "Source verification and human independence require process records; a script cannot certify them.",
            "Legacy AI-assisted primary codes remain preserved in the key and are not silently treated as verified human labels.",
            "Not-assessed and blank labels are excluded; not-applicable remains its own category.",
            "Pooled heterogeneous fields are descriptive; inspect field-specific results.",
            "Paper-cluster intervals describe this preserved convenience-source sample, not all benchmark literature.",
            "Low agreement alone does not identify whether disagreement arose from papers, the manual, or coders.",
        ],
        "overall": overall,
        "per_field": {f: summarize([r for r in rows if r["field_id"] == f], n_bootstrap, seed)
                      for f in FIELDS},
        "per_part": {p: summarize(group, n_bootstrap, seed) for p, group in sorted(by_part.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=SHEET_PATH)
    parser.add_argument("--key", type=Path, default=KEY_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    blind, key = read_rows(args.sheet), read_rows(args.key)
    # CLI accepts the one fixed sample only. Test helpers can use small fixtures.
    if len(blind) != 595 or len({r["paper_id"] for r in blind}) != 26:
        raise ValueError("Use the canonical 26-paper / 35-evaluation / 595-item package")
    manifest = json.loads((PACKAGE_DIR / "sample_26_manifest.json").read_text(encoding="utf-8-sig"))
    expected = {(Path(r["file"]).stem, r["dataset"], f) for r in manifest for f in FIELDS}
    if {(r["paper_id"], r["dataset"], r["field_id"]) for r in blind} != expected:
        raise ValueError("Sheet membership differs from the preserved manifest")
    report = build_report(blind, key, args.bootstrap, args.seed)
    report["blind_sheet"] = str(args.sheet.resolve())
    report["primary_key"] = str(args.key.resolve())
    protected = {args.sheet.resolve(), args.key.resolve(),
                 (PACKAGE_DIR / "sample_26_manifest.json").resolve(),
                 (PACKAGE_DIR / "sample_provenance.json").resolve()}
    if args.output.suffix.lower() != ".json" or args.output.resolve() in protected:
        raise ValueError("Output must be a report JSON, never raw labels or a preserved manifest")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Status: {report['status']}; coder2 labels {report['total_coder2_labels']}/595; comparable pairs {report['total_filled_pairs']}/595")
    print(f"Report: {args.output}")
    print("No historical fallback. Human identity and independence need process verification.")


if __name__ == "__main__":
    main()
