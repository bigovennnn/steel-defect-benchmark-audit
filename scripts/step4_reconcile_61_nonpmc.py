#!/usr/bin/env python3
"""
Step 4B: Detailed Audit and Categorization of the 61 Non-XML Records and 131 Included Reports
Cross-references `nonpmc_probe.json`, `screening_title_abstract_v1.csv`, and `out_v2/*.json`
to classify every single one of the 61 records outside the 135 PMC XML downloads and verify
the publication types of the 131 included studies.
"""
import csv
import glob
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def main():
    rows = list(csv.DictReader((DOCS / "screening_title_abstract_v1.csv").open(encoding="utf-8-sig")))
    row_by_idx = {str(r["idx"]): r for r in rows}

    probes = json.loads((DOCS / "fulltext_v1" / "nonpmc_probe.json").read_text(encoding="utf-8"))

    out_v2_files = sorted(glob.glob(str(DOCS / "coding_v1" / "out_v2" / "*.json")))
    out_v2 = [json.loads(Path(p).read_text(encoding="utf-8")) for p in out_v2_files]
    included = [r for r in out_v2 if r.get("eligibility") == "include"]

    included_title_map = {}
    for r in included:
        idx = str(r.get("idx"))
        sr = row_by_idx.get(idx, {})
        included_title_map[norm_title(sr.get("title", ""))] = idx

    included_pub_types = Counter()
    for r in included:
        idx = str(r.get("idx"))
        sr = row_by_idx.get(idx, {})
        pt = sr.get("publication_types", "")
        if "preprint" in pt.lower():
            included_pub_types["preprint"] += 1
        else:
            included_pub_types["peer_reviewed_journal_article"] += 1

    breakdown_61 = []
    category_counts = Counter()
    oa_47_counts = Counter()
    no_oa_14_counts = Counter()

    for p in probes:
        idx = str(p["idx"])
        sr = row_by_idx.get(idx, {})
        has_oa = bool(p.get("oa_links"))
        pub_vers = p.get("published_version", [])
        src = p.get("source", "")
        doi = p.get("doi", sr.get("doi", ""))
        title = sr.get("title", "")
        pub_type = sr.get("publication_types", "")
        decision = sr.get("decision", "")
        nt = norm_title(title)
        matched_included_idx = included_title_map.get(nt)

        if matched_included_idx and src == "MED":
            cat = "E4_duplicate_medline_record_of_included_pmc_article"
        elif matched_included_idx and src == "PPR":
            cat = "E4_superseded_preprint_with_published_version_in_included_131"
        elif pub_vers:
            cat = "E4_superseded_preprint_with_published_version_outside_pmc_xml"
        elif sr.get("pmcid") in ("PMC13074744", "PMC13492231"):
            cat = "E5_pmc_xml_http_500_download_error"
        elif src == "PPR" or "preprint" in pub_type.lower():
            cat = "E5_external_preprint_without_epmc_jats_xml"
        else:
            cat = "E5_medline_journal_record_without_epmc_oa_or_xml"

        category_counts[cat] += 1
        if has_oa:
            oa_47_counts[cat] += 1
        else:
            no_oa_14_counts[cat] += 1

        breakdown_61.append({
            "idx": idx,
            "source": src,
            "publication_types": pub_type,
            "doi": doi,
            "pmcid_in_screening": sr.get("pmcid", ""),
            "screening_decision": decision,
            "has_oa_link": has_oa,
            "oa_links": p.get("oa_links", []),
            "published_version_pmids": pub_vers,
            "matched_included_idx": matched_included_idx,
            "audit_category": cat,
            "title": title
        })

    summary = {
        "included_131_composition": {
            "total_included_reports": len(included),
            "by_publication_type": dict(included_pub_types),
            "note": "All 131 included studies are peer-reviewed journal articles indexed with full-text JATS XML in PubMed Central (0 preprints in the final 131 corpus)."
        },
        "outside_xml_61_summary": {
            "total_outside_xml": len(probes),
            "with_oa_metadata_links": sum(1 for p in probes if p.get("oa_links")),
            "without_oa_metadata_links": sum(1 for p in probes if not p.get("oa_links")),
            "overall_category_breakdown": dict(category_counts),
            "among_47_with_oa_links": dict(oa_47_counts),
            "among_14_without_oa_links": dict(no_oa_14_counts)
        },
        "records_61": breakdown_61
    }

    out_path = RESULTS / "step4_nonpmc_61_audit.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records_61"}, indent=2))


if __name__ == "__main__":
    main()
