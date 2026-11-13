#!/usr/bin/env python3
"""
WS-G: External Subscription Database Deduplication and Stratified Random Sampler.

Loads exported records from Scopus (.csv), Web of Science (.txt/.tsv/.csv), or IEEE Xplore (.csv)
placed in `论文2/data_external/raw/`, deduplicates against the 433-record Europe PMC ledger
(`论文2/docs/search_europepmc_20261030/candidates.csv`), and draws a fixed-seed (20261111)
stratified random sample of N=60 primary non-PMC papers + 40 ordered reserve papers.

Usage:
  python scripts/stepG_dedupe_and_sample.py
  python scripts/stepG_dedupe_and_sample.py --self-test
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
PMC_CANDIDATES_CSV = ROOT_DIR / "docs" / "search_europepmc_20261030" / "candidates.csv"
RAW_EXTERNAL_DIR = ROOT_DIR / "data_external" / "raw"
OUT_PACK_DIR = ROOT_DIR / "docs" / "外部检索包_20261111"
RESULTS_DIR = ROOT_DIR / "results"


def norm_doi(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def norm_title(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(s).lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def pick_col(row: dict, candidates: list[str]) -> str:
    lower_map = {k.strip().lower().lstrip("\ufeff"): v for k, v in row.items() if k is not None}
    for c in candidates:
        c_low = c.lower()
        if c_low in lower_map and lower_map[c_low] is not None:
            val = str(lower_map[c_low]).strip()
            if val:
                return val
    return ""


def parse_external_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    if not lines:
        return []
    delim = "\t" if (path.suffix.lower() in {".txt", ".tsv"} or lines[0].count("\t") > lines[0].count(",")) else ","
    reader = csv.DictReader(lines, delimiter=delim)
    records = []
    for row in reader:
        title = pick_col(row, ["Title", "Document Title", "Article Title", "TI", "item_title"])
        doi = norm_doi(pick_col(row, ["DOI", "DI", "doi"]))
        year_raw = pick_col(row, ["Year", "Publication Year", "PY", "coverDate", "year"])
        m_yr = re.search(r"(20\d{2})", year_raw)
        year = int(m_yr.group(1)) if m_yr else 0
        journal = pick_col(row, ["Source title", "Publication Title", "Journal", "SO", "Publication Name"])
        abstract = pick_col(row, ["Abstract", "AB", "abstract"])
        keywords = pick_col(row, ["Author Keywords", "Index Keywords", "Keywords", "DE", "ID"])
        pmid = pick_col(row, ["PubMed ID", "PMID", "PM", "pmid"])
        pmcid = pick_col(row, ["PMCID", "pmcid"])
        doctype = pick_col(row, ["Document Type", "DT", "type", "Content Type"])
        authors = pick_col(row, ["Authors", "AU", "AF", "Author Full Names"])
        if not title and not doi:
            continue
        blob = f"{title} {abstract} {keywords}".lower()
        mentions_bench = any(
            k in blob
            for k in ["neu-det", "gc10-det", "neu surface", "gc10", "northeastern university surface"]
        )
        mentions_steel_det = ("steel" in blob and "defect" in blob) and any(
            k in blob for k in ["yolo", "detect", "inspection", "r-cnn", "detr", "fpn"]
        )
        records.append(
            {
                "source_file": path.name,
                "title": title,
                "norm_title": norm_title(title),
                "doi": doi,
                "year": year,
                "journal": journal,
                "authors": authors,
                "doctype": doctype,
                "pmid": pmid,
                "pmcid": pmcid,
                "abstract": abstract,
                "keywords": keywords,
                "mentions_benchmark": mentions_bench,
                "mentions_steel_detection": mentions_steel_det,
            }
        )
    return records


def load_pmc_ledger(csv_path: Path) -> tuple[set[str], set[str], set[str]]:
    dois, titles, pmids = set(), set(), set()
    if not csv_path.exists():
        return dois, titles, pmids
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            d = norm_doi(row.get("doi", ""))
            t = norm_title(row.get("title", ""))
            p = str(row.get("pmid", "")).strip()
            if d:
                dois.add(d)
            if t:
                titles.add(t)
            if p:
                pmids.add(p)
    return dois, titles, pmids


def year_stratum(year: int) -> str:
    if year <= 2022:
        return "2020-2022"
    if year <= 2024:
        return "2023-2024"
    return "2025-2026"


def run_pipeline(raw_dir: Path, out_dir: Path, seed: int = 20261111, n_primary: int = 60, n_reserve: int = 40) -> dict:
    pmc_dois, pmc_titles, pmc_pmids = load_pmc_ledger(PMC_CANDIDATES_CSV)
    files = sorted(
        [
            p
            for p in raw_dir.glob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".tsv"} and not p.name.startswith(".")
        ]
    )
    if not files:
        raise FileNotFoundError(f"未在 {raw_dir} 找到导出文件（.csv/.txt/.tsv）。请先按检索操作单放入导出文件。")

    raw_records = []
    for f in files:
        raw_records.extend(parse_external_file(f))

    # Deduplicate across exported files
    seen_dois, seen_titles = set(), set()
    union_deduped = []
    for r in raw_records:
        if r["year"] and (r["year"] < 2020 or r["year"] > 2026):
            continue
        key_d = r["doi"]
        key_t = r["norm_title"]
        if (key_d and key_d in seen_dois) or (key_t and key_t in seen_titles):
            continue
        if key_d:
            seen_dois.add(key_d)
        if key_t:
            seen_titles.add(key_t)
        union_deduped.append(r)

    # Filter out records already in Europe PMC 433-record ledger or carrying a PMCID
    nonpmc_pool = []
    overlap_pmc_count = 0
    for r in union_deduped:
        in_pmc = (
            bool(r["pmcid"])
            or (r["doi"] and r["doi"] in pmc_dois)
            or (r["norm_title"] and r["norm_title"] in pmc_titles)
            or (r["pmid"] and r["pmid"] in pmc_pmids)
        )
        if in_pmc:
            overlap_pmc_count += 1
            continue
        if r["mentions_benchmark"] or r["mentions_steel_detection"]:
            nonpmc_pool.append(r)

    # Prioritize explicit benchmark mentions or sample across eligible pool stratified by year
    rng = np.random.default_rng(seed)
    strata = {"2020-2022": [], "2023-2024": [], "2025-2026": []}
    for r in nonpmc_pool:
        strata[year_stratum(r["year"])].append(r)

    for k in strata:
        idxs = rng.permutation(len(strata[k]))
        strata[k] = [strata[k][i] for i in idxs]

    total_pool = len(nonpmc_pool)
    target_total = min(total_pool, n_primary + n_reserve)
    # Proportional allocation across year strata
    selected = []
    remaining = []
    for k, items in strata.items():
        quota = int(round(n_primary * (len(items) / max(1, total_pool))))
        selected.extend(items[:quota])
        remaining.extend(items[quota:])

    rng_rem = rng.permutation(len(remaining))
    remaining = [remaining[i] for i in rng_rem]
    while len(selected) < min(n_primary, total_pool) and remaining:
        selected.append(remaining.pop(0))
    if len(selected) > n_primary:
        extra = selected[n_primary:]
        selected = selected[:n_primary]
        remaining = extra + remaining

    reserve = remaining[:n_reserve]

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "role",
        "year",
        "stratum",
        "doi",
        "doi_url",
        "title",
        "journal",
        "authors",
        "doctype",
        "mentions_benchmark",
        "pdf_filename_to_save",
        "screening_decision_E1_E6",
        "notes",
    ]

    download_rows = []
    for idx, r in enumerate(selected, 1):
        sid = f"G{idx:03d}"
        download_rows.append(
            {
                "sample_id": sid,
                "role": "primary",
                "year": r["year"],
                "stratum": year_stratum(r["year"]),
                "doi": r["doi"],
                "doi_url": f"https://doi.org/{r['doi']}" if r["doi"] else "",
                "title": r["title"],
                "journal": r["journal"],
                "authors": r["authors"],
                "doctype": r["doctype"],
                "mentions_benchmark": r["mentions_benchmark"],
                "pdf_filename_to_save": f"{sid}.pdf",
                "screening_decision_E1_E6": "",
                "notes": "",
            }
        )
    for idx, r in enumerate(reserve, n_primary + 1):
        sid = f"R{idx:03d}"
        download_rows.append(
            {
                "sample_id": sid,
                "role": "reserve",
                "year": r["year"],
                "stratum": year_stratum(r["year"]),
                "doi": r["doi"],
                "doi_url": f"https://doi.org/{r['doi']}" if r["doi"] else "",
                "title": r["title"],
                "journal": r["journal"],
                "authors": r["authors"],
                "doctype": r["doctype"],
                "mentions_benchmark": r["mentions_benchmark"],
                "pdf_filename_to_save": f"{sid}.pdf",
                "screening_decision_E1_E6": "",
                "notes": "",
            }
        )

    dl_csv = out_dir / "全文下载清单.csv"
    with dl_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(download_rows)

    summary = {
        "seed": seed,
        "input_files": [f.name for f in files],
        "raw_records_loaded": len(raw_records),
        "union_deduped_2020_2026": len(union_deduped),
        "overlap_with_europepmc_excluded": overlap_pmc_count,
        "eligible_nonpmc_candidate_pool": len(nonpmc_pool),
        "primary_sample_count": len(selected),
        "reserve_sample_count": len(reserve),
        "output_download_list": str(dl_csv),
    }
    (out_dir / "stepG_sampling_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_d = tmp_path / "raw"
        out_d = tmp_path / "out"
        raw_d.mkdir()
        fake_csv = raw_d / "scopus_test.csv"
        rows = [
            {
                "Title": "YOLO-RMF: A Real-Time Surface Defect Detection Model for Hot-Rolled Strip Steel with Reparameterized Convolution and Axial Attention.",
                "DOI": "10.3390/s26206533",
                "Year": "2026",
                "Source title": "Sensors",
                "Abstract": "Evaluated on NEU-DET dataset.",
                "Document Type": "Article",
            }
        ]
        for i in range(1, 85):
            rows.append(
                {
                    "Title": f"Synthetic IEEE Steel Defect Detection Study {i} on NEU-DET and GC10-DET",
                    "DOI": f"10.1109/tim.2025.{10000+i}",
                    "Year": str(2020 + (i % 7)),
                    "Source title": "IEEE Transactions on Instrumentation and Measurement",
                    "Abstract": "We evaluate bounding-box object detection on NEU-DET and GC10-DET using YOLO.",
                    "Document Type": "Article",
                }
            )
        with fake_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        res = run_pipeline(raw_d, out_d, seed=20261111, n_primary=60, n_reserve=20)
        assert res["overlap_with_europepmc_excluded"] == 1, f"Expected 1 PMC overlap, got {res}"
        assert res["eligible_nonpmc_candidate_pool"] == 84, f"Expected 84 non-PMC, got {res}"
        assert res["primary_sample_count"] == 60
        assert res["reserve_sample_count"] == 20
        print("[SELF-TEST PASS]", json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_EXTERNAL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_PACK_DIR)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--n-primary", type=int, default=60)
    ap.add_argument("--n-reserve", type=int, default=40)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        run_self_test()
        return
    summary = run_pipeline(args.raw_dir, args.out_dir, args.seed, args.n_primary, args.n_reserve)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
