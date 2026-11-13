#!/usr/bin/env python3
"""
WS-E: Verbatim-Grounded Extraction of Claimed Delta mAP@0.5 Across 182 Evaluations (131 PMC Papers).

Parses the full JATS XML (`docs/fulltext_v1/xml/<PMCID>.xml`) and existing coding JSONs (`docs/coding_v1/out_v2/*.json`),
purges any unverified placeholder numbers, and extracts verbatim-verified (baseline_map50, proposed_map50, delta_map50_pp)
from:
  1. Verbatim evidence strings in `out_v2/*.json` when both numbers or explicit delta appear in text,
  2. Structured `<table-wrap>` rows in the JATS XML matching either dataset caption or the known `proposed_map50` cell,
  3. Explicit comparative sentences in the abstract / results text.

Outputs:
  - `results/step4_claimed_deltas_182.json`
  - `results/step4_claimed_deltas_182.csv` (for dual-author verification)
"""

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
XML_DIR = ROOT_DIR / "docs" / "fulltext_v1" / "xml"
CODING_DIR = ROOT_DIR / "docs" / "coding_v1" / "out_v2"
RESULTS_DIR = ROOT_DIR / "results"


def tag(elem) -> str:
    return elem.tag.split("}")[-1] if elem.tag else ""


def text_of(elem) -> str:
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def extract_xml_text_and_tables(xml_path: Path):
    if not xml_path.exists():
        return "", []
    root = ET.parse(xml_path).getroot()
    paragraphs = []
    for p in root.iter():
        if tag(p) in {"abstract", "p", "title"}:
            t = text_of(p)
            if len(t) > 15:
                paragraphs.append(t)

    tables = []
    for idx, tw in enumerate([e for e in root.iter() if tag(e) == "table-wrap"], 1):
        lbl = ""
        cap = ""
        rows = []
        for ch in tw.iter():
            t_name = tag(ch)
            if t_name == "label" and not lbl:
                lbl = text_of(ch)
            elif t_name == "caption" and not cap:
                cap = text_of(ch)
            elif t_name == "tr":
                cells = [text_of(c) for c in ch if tag(c) in {"th", "td"}]
                if cells:
                    rows.append(cells)
        tables.append({"index": idx, "label": lbl or f"Table {idx}", "caption": cap, "rows": rows})
    return "\n".join(paragraphs), tables


def ds_matches_text(dataset: str, text: str) -> bool:
    t = text.lower()
    if dataset == "NEU-DET":
        return any(k in t for k in ["neu-det", "neu dataset", "neu surface", "northeastern", "hot-rolled", "six defect", "neu "])
    if dataset == "GC10-DET":
        return any(k in t for k in ["gc10-det", "gc10", "gc-10", "ten defect", "10 defect"])
    return False


def parse_cell_map50(cell_str: str):
    m_num = re.search(r"(\d{2}\.\d+|\d\.\d{3,4}|\d{2})", cell_str.strip())
    if not m_num:
        return None
    val = float(m_num.group(1))
    if 0.40 <= val <= 0.999:
        val = round(val * 100.0, 2)
    if 40.0 <= val <= 99.9:
        return round(val, 2)
    return None


def try_extract_from_tables(tables: list[dict], dataset: str, known_m50: float = None):
    other_ds = "gc10" if dataset == "NEU-DET" else "neu"
    best_candidate = None

    for tb in tables:
        cap = f"{tb['label']} {tb['caption']}"
        rows = tb["rows"]
        if len(rows) < 2:
            continue

        # Strategy 1: If known_m50 is provided, check if any column in this table contains known_m50
        candidate_cols = []
        if known_m50 is not None:
            for r in rows[1:]:
                for c_idx, cell in enumerate(r):
                    v = parse_cell_map50(cell)
                    if v is not None and abs(v - known_m50) <= 0.11 and c_idx > 0:
                        candidate_cols.append(c_idx)

        # Strategy 2: Header column matching mAP
        for r in rows[:3]:
            for c_idx, cell in enumerate(r):
                c_low = cell.lower()
                if ("map" in c_low or "ap50" in c_low or "ap@0.5" in c_low) and not (
                    "50:95" in c_low or "50-95" in c_low or "0.5:0.95" in c_low or "75" in c_low
                ):
                    if c_idx not in candidate_cols:
                        candidate_cols.append(c_idx)

        for map_col in candidate_cols:
            parsed_rows = []
            for r in rows[1:]:
                if len(r) <= map_col or not r[0]:
                    continue
                v = parse_cell_map50(r[map_col])
                if v is not None:
                    parsed_rows.append((r[0].strip(), v, " | ".join(r)))
            if len(parsed_rows) < 2:
                continue

            # Find proposed row
            proposed = None
            if known_m50 is not None:
                for pr in reversed(parsed_rows):
                    if abs(pr[1] - known_m50) <= 0.11:
                        proposed = pr
                        break
            if proposed is None:
                proposed = parsed_rows[-1]
                for pr in parsed_rows:
                    if any(k in pr[0].lower() for k in ["ours", "proposed", "improved"]):
                        proposed = pr
                        break

            # Find baseline row in the same table
            baseline = None
            for pr in parsed_rows:
                if pr[0] == proposed[0]:
                    continue
                if any(k in pr[0].lower() for k in ["baseline", "base", "original", "✓" if False else "0"]):
                    if pr[1] < proposed[1]:
                        baseline = pr
                        break
            if baseline is None:
                yolo_rows = [
                    pr
                    for pr in parsed_rows
                    if pr[0] != proposed[0]
                    and any(
                        k in pr[0].lower()
                        for k in ["yolov", "yolo11", "yolo10", "yolo9", "yolo8", "yolo7", "yolo5", "rt-detr", "rtdetr", "faster"]
                    )
                    and pr[1] < proposed[1]
                ]
                if yolo_rows:
                    baseline = yolo_rows[-1]
                else:
                    lower_rows = [pr for pr in parsed_rows if pr[1] < proposed[1]]
                    if lower_rows:
                        baseline = lower_rows[0]

            if baseline and proposed and 0.05 <= (proposed[1] - baseline[1]) <= 25.0:
                cand = {
                    "proposed_model": proposed[0],
                    "proposed_map50": round(proposed[1], 2),
                    "baseline_model": baseline[0],
                    "baseline_map50": round(baseline[1], 2),
                    "delta_pp": round(proposed[1] - baseline[1], 2),
                    "method": f"xml_table ({tb['label']})",
                    "location": f"{tb['label']}: {tb['caption'][:80]}",
                    "evidence": f"Baseline [{baseline[2]}] vs. Proposed [{proposed[2]}]",
                }
                if ds_matches_text(dataset, cap) or (
                    known_m50 is not None and abs(proposed[1] - known_m50) <= 0.11 and other_ds not in cap.lower()
                ):
                    return cand
                if best_candidate is None and other_ds not in cap.lower():
                    best_candidate = cand

    return best_candidate


def try_extract_from_sentences(full_text: str, dataset: str, known_m50: float = None):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", full_text)
    for i, s in enumerate(sentences):
        window = " ".join(sentences[max(0, i - 1) : min(len(sentences), i + 2)])
        if not ds_matches_text(dataset, window) and (
            known_m50 is None or str(known_m50) not in s
        ):
            continue
        s_low = s.lower()
        if "map" not in s_low and "ap" not in s_low:
            continue

        m_from_to = re.search(
            r"(?:from|by)\s+(\d{2}(?:\.\d+)?)\s*%?\s+to\s+(\d{2}(?:\.\d+)?)\s*%", s, flags=re.I
        )
        if m_from_to:
            b = float(m_from_to.group(1))
            p = float(m_from_to.group(2))
            if 40.0 <= b < p <= 99.9 and (p - b) <= 25.0:
                return {
                    "proposed_map50": round(p, 2),
                    "baseline_map50": round(b, 2),
                    "delta_pp": round(p - b, 2),
                    "method": "sentence_from_to",
                    "evidence": s.strip(),
                }

        m_pair = re.search(
            r"(\d{2}(?:\.\d+)?)\s*%[^.;]{0,140}?(?:impro\w+|increas\w+|higher|outperform\w+|gain\w*|up|exceed\w*)\s+(?:by\s+|of\s+)?(\d{1,2}(?:\.\d+)?)\s*(?:%|percentage\s*points?|pp)",
            s,
            flags=re.I,
        )
        if m_pair:
            p = float(m_pair.group(1))
            d = float(m_pair.group(2))
            if 45.0 <= p <= 99.9 and 0.1 <= d <= 20.0:
                return {
                    "proposed_map50": round(p, 2),
                    "baseline_map50": round(p - d, 2),
                    "delta_pp": round(d, 2),
                    "method": "sentence_map_and_delta",
                    "evidence": s.strip(),
                }

        m_rev = re.search(
            r"(?:impro\w+|increas\w+|higher|gain\w*|surpass\w*)\s+[^.;]{0,60}?(?:by|of)\s+(\d{1,2}(?:\.\d+)?)\s*(?:%|percentage\s*points?|pp)[^.;]{0,120}?(\d{2}(?:\.\d+)?)\s*%",
            s,
            flags=re.I,
        )
        if m_rev:
            d = float(m_rev.group(1))
            p = float(m_rev.group(2))
            if 45.0 <= p <= 99.9 and 0.1 <= d <= 20.0:
                return {
                    "proposed_map50": round(p, 2),
                    "baseline_map50": round(p - d, 2),
                    "delta_pp": round(d, 2),
                    "method": "sentence_delta_and_map",
                    "evidence": s.strip(),
                }
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CODING_DIR.glob("*.json"))

    records = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("eligibility") != "include":
            continue
        pmcid = d["pmcid"]
        xml_path = XML_DIR / f"{pmcid}.xml"
        full_text, tables = extract_xml_text_and_tables(xml_path)

        for ds in d.get("datasets", []):
            ds_name = ds.get("dataset")
            if ds_name not in {"NEU-DET", "GC10-DET"}:
                continue
            mr = ds.get("main_result", {}) or {}
            imp = mr.get("improvement_pp")
            ev = str(mr.get("evidence") or "").strip()
            m50 = mr.get("map50")
            b50 = mr.get("baseline_map50")

            ev_nums = [float(x) for x in re.findall(r"\d+\.\d+|\d+", ev)]
            is_placeholder = (imp == 2.5 and "2.5" not in ev and "0.025" not in ev) or (
                m50 == 75.0 and imp == 2.0
            )
            known_m50 = float(m50) if (m50 is not None and not (m50 == 75.0 and imp == 2.0)) else None

            has_verbatim_match = (
                not is_placeholder
                and imp is not None
                and ev
                and (
                    any(abs(x - float(imp)) < 0.06 or abs(x * 100 - float(imp)) < 0.06 for x in ev_nums)
                    or (
                        m50 is not None
                        and b50 is not None
                        and any(abs(x - float(m50)) < 0.06 or abs(x * 100 - float(m50)) < 0.06 for x in ev_nums)
                        and any(abs(x - float(b50)) < 0.06 or abs(x * 100 - float(b50)) < 0.06 for x in ev_nums)
                    )
                )
            )

            if has_verbatim_match:
                records.append(
                    {
                        "coding_file": f.name,
                        "pmcid": pmcid,
                        "dataset": ds_name,
                        "detector_family": d.get("detector_family", ""),
                        "proposed_model": mr.get("model", ""),
                        "proposed_map50": m50,
                        "baseline_model": mr.get("baseline", ""),
                        "baseline_map50": b50,
                        "claimed_delta_map50_pp": round(float(imp), 2),
                        "verification_source": "coding_v2_verbatim_verified",
                        "location": mr.get("location", ""),
                        "verbatim_evidence": ev,
                        "F13_repeated_runs": ds["fields"]["F13"]["status"],
                        "F14_baseline_source": ds["fields"]["F14"]["status"],
                    }
                )
                continue

            tb_hit = try_extract_from_tables(tables, ds_name, known_m50=known_m50)
            sent_hit = try_extract_from_sentences(full_text, ds_name, known_m50=known_m50)
            hit = tb_hit or sent_hit

            if hit:
                records.append(
                    {
                        "coding_file": f.name,
                        "pmcid": pmcid,
                        "dataset": ds_name,
                        "detector_family": d.get("detector_family", ""),
                        "proposed_model": hit.get("proposed_model", mr.get("model", "")),
                        "proposed_map50": hit["proposed_map50"],
                        "baseline_model": hit.get("baseline_model", mr.get("baseline", "")),
                        "baseline_map50": hit["baseline_map50"],
                        "claimed_delta_map50_pp": hit["delta_pp"],
                        "verification_source": hit["method"],
                        "location": hit.get("location", mr.get("location", "XML fulltext")),
                        "verbatim_evidence": hit["evidence"],
                        "F13_repeated_runs": ds["fields"]["F13"]["status"],
                        "F14_baseline_source": ds["fields"]["F14"]["status"],
                    }
                )
            else:
                records.append(
                    {
                        "coding_file": f.name,
                        "pmcid": pmcid,
                        "dataset": ds_name,
                        "detector_family": d.get("detector_family", ""),
                        "proposed_model": mr.get("model", ""),
                        "proposed_map50": known_m50,
                        "baseline_model": mr.get("baseline", ""),
                        "baseline_map50": b50 if not is_placeholder else None,
                        "claimed_delta_map50_pp": None,
                        "verification_source": "no_explicit_map50_baseline_pair_in_xml",
                        "location": mr.get("location", ""),
                        "verbatim_evidence": ev,
                        "F13_repeated_runs": ds["fields"]["F13"]["status"],
                        "F14_baseline_source": ds["fields"]["F14"]["status"],
                    }
                )

    verified_deltas = [
        r["claimed_delta_map50_pp"]
        for r in records
        if r["claimed_delta_map50_pp"] is not None and 0.0 < r["claimed_delta_map50_pp"] <= 25.0
    ]
    neu_deltas = [
        r["claimed_delta_map50_pp"]
        for r in records
        if r["dataset"] == "NEU-DET" and r["claimed_delta_map50_pp"] is not None and 0.0 < r["claimed_delta_map50_pp"] <= 25.0
    ]
    gc10_deltas = [
        r["claimed_delta_map50_pp"]
        for r in records
        if r["dataset"] == "GC10-DET" and r["claimed_delta_map50_pp"] is not None and 0.0 < r["claimed_delta_map50_pp"] <= 25.0
    ]

    def stats_dict(arr):
        a = np.asarray(arr, dtype=float)
        return {
            "n": len(a),
            "mean": round(float(np.mean(a)), 2),
            "sd": round(float(np.std(a, ddof=1)), 2),
            "median": round(float(np.median(a)), 2),
            "q25": round(float(np.percentile(a, 25)), 2),
            "q75": round(float(np.percentile(a, 75)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
            "pct_le_2pp": round(float(np.mean(a <= 2.0) * 100.0), 1),
            "pct_le_3pp": round(float(np.mean(a <= 3.0) * 100.0), 1),
            "pct_le_4pp": round(float(np.mean(a <= 4.0) * 100.0), 1),
        }

    summary = {
        "total_evaluations": len(records),
        "verbatim_verified_count": len(verified_deltas),
        "all_verified": stats_dict(verified_deltas),
        "neu_det_verified": stats_dict(neu_deltas),
        "gc10_det_verified": stats_dict(gc10_deltas),
        "records": records,
    }

    (RESULTS_DIR / "step4_claimed_deltas_182.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RESULTS_DIR / "step4_claimed_deltas_182.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    print("Summary of Verbatim-Verified Claimed Delta mAP@0.5 (v2):")
    print("  All  :", summary["all_verified"])
    print("  NEU  :", summary["neu_det_verified"])
    print("  GC10 :", summary["gc10_det_verified"])


if __name__ == "__main__":
    main()
