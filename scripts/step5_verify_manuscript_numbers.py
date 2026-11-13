#!/usr/bin/env python3
"""Comprehensive Manuscript & Benchmark Consistency Verification (Paper 2 / JIM sn-jnl)."""
import json
import os
import re
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
MANUSCRIPT_DIR = ROOT_DIR / "manuscript"
TEX_PATH = MANUSCRIPT_DIR / "main.tex"
BIB_PATH = MANUSCRIPT_DIR / "refs.bib"
BLG_PATH = MANUSCRIPT_DIR / "main.blg"
LOG_PATH = MANUSCRIPT_DIR / "main.log"
FIG05_SRC = MANUSCRIPT_DIR / "figures_src" / "fig05_duplicate_pipeline.py"
STATS_JSON = RESULTS_DIR / "cloud_297_comprehensive_stats.json"


def check(name: str, passed: bool, detail: str, bag: dict):
    bag[name] = {"pass": bool(passed), "detail": detail}


def main():
    print("=== Step 5: Comprehensive JIM Manuscript Verification ===")
    assert TEX_PATH.exists(), f"Missing {TEX_PATH}"
    assert BIB_PATH.exists(), f"Missing {BIB_PATH}"
    assert STATS_JSON.exists(), f"Missing {STATS_JSON}"

    tex = TEX_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    checks = {}

    # 1. Springer sn-jnl Author-Year structure and Introduction subheading ban
    intro_block = tex.split(r"\section{Introduction}")[1].split(r"\section{Related Work}")[0]
    has_sn = (
        r"\documentclass[pdflatex,sn-apa]{sn-jnl}" in tex
        or r"\documentclass[pdflatex,sn-basic]{sn-jnl}" in tex
    )
    intro_no_sub = r"\subsection" not in intro_block
    check(
        "springer_sn_jnl_structure",
        has_sn and intro_no_sub,
        f"sn-jnl[sn-apa/sn-basic]={has_sn}, zero_intro_subheadings={intro_no_sub}",
        checks,
    )

    # 2. Abstract compliance: 150-250 words, no $, no internal jargon/filenames
    abs_txt = ""
    idx_abs = tex.find(r"\abstract{")
    if idx_abs != -1:
        pos = idx_abs + len(r"\abstract{")
        depth = 1
        start = pos
        while pos < len(tex) and depth > 0:
            if tex[pos] == "{":
                depth += 1
            elif tex[pos] == "}":
                depth -= 1
            pos += 1
        abs_txt = tex[start : pos - 1]
    abs_words = len(abs_txt.split())
    abs_clean = (
        150 <= abs_words <= 250
        and "$" not in abs_txt
        and "F01" not in abs_txt
        and "neu_grouped" not in abs_txt
        and "CBAM" not in abs_txt
    )
    check(
        "abstract_springer_compliance",
        abs_clean,
        f"words={abs_words} (150-250), zero_math={'$' not in abs_txt}, zero_jargon={abs_clean}",
        checks,
    )

    # 3. Literature F04 (3 / 1.6%) and 131 / 182 counts + zero stale F04=1.1%
    f04_ok = (
        bool(re.search(r"\\textbf\{F04\}.*?\\textbf\{3 \(1\.6\\%\)\}.*?\\textbf\{178 \(97\.8\\%\)\}", tex))
        and "131 " in tex
        and "182 " in tex
    )
    check("literature_f04_and_corpus_counts", f04_ok, "F04=3 (1.6%) and 131/182 verified", checks)

    # 4. Terminology hygiene: no E2/E5/E7 experiment-ID collisions, no raw file paths, no pre-registered
    raw_terms = [
        "neu_v5_frozen",
        "neu_grouped_v6",
        "gc10_grouped_v1",
        "neu_naive_random",
        "e2_g00",
        "neu_k00",
        "pre-registered",
        "smoke-test",
        "2.92 pp",
        "crazing_119",
        "**",
    ]
    found_raw = [t for t in raw_terms if t.lower() in tex.lower()]
    no_exp_collision = not bool(
        re.search(
            r"\\texttt\{(?:E7|e2_|e5_|e7_)\}|(?:[Ss]plit|ANOVA|[Dd]ose|[Ss]caling|[Mm]odule)[^\n]{0,20}\b(?:E2|E5|E7)\b|\b(?:E2|E5|E7)\b[^\n]{0,20}(?:ANOVA|[Dd]ose|[Ss]caling|[Mm]odule)",
            tex,
        )
    )
    check(
        "terminology_and_collision_hygiene",
        len(found_raw) == 0 and no_exp_collision,
        f"found_raw={found_raw}, zero_exp_E_collision={no_exp_collision}",
        checks,
    )

    # 5. Live numbers match cloud_297_comprehensive_stats.json
    e2_sd_split = f"{stats['e2_variance_decomposition']['variance_decomposition']['test_map50']['sd_split']:.2f}"
    e2_span = f"{stats['e2_variance_decomposition']['variance_decomposition']['test_map50']['split_means_summary']['range']:.2f}"
    cf_dl = f"{stats['counterfactual_k10']['split_level_k10']['map50']['delta_leak']['mean']:.2f}"
    cf_micro = f"{stats['counterfactual_k10']['split_level_k10']['micro_ap50']['delta_leak']['mean']:.2f}"
    op15_miss = f"{stats['counterfactual_k10']['operating_point_escape_rates']['fppi_0.15']['leak_inst_miss']['delta_L_minus_C']['mean']:.2f}"
    op15_esc = f"{stats['counterfactual_k10']['operating_point_escape_rates']['fppi_0.15']['leak_img_escape']['delta_L_minus_C']['mean']:.2f}"
    tokens_needed = [e2_sd_split, e2_span, cf_dl, cf_micro, op15_miss.replace("-", ""), op15_esc.replace("-", "")]
    missing_vals = [t for t in tokens_needed if t not in tex]
    # Module-study values come from runs trained without the COCO weights and must stay withheld.
    e7_cbam = f"{stats['e7_modules']['paired_vs_base']['yolo11n-cbam']['all_10_test_map50']['mean']:.2f}"
    e7_withheld = f"+{e7_cbam} [" not in tex and "+2.80/+3.06" not in tex
    check(
        "stats_json_297_consistency",
        len(missing_vals) == 0 and e7_withheld,
        f"checked={tokens_needed}, missing={missing_vals}, e7_prefix_values_withheld={e7_withheld}",
        checks,
    )

    # 5b. Audit mean compliance recomputed from the per-evaluation codes (+ F04 post-hoc correction)
    import glob
    import statistics
    f04_fix = json.loads((RESULTS_DIR / "step4_f04_posthoc_correction_20261108.json").read_text(encoding="utf-8"))
    fix_pmcid = f04_fix["original_model_assisted_record"]["pmcid"]
    fix_ds = f04_fix["original_model_assisted_record"]["dataset"]
    scores = []
    for fp in sorted(glob.glob(str(ROOT_DIR / "docs" / "coding_v1" / "out_v2" / "*.json"))):
        rec = json.loads(Path(fp).read_text(encoding="utf-8"))
        if rec.get("eligibility") != "include":
            continue
        for ds in rec["datasets"]:
            fields = ds["fields"]
            n = sum(fields[f"F{i:02d}"]["status"] == "reported" for i in range(1, 18))
            if rec["pmcid"] == fix_pmcid and ds["dataset"] == fix_ds and fields["F04"]["status"] != "reported":
                n += 1
            scores.append(n)
    comp_token = f"{statistics.mean(scores):.2f} \\pm {statistics.stdev(scores):.2f}"
    check(
        "audit_mean_compliance",
        len(scores) == 182 and comp_token in tex,
        f"n_evals={len(scores)}, expected='{comp_token}' in text={comp_token in tex}",
        checks,
    )

    # 5b'. Hand-typed counts in fig04 source agree with the audit rollup (+ F04 correction)
    rollup = json.loads((ROOT_DIR / "docs" / "coding_v1" / "rollup_20261103.json").read_text(encoding="utf-8"))["evaluation_level_stats"]
    fig04_src = (MANUSCRIPT_DIR / "figures_src" / "fig04_reporting_completeness.py").read_text(encoding="utf-8")
    fig04_bad = []
    for fid, st_ in rollup.items():
        if not isinstance(st_, dict) or "reported_count" not in st_:
            continue
        rep, amb, nr = st_["reported_count"], st_["ambiguous_count"], st_["not_reported_count"]
        if fid == "F04":
            rep, nr = rep + 1, nr - 1
        if not re.search(rf'\("{fid}",\s*"[^"]*",\s*{rep},\s*{amb},\s*{nr}\)', fig04_src):
            fig04_bad.append(fid)
    check("fig04_counts_match_rollup", not fig04_bad, f"mismatched_fields={fig04_bad}", checks)

    # 5c. Training protocol text matches args.yaml of the cloud runs (RT-DETR-L uses optimizer=auto + AMP)
    runs_dir = ROOT_DIR / "benchmark" / "cloud_results_20261113" / "runs"
    rt_args = list(runs_dir.glob("baseline_*/rtdetr-l_*/args.yaml"))
    yolo_args = [p for p in runs_dir.glob("baseline_*/*/args.yaml") if "rtdetr" not in p.parent.name]
    rt_non_sgd = all(re.search(r"^optimizer: (auto|AdamW)$", p.read_text(encoding="utf-8"), re.M) for p in rt_args)
    yolo_sgd = all(re.search(r"^optimizer: SGD$", p.read_text(encoding="utf-8"), re.M) for p in yolo_args)
    protocol_ok = (not rt_args or (rt_non_sgd and "AdamW" in tex and "optimizer=auto" in tex)) and yolo_sgd
    check(
        "training_protocol_matches_args",
        protocol_ok,
        f"rtdetr_runs={len(rt_args)} auto/AdamW={rt_non_sgd}, yolo_runs={len(yolo_args)} SGD={yolo_sgd}, disclosed={'AdamW' in tex}",
        checks,
    )

    # 5d. Dose trend tests and missed-pair counts are produced by committed scripts
    dose = json.loads((RESULTS_DIR / "step3_dose_trend_tests.json").read_text(encoding="utf-8"))["metrics"]
    missed = json.loads((RESULTS_DIR / "step2_missed_overlap_pairs_64.json").read_text(encoding="utf-8"))
    g0 = missed["grouped_splits"]["G0"]
    clean_touch = [missed["counterfactual_splits"][f"K{k}"]["L"]["touch_test_clean"] for k in range(10)]
    trace_tokens = [
        f"L={dose['leak_micro']['page']['L']:.1f}",
        f"+{dose['leak_micro']['slope_mean']:.2f} \\pm {dose['leak_micro']['slope_sd']:.2f}",
        f"{missed['n_missed_cross_component']} of those",
        f"{g0['cross']} cases on $G_0$ ({g0['train_test']} between training and test)",
        f"{min(clean_touch)}--{max(clean_touch)} per split touch",
    ]
    missing_trace = [t for t in trace_tokens if t not in tex]
    check("dose_and_missed_pair_traceability", not missing_trace, f"checked={trace_tokens}, missing={missing_trace}", checks)

    # 5e. Stale phrases found in the 2026-11-13 audit must not return
    stale = ["4.69 \\pm 1.98", "Five detectors, one training recipe", "under an hour", "weight-transfer mismatch", "module replications", "causal leakage/module"]
    found_stale = [s for s in stale if s in tex]
    check("no_stale_phrases", not found_stale, f"found={found_stale}", checks)

    # 6. Figure 05 threshold sync (0.95 / 0.90) + Fig 06b registration figure included
    f5_src = FIG05_SRC.read_text(encoding="utf-8")
    fig_sync = (
        "0.95" in f5_src
        and "0.90" in f5_src
        and "0.85" not in f5_src
        and "fig06b_registration_examples.pdf" in tex
        and (MANUSCRIPT_DIR / "figures" / "fig06b_registration_examples.pdf").exists()
    )
    check("figure_pipeline_and_examples_sync", fig_sync, "fig05 uses 0.95/0.90; fig06b PDF present & cited", checks)

    # 7. Verified citations in refs.bib and clean BibTeX/LaTeX build
    bib_ok = all(
        k in bib
        for k in (
            "ramos2025leakage",
            "bourezgui2026rock",
            "wei2026amfnet",
            "li2026dpsnet",
            "cheng2026survey",
            "liu2025sfc",
            "lema2025reproducible",
        )
    )
    blg_txt = BLG_PATH.read_text(encoding="utf-8") if BLG_PATH.exists() else ""
    blg_ok = ("warning$ -- 0" in blg_txt or "\nwarning$ -- 0" in blg_txt or "warnings)" not in blg_txt) and ("error message" not in blg_txt)
    log_txt = LOG_PATH.read_text(encoding="utf-8", errors="ignore") if LOG_PATH.exists() else ""
    overfull_pts = [float(m.group(1)) for m in re.finditer(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt)]
    log_clean = (
        "LaTeX Error:" not in log_txt
        and "Undefined control sequence" not in log_txt
        and "There were undefined references" not in log_txt
        and "Citation `" not in log_txt
        and all(pt <= 1.5 for pt in overfull_pts)
    )
    check(
        "bibtex_and_latex_clean_build",
        bib_ok and blg_ok and log_clean,
        f"bib_entries_ok={bib_ok}, blg_zero_err_warn={blg_ok}, log_clean={log_clean}, max_overfull_pt={max(overfull_pts) if overfull_pts else 0.0:.2f}",
        checks,
    )

    all_pass = all(v["pass"] for v in checks.values())
    print("\n--- Manuscript Verification Checklist ---")
    for k, v in checks.items():
        tag = "PASS [OK]" if v["pass"] else "FAIL [X]"
        print(f"[{tag}] {k}: {v['detail']}")

    out_path = RESULTS_DIR / "step5_verification_report.json"
    out_path.write_text(json.dumps({"overall_pass": all_pass, "checks": checks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {out_path}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
