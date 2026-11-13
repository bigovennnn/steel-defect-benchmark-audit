#!/usr/bin/env python3
"""
Export complete counterfactual split manifests and explicit leaked-partner lists
for both K=10 multi-split counterfactuals (plus E5 dose-response 25/50/75%)
and the M=3 single-split counterfactual ablation.
"""

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT_DIR / "benchmark"
DATASETS_DIR = BENCH_DIR / "datasets"
SPLIT_DESIGNS_DIR = BENCH_DIR / "split_designs"
RESULTS_DIR = ROOT_DIR / "results"

V6_SPLIT_PATH = DATASETS_DIR / "neu_grouped_v6" / "split.json"
EDGES_CSV_PATH = BENCH_DIR / "audits" / "neu_candidate_edges_223_20261101.csv"
CAND_JSON_PATH = BENCH_DIR / "near_duplicate_20261101" / "candidate_pairs_hs_v1.json"
STEP2_MANIFEST_PATH = RESULTS_DIR / "step2_v6_splits_manifest.json"
STEP3_MANIFEST_PATH = RESULTS_DIR / "step3_counterfactual_splits_manifest.json"
NAIVE_SPLIT_PATH = SPLIT_DESIGNS_DIR / "neu_naive_random_split_20261103.json"

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def stem_class(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def split_sha256(split_obj: dict) -> str:
    canonical = json.dumps(
        {k: sorted(split_obj[k]) for k in ["train", "val", "test"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_connected_components(stems: list[str], edges: list[tuple[str, str]]) -> tuple[list[list[str]], dict[str, int]]:
    adj = defaultdict(set)
    stem_set = set(stems)
    for u, v in edges:
        if u in stem_set and v in stem_set:
            adj[u].add(v)
            adj[v].add(u)
    visited = set()
    comps = []
    stem_to_comp = {}
    for s in sorted(stems):
        if s not in visited:
            comp = []
            q = [s]
            visited.add(s)
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for nb in sorted(adj[curr]):
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comp_sorted = sorted(comp)
            cid = len(comps)
            comps.append(comp_sorted)
            for item in comp_sorted:
                stem_to_comp[item] = cid
    return comps, stem_to_comp


def main():
    v6_sp = json.loads(V6_SPLIT_PATH.read_text(encoding="utf-8"))
    all_stems = sorted(set(v6_sp["train"] + v6_sp["val"] + v6_sp["test"]))
    assert len(all_stems) == 1797, f"Expected 1797 stems, got {len(all_stems)}"

    df_edges = pd.read_csv(EDGES_CSV_PATH)
    assert len(df_edges) == 223, f"Expected 223 edges, got {len(df_edges)}"

    # 1. Materialize candidate_pairs_hs_v1.json locally
    CAND_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    cand_pairs = [
        {
            "img_a": str(r["a_file"]),
            "img_b": str(r["b_file"]),
            "pixel_ncc": float(r["pixel_ncc"]),
            "gradient_ncc": float(r["gradient_ncc"]),
        }
        for _, r in df_edges.iterrows()
    ]
    CAND_JSON_PATH.write_text(
        json.dumps({"total_edges": len(cand_pairs), "candidates": cand_pairs}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Wrote {CAND_JSON_PATH} ({len(cand_pairs)} edges)")

    # Build edge metadata lookup (undirected, preserving per-edge attributes)
    edges_list = []
    adj = defaultdict(set)
    edge_meta_by_pair = {}
    for _, r in df_edges.iterrows():
        u = Path(str(r["a_file"])).stem
        v = Path(str(r["b_file"])).stem
        edges_list.append((u, v))
        adj[u].add(v)
        adj[v].add(u)
        key = tuple(sorted((u, v)))
        edge_meta_by_pair[key] = {
            "candidate_id": str(r["candidate_id"]),
            "a_stem": u,
            "b_stem": v,
            "rotation_deg": int(r["rotation_b_ccw_degrees"]),
            "dx": int(r["shift_b_dx_pixels"]),
            "dy": int(r["shift_b_dy_pixels"]),
            "overlap_fraction": float(r["overlap_fraction"]),
            "pixel_ncc": float(r["pixel_ncc"]),
            "gradient_ncc": float(r["gradient_ncc"]),
        }

    _, stem_to_comp = build_connected_components(all_stems, edges_list)

    by_class = {c: sorted([s for s in all_stems if stem_class(s) == c]) for c in NEU_CLASSES}

    # 2. Enrich K=10 counterfactual splits in step2_v6_splits_manifest.json
    manifest = json.loads(STEP2_MANIFEST_PATH.read_text(encoding="utf-8"))
    step3_manifest = json.loads(STEP3_MANIFEST_PATH.read_text(encoding="utf-8"))
    naive_split = json.loads(NAIVE_SPLIT_PATH.read_text(encoding="utf-8"))

    k10_csv_rows = []
    enriched_splits = []

    for sp_info in manifest["multi_split_counterfactual_k10"]["splits"]:
        k = int(sp_info["split_index_k"])
        seed_k = int(sp_info["split_seed"])
        rng = np.random.default_rng(seed_k)
        train_s, val_s, test_s = [], [], []
        for c in NEU_CLASSES:
            c_stems = list(by_class[c])
            perm = rng.permutation(len(c_stems))
            c_stems = [c_stems[i] for i in perm]
            test_s.extend(c_stems[:30])
            val_s.extend(c_stems[30:60])
            train_s.extend(c_stems[60:])
        train_s, val_s, test_s = sorted(train_s), sorted(val_s), sorted(test_s)
        train_set, val_set, test_set = set(train_s), set(val_s), set(test_s)

        computed_sha = split_sha256({"train": train_s, "val": val_s, "test": test_s})
        assert computed_sha == sp_info["naive_sha256"], f"SHA mismatch at k={k}"

        test_leaked = sorted([u for u in test_s if any(v in train_set for v in adj[u])])
        test_clean = sorted([u for u in test_s if u not in set(test_leaked)])
        p_leak = sorted([v for v in train_s if any(u in set(test_leaked) for u in adj[v])])
        val_clean = sorted([v for v in val_s if not any((nb in train_set or nb in test_set) for nb in adj[v])])
        val_contaminated = sorted([v for v in val_s if v not in set(val_clean)])

        assert test_leaked == sorted(sp_info["test_leaked_stems"]), f"test_leaked mismatch at k={k}"
        assert p_leak == sorted(sp_info["p_leak_stems"]), f"p_leak mismatch at k={k}"
        assert val_clean == sorted(sp_info["val_clean_stems"]), f"val_clean mismatch at k={k}"

        r_holdout = sorted(sp_info["r_holdout_stems"])
        p_leak_set = set(p_leak)
        r_holdout_set = set(r_holdout)
        assert p_leak_set.isdisjoint(r_holdout_set), f"Overlap p_leak & r_holdout at k={k}"
        assert Counter(stem_class(u) for u in p_leak) == Counter(stem_class(u) for u in r_holdout)

        d_core = sorted([u for u in train_s if u not in p_leak_set and u not in r_holdout_set])
        train_L = sorted(d_core + p_leak)
        train_C = sorted(d_core + r_holdout)
        assert len(train_L) == len(train_C) == sp_info["n_train_matched"]
        assert len(test_clean) == sp_info["n_test_clean"]

        # Verify against materialized disk directories neu_k{k:02d}_{L,C} if present
        for cond, expected_train in [("L", train_L), ("C", train_C)]:
            ds_dir = DATASETS_DIR / f"neu_k{k:02d}_{cond}"
            if ds_dir.exists():
                disk_train = sorted(p.stem for p in (ds_dir / "images" / "train").glob("*.jpg"))
                disk_val = sorted(p.stem for p in (ds_dir / "images" / "val").glob("*.jpg"))
                disk_test = sorted(p.stem for p in (ds_dir / "images" / "test").glob("*.jpg"))
                disk_leaked = sorted(p.stem for p in (ds_dir / "images" / "test_leaked").glob("*.jpg"))
                disk_clean = sorted(p.stem for p in (ds_dir / "images" / "test_clean").glob("*.jpg"))
                assert disk_train == expected_train, f"Disk train mismatch at k={k} {cond}"
                assert disk_val == val_clean, f"Disk val mismatch at k={k} {cond}"
                assert disk_test == test_s, f"Disk test mismatch at k={k} {cond}"
                assert disk_leaked == test_leaked, f"Disk test_leaked mismatch at k={k} {cond}"
                assert disk_clean == test_clean, f"Disk test_clean mismatch at k={k} {cond}"

                # Write split.json into dataset directory
                ds_split_payload = {
                    "split_name": f"neu_k{k:02d}_{cond}",
                    "split_index_k": k,
                    "split_seed": seed_k,
                    "condition": cond,
                    "naive_sha256": computed_sha,
                    "train": expected_train,
                    "val": val_clean,
                    "test": test_s,
                    "test_leaked": test_leaked,
                    "test_clean": test_clean,
                    "p_leak_stems": p_leak,
                    "r_holdout_stems": r_holdout,
                    "d_core_stems": d_core,
                }
                (ds_dir / "split.json").write_text(json.dumps(ds_split_payload, indent=2), encoding="utf-8")

        # Build explicit leaked-partner graph
        test_to_train_partners = {}
        train_partner_to_test_stems = defaultdict(list)
        leaked_partner_edges = []

        for u in test_leaked:
            partners = sorted([v for v in adj[u] if v in p_leak_set])
            assert len(partners) >= 1, f"Leaked test {u} has no train partner at k={k}"
            test_to_train_partners[u] = partners
            for v in partners:
                train_partner_to_test_stems[v].append(u)
                meta = edge_meta_by_pair[tuple(sorted((u, v)))]
                edge_rec = {
                    "split_k": k,
                    "candidate_id": meta["candidate_id"],
                    "test_stem": u,
                    "train_partner_stem": v,
                    "class": stem_class(u),
                    "cluster_id": int(stem_to_comp[u]),
                    "rotation_deg": meta["rotation_deg"],
                    "dx": meta["dx"],
                    "dy": meta["dy"],
                    "overlap_fraction": meta["overlap_fraction"],
                    "pixel_ncc": meta["pixel_ncc"],
                    "gradient_ncc": meta["gradient_ncc"],
                }
                leaked_partner_edges.append(edge_rec)
                k10_csv_rows.append(edge_rec)

        train_partner_to_test_stems_sorted = {
            v: sorted(us) for v, us in sorted(train_partner_to_test_stems.items())
        }
        assert set(train_partner_to_test_stems_sorted.keys()) == p_leak_set

        # E5 dose-response manifests for k in {0, 1, 2}
        e5_dose_splits = {}
        if k in {0, 1, 2}:
            p_by_cls = defaultdict(list)
            r_by_cls = defaultdict(list)
            for u in p_leak:
                p_by_cls[stem_class(u)].append(u)
            for u in r_holdout:
                r_by_cls[stem_class(u)].append(u)

            for dose_pct in [25, 50, 75]:
                frac = dose_pct / 100.0
                kept_p = []
                kept_r = []
                for c in NEU_CLASSES:
                    p_list = p_by_cls[c]
                    r_list = r_by_cls[c]
                    n_total = len(p_list)
                    n_keep_leak = int(round(n_total * frac))
                    n_keep_clean = n_total - n_keep_leak
                    kept_p.extend(p_list[:n_keep_leak])
                    kept_r.extend(r_list[:n_keep_clean])
                kept_p = sorted(kept_p)
                kept_r = sorted(kept_r)
                train_dose = sorted(d_core + kept_p + kept_r)
                assert len(train_dose) == len(train_L)
                assert Counter(stem_class(u) for u in train_dose) == Counter(stem_class(u) for u in train_L)
                e5_dose_splits[f"dose{dose_pct}"] = {
                    "dose_pct": dose_pct,
                    "n_kept_p_leak": len(kept_p),
                    "n_kept_r_holdout": len(kept_r),
                    "kept_p_leak_stems": kept_p,
                    "kept_r_holdout_stems": kept_r,
                    "train_dose_stems": train_dose,
                }

        enriched = dict(sp_info)
        enriched.update(
            {
                "naive_train_stems": train_s,
                "naive_val_stems": val_s,
                "val_contaminated_stems": val_contaminated,
                "test_all_stems": test_s,
                "test_clean_stems": test_clean,
                "d_core_stems": d_core,
                "train_L_stems": train_L,
                "train_C_stems": train_C,
                "n_leaked_partner_edges": len(leaked_partner_edges),
                "leaked_partner_edges": leaked_partner_edges,
                "test_to_train_partners": test_to_train_partners,
                "train_partner_to_test_stems": train_partner_to_test_stems_sorted,
            }
        )
        if e5_dose_splits:
            enriched["e5_dose_response"] = e5_dose_splits
        enriched_splits.append(enriched)

    manifest["multi_split_counterfactual_k10"]["splits"] = enriched_splits
    STEP2_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Enriched {STEP2_MANIFEST_PATH}")

    # 3. Write standalone split_designs/neu_counterfactual_k10_splits.json
    k10_standalone = {
        "description": "Complete K=10 counterfactual split manifests, E5 dose-response splits (k=0,1,2), and explicit train-test leaked partner mappings for NEU-DET.",
        "generated_at": manifest["generated_at"],
        "k_count": manifest["multi_split_counterfactual_k10"]["k_count"],
        "base_seed": manifest["multi_split_counterfactual_k10"]["base_seed"],
        "total_leaked_test_image_evaluations": manifest["multi_split_counterfactual_k10"]["total_leaked_test_image_evaluations"],
        "unique_leaked_test_images_across_k10": manifest["multi_split_counterfactual_k10"]["unique_leaked_test_images_across_k10"],
        "splits": enriched_splits,
    }
    k10_standalone_path = SPLIT_DESIGNS_DIR / "neu_counterfactual_k10_splits.json"
    k10_standalone_path.write_text(json.dumps(k10_standalone, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {k10_standalone_path}")

    # 4. Write results/counterfactual_leaked_partners_k10.csv
    k10_csv_path = RESULTS_DIR / "counterfactual_leaked_partners_k10.csv"
    fieldnames = [
        "split_k",
        "candidate_id",
        "test_stem",
        "train_partner_stem",
        "class",
        "cluster_id",
        "rotation_deg",
        "dx",
        "dy",
        "overlap_fraction",
        "pixel_ncc",
        "gradient_ncc",
    ]
    with k10_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(k10_csv_rows)
    print(f"[OK] Wrote {k10_csv_path} ({len(k10_csv_rows)} edge rows across K=10 splits)")

    # 5. Write results/counterfactual_leaked_partners_m3_single_split.csv
    naive_train_set = set(naive_split["train"])
    naive_leaked_test = sorted(naive_split["candidate_edges_cross_split_audit"]["leaked_test_images"])
    assert len(naive_leaked_test) == 25
    expected_p_leak = sorted(step3_manifest["draws"][0]["p_leak_stems"])
    assert len(expected_p_leak) == 31

    m3_rows = []
    found_partners = set()
    for u in naive_leaked_test:
        partners = sorted([v for v in adj[u] if v in naive_train_set])
        for v in partners:
            found_partners.add(v)
            meta = edge_meta_by_pair[tuple(sorted((u, v)))]
            m3_rows.append(
                {
                    "split_name": "neu_naive_random_m3",
                    "candidate_id": meta["candidate_id"],
                    "test_stem": u,
                    "train_partner_stem": v,
                    "class": stem_class(u),
                    "cluster_id": int(stem_to_comp[u]),
                    "rotation_deg": meta["rotation_deg"],
                    "dx": meta["dx"],
                    "dy": meta["dy"],
                    "overlap_fraction": meta["overlap_fraction"],
                    "pixel_ncc": meta["pixel_ncc"],
                    "gradient_ncc": meta["gradient_ncc"],
                }
            )
    assert sorted(found_partners) == expected_p_leak, "M=3 partner set mismatch!"
    assert len(m3_rows) == 32, f"Expected 32 train-test edges for M=3 single split, got {len(m3_rows)}"

    m3_csv_path = RESULTS_DIR / "counterfactual_leaked_partners_m3_single_split.csv"
    m3_fieldnames = ["split_name"] + fieldnames[1:]
    with m3_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=m3_fieldnames)
        writer.writeheader()
        writer.writerows(m3_rows)
    print(f"[OK] Wrote {m3_csv_path} ({len(m3_rows)} edges connecting {len(naive_leaked_test)} test & {len(found_partners)} train partners)")


if __name__ == "__main__":
    main()
