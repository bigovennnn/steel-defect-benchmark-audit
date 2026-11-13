"""Count how the human-confirmed overlap pairs that the registration graph misses
cross the released grouped and counterfactual splits.

Inputs (all in the repository):
  results/human_audit/internal_pool_mapping.json  349 audited pairs with source pool
  results/human_audit/pair_labels_coder{1,2}.csv  blind labels of the two authors
  benchmark/split_designs/neu_grouped_v6_split.json, neu_grouped_g0{1..9}_split.json  G0-G9
  benchmark/datasets/neu_k{00..09}_{L,C}/split.json  counterfactual splits K0-K9
  results/cloud_297_comprehensive_stats.json  split means for the correlation check

Output: results/step2_missed_overlap_pairs_64.json
Numbers are quoted in Section 4.1 ("What the graph misses") of the manuscript.
"""
import csv
import json
from collections import Counter
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GRAPH_POOL = "Pool_A_Registration_High_NCC_ge_085"


def stem(name: str) -> str:
    return Path(name).stem


def read_labels(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["audit_id"]: int(r["is_visually_overlapping"]) for r in csv.DictReader(f)}


def components(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    return find


def crossing(pairs, assign):
    """Count pairs whose two images fall in different partitions, and train-test pairs."""
    n_cross, n_tt = 0, 0
    for a, b in pairs:
        pa, pb = assign.get(a), assign.get(b)
        if pa is None or pb is None or pa == pb:
            continue
        n_cross += 1
        if {pa, pb} == {"train", "test"}:
            n_tt += 1
    return n_cross, n_tt


def main():
    pairs = json.loads((ROOT / "results/human_audit/internal_pool_mapping.json").read_text(encoding="utf-8"))
    c1 = read_labels(ROOT / "results/human_audit/pair_labels_coder1.csv")
    c2 = read_labels(ROOT / "results/human_audit/pair_labels_coder2.csv")

    graph = [(stem(p["image_a"]), stem(p["image_b"])) for p in pairs if p["source_pool"] == GRAPH_POOL]
    find = components(graph)

    non_graph_both = [p for p in pairs if p["source_pool"] != GRAPH_POOL and c1[p["audit_id"]] == 1 and c2[p["audit_id"]] == 1]
    missed = [p for p in non_graph_both if find(stem(p["image_a"])) != find(stem(p["image_b"]))]
    missed_pairs = [(stem(p["image_a"]), stem(p["image_b"])) for p in missed]

    out = {
        "n_graph_edges": len(graph),
        "n_non_graph_candidates": sum(p["source_pool"] != GRAPH_POOL for p in pairs),
        "n_non_graph_both_positive": len(non_graph_both),
        "n_missed_cross_component": len(missed),
        "missed_by_pool": dict(Counter(p["source_pool"] for p in missed)),
        "missed_pairs": missed_pairs,
        "grouped_splits": {},
        "counterfactual_splits": {},
    }

    split_files = {"G0": ROOT / "benchmark/split_designs/neu_grouped_v6_split.json"}
    split_files.update({f"G{i}": ROOT / f"benchmark/split_designs/neu_grouped_g0{i}_split.json" for i in range(1, 10)})
    for name, path in split_files.items():
        sp = json.loads(path.read_text(encoding="utf-8"))
        assign = {x: part for part in ("train", "val", "test") for x in sp[part]}
        n_cross, n_tt = crossing(missed_pairs, assign)
        test_touched = {x for a, b in missed_pairs for x in (a, b)
                        if assign.get(x) == "test" and {assign.get(a), assign.get(b)} == {"train", "test"}}
        out["grouped_splits"][name] = {"cross": n_cross, "train_test": n_tt, "test_images_touched": len(test_touched)}

    for k in range(10):
        row = {}
        for cond in ("L", "C"):
            sp = json.loads((ROOT / f"benchmark/datasets/neu_k{k:02d}_{cond}/split.json").read_text(encoding="utf-8"))
            assign = {x: part for part in ("train", "val", "test") for x in sp[part]}
            n_cross, n_tt = crossing(missed_pairs, assign)
            leak, clean = set(sp["test_leaked"]), set(sp["test_clean"])
            row[cond] = {
                "cross": n_cross,
                "train_test": n_tt,
                "touch_test_leaked": sum((a in leak) or (b in leak) for a, b in missed_pairs),
                "touch_test_clean": sum((a in clean) or (b in clean) for a, b in missed_pairs),
            }
        out["counterfactual_splits"][f"K{k}"] = row

    # Does the number of missed train-test pairs track the split mean of the Split-ANOVA runs?
    e2 = json.loads((ROOT / "results/cloud_297_comprehensive_stats.json").read_text(encoding="utf-8"))["e2_variance_decomposition"]["per_split"]
    tt = [out["grouped_splits"][f"G{i}"]["train_test"] for i in range(10)]
    means = [e2[f"e2_g{i:02d}"]["test_map50"]["mean"] for i in range(10)]
    r, p = stats.pearsonr(tt, means)
    rho, p_rho = stats.spearmanr(tt, means)
    out["g_split_correlation"] = {"pearson_r": round(float(r), 3), "pearson_p": round(float(p), 3),
                                  "spearman_rho": round(float(rho), 3), "spearman_p": round(float(p_rho), 3)}

    dst = ROOT / "results/step2_missed_overlap_pairs_64.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    g = out["grouped_splits"]
    kk = out["counterfactual_splits"]
    print("missed pairs:", out["n_missed_cross_component"], out["missed_by_pool"])
    print("G0:", g["G0"])
    print("G1-G9 cross:", [g[f"G{i}"]["cross"] for i in range(1, 10)], "train-test:", [g[f"G{i}"]["train_test"] for i in range(1, 10)])
    for cond in ("L", "C"):
        print(cond, "cross:", [kk[f"K{k}"][cond]["cross"] for k in range(10)],
              "tt:", [kk[f"K{k}"][cond]["train_test"] for k in range(10)],
              "leak:", [kk[f"K{k}"][cond]["touch_test_leaked"] for k in range(10)],
              "clean:", [kk[f"K{k}"][cond]["touch_test_clean"] for k in range(10)])
    print("corr:", out["g_split_correlation"])
    print("written", dst)


if __name__ == "__main__":
    main()
