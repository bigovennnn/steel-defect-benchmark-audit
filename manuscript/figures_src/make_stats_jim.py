"""Rebuilds data/stats.json from a clone of the companion repository.
Usage: python make_stats.py /path/to/steel-defect-benchmark-audit"""
import json, csv, collections, glob, sys, os
repo = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P = lambda *a: os.path.join(repo, *a)
def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
F = lambda x: float(x) if x not in ("", None) else None
r = rd(P("results", "cloud_297_runs_manifest.csv")); J = json.load(open(P("results", "cloud_297_comprehensive_stats.json")))
D = {"base": {}}
for suite, ds in (("baseline_neu", "NEU-DET"), ("baseline_gc10", "GC10-DET")):
    g = collections.defaultdict(list)
    for x in r:
        if x["suite"] == suite: g[x["arch"]].append((int(x["seed"]), 100 * F(x["val_map50"]), 100 * F(x["test_map50"]), 100 * F(x["test_map50_95"])))
    D["base"][ds] = {a: sorted(v) for a, v in g.items()}
pc = collections.defaultdict(lambda: collections.defaultdict(list))
_bneu_files = glob.glob(P("results", "cloud_297_runs", "baseline_neu", "*", "eval_summary.json")) or glob.glob(P("benchmark", "cloud_results_20261113", "runs", "baseline_neu", "*", "eval_summary.json"))
for p in _bneu_files:
    j = json.load(open(p))
    for c, v in j["test"]["per_class_ap50"].items(): pc[j["arch"]][c].append(100 * v)
D["per_class"] = {a: dict(d) for a, d in pc.items()}
e2 = J["e2_variance_decomposition"]
D["e2"] = {k: {kk: v[kk] for kk in ("seeds", "test_map50", "val_map50")} for k, v in e2["per_split"].items()}
D["e2_var"] = {k: e2["variance_decomposition"]["test_map50"][k] for k in ("var_split", "var_seed", "sd_split", "sd_seed", "pct_split", "pct_seed", "f_stat", "p_value")}
cf = J["counterfactual_k10"]; sl = cf["split_level_k10"]; keep = ("mean", "sd", "ci95_lo", "ci95_hi", "values", "p_wilcoxon", "p_ttest", "n_pos")
D["cf_per_split"] = cf["per_split"]
D["cf"] = {m: {k: {kk: sl[m][k].get(kk) for kk in keep} for k in sl[m]} for m in sl}
D["cf_class"] = {c: {k: {kk: v[k].get(kk) for kk in keep} for k in v} for c, v in cf["per_class_ap50"].items()}
D["escape"] = cf["operating_point_escape_rates"]
D["dose"] = {k: {kk: {a: vv.get(a) for a in ("mean", "sd", "values", "p_wilcoxon", "n")} for kk, vv in v.items() if isinstance(vv, dict)} for k, v in J["e5_dose_response"]["by_dose"].items()}
m = json.load(open(P("results", "human_audit", "internal_pool_mapping.json")))
c1 = {x["audit_id"]: int(x["is_visually_overlapping"]) for x in rd(P("results", "human_audit", "pair_labels_coder1.csv"))}
c2 = {x["audit_id"]: int(x["is_visually_overlapping"]) for x in rd(P("results", "human_audit", "pair_labels_coder2.csv"))}
for x in m: x["c1"], x["c2"] = c1[x["audit_id"]], c2[x["audit_id"]]
D["pairs"] = m
recs = json.load(open(P("results", "step4_claimed_deltas_182.json")))["records"]   # replace with the cleaned table when ready
D["claims"] = [{"dataset": x.get("dataset"), "delta": x.get("claimed_delta_map50_pp"), "src": x.get("verification_source")} for x in recs]
_out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_jim.json")
with open(_out_json, "w", encoding="utf-8") as _f:
    json.dump(D, _f)
print("stats.json written")
