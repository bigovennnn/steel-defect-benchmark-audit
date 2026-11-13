"""Ordered-trend tests for the CF-Dose series (0/25/50/75/100% of partners retained).

Each of the nine (split, seed) series contributes one score per dose level, read from
results/cloud_297_comprehensive_stats.json (e5_dose_response.by_dose.*.values, same order).
Reports Page's L (normal approximation), the per-series least-squares slope over dose
fraction 0..1 with a one-sample t-test, and the 0 -> 100% endpoint difference.

Output: results/step3_dose_trend_tests.json (quoted in the "Dose" paragraph of Section 5.2).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["C", "D25", "D50", "D75", "L"]
DOSE = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def page_test(M: np.ndarray) -> dict:
    n, k = M.shape
    ranks = np.array([stats.rankdata(row) for row in M])
    L = float((ranks * np.arange(1, k + 1)).sum())
    mu = n * k * (k + 1) ** 2 / 4
    sd = np.sqrt(n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144)
    z = (L - mu) / sd
    return {"L": L, "z": round(float(z), 3), "p_one_sided": round(float(1 - stats.norm.cdf(z)), 4),
            "p_two_sided": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4)}


def main():
    by_dose = json.loads((ROOT / "results/cloud_297_comprehensive_stats.json").read_text(encoding="utf-8"))["e5_dose_response"]["by_dose"]
    out = {"n_series": None, "metrics": {}}
    for metric in ("leak_micro", "leak_map50", "leak_map50_95", "leak_inst_rec"):
        M = np.array([by_dose[lv][metric]["values"] for lv in LEVELS]).T
        out["n_series"] = int(M.shape[0])
        slopes = np.array([np.polyfit(DOSE, row, 1)[0] for row in M])
        endpoint = M[:, -1] - M[:, 0]
        t = stats.ttest_1samp(slopes, 0.0)
        out["metrics"][metric] = {
            "level_means": [round(float(v), 2) for v in M.mean(axis=0)],
            "page": page_test(M),
            "slope_mean": round(float(slopes.mean()), 2),
            "slope_sd": round(float(slopes.std(ddof=1)), 2),
            "slope_t": round(float(t.statistic), 2),
            "slope_df": int(len(slopes) - 1),
            "slope_p_two_sided": round(float(t.pvalue), 3),
            "endpoint_mean": round(float(endpoint.mean()), 2),
            "endpoint_sd": round(float(endpoint.std(ddof=1)), 2),
            "endpoint_p_wilcoxon": round(float(stats.wilcoxon(endpoint).pvalue), 4),
        }
    dst = ROOT / "results/step3_dose_trend_tests.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for m, v in out["metrics"].items():
        print(m, v)
    print("written", dst)


if __name__ == "__main__":
    main()
