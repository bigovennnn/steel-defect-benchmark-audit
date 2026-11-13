"""
Fig. 9 (Headline Synthesis Figure):
Published Claimed mAP@0.5 Gains (n=174 verbatim-verified evaluations across 131 papers)
placed directly against empirical protocol variance, E2 split realization, K=10 leakage, and E7 module replications.
"""
from pathlib import Path
import json
import numpy as np
from style import *

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DELTAS_JSON = ROOT_DIR / "results" / "step4_claimed_deltas_182.json"
STATS = json.loads((ROOT_DIR / "results" / "cloud_297_comprehensive_stats.json").read_text(encoding="utf-8"))

d = json.loads(DELTAS_JSON.read_text(encoding="utf-8"))
neu_vals = np.array([
    r["claimed_delta_map50_pp"]
    for r in d["records"]
    if r["dataset"] == "NEU-DET" and r["claimed_delta_map50_pp"] is not None and 0 < r["claimed_delta_map50_pp"] <= 16.0
])
gc10_vals = np.array([
    r["claimed_delta_map50_pp"]
    for r in d["records"]
    if r["dataset"] == "GC10-DET" and r["claimed_delta_map50_pp"] is not None and 0 < r["claimed_delta_map50_pp"] <= 16.0
])
all_vals = np.array([
    r["claimed_delta_map50_pp"]
    for r in d["records"]
    if r["claimed_delta_map50_pp"] is not None and 0 < r["claimed_delta_map50_pp"] <= 25.0
])

fig = plt.figure(figsize=(W2 * MM, 82 * MM))
gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.16, top=0.87, wspace=0.56, width_ratios=[1.12, 1.08])

# Panel (a): Distribution of claimed improvements vs. protocol noise zone
ax = fig.add_subplot(gs[0])
panel(ax, "(a)", x=-0.12, y=1.05)

bins = np.arange(0, 16.5, 1.0)
ax.hist(
    [neu_vals, gc10_vals],
    bins=bins,
    stacked=True,
    color=[C["blue"], C["teal"]],
    edgecolor="white",
    linewidth=0.7,
    label=["NEU-DET (n = 121)", "GC10-DET (n = 53)"],
    zorder=3,
)

# Shade 10-seed NEU-DET mean range (3.20 pp; arch range 2.58-3.87 pp)
ax.axvspan(0, 3.87, color=C["amber"], alpha=0.16, zorder=1)
ax.axvline(3.20, color=C["amber"], lw=1.3, ls="--", zorder=4)
ax.text(
    4.10,
    36.2,
    "10-seed NEU range: 2.58–3.87 pp\n(mean 3.20 pp; 38.5% claims ≤ 3.0 pp)",
    fontsize=6.7,
    color=C["amber"],
    fontweight="bold",
    ha="left",
    va="top",
)

med_all = float(np.median(all_vals))
ax.axvline(med_all, color=C["navy"], lw=1.3, ls="-", zorder=4)
ax.text(
    med_all + 0.30,
    29.0,
    f"Literature median = +{med_all:.2f} pp",
    fontsize=6.9,
    color=C["navy"],
    fontweight="bold",
    ha="left",
    va="top",
)

ax.set_xlim(0, 16)
ax.set_ylim(0, 38)
ax.set_xlabel("Claimed improvement over baseline ΔmAP@0.5 (pp)")
ax.set_ylabel("Number of evaluations")
ax.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.63), frameon=True, facecolor="white", edgecolor="none", fontsize=7.0)

# Panel (b): Scale comparison across all 297 cloud runs
ax2 = fig.add_subplot(gs[1])
panel(ax2, "(b)", x=-0.52, y=1.05)

cf50 = STATS["counterfactual_k10"]["split_level_k10"]["map50"]["delta_leak"]
cf_all50 = STATS["counterfactual_k10"]["split_level_k10"]["map50"]["delta_all"]
e7_cbam = STATS["e7_modules"]["paired_vs_base"]["yolo11n-cbam"]["all_10_test_map50"]
e7_bifpn = STATS["e7_modules"]["paired_vs_base"]["yolo11n-bifpn"]["all_10_test_map50"]
e7_ema = STATS["e7_modules"]["paired_vs_base"]["yolo11n-ema"]["all_10_test_map50"]

items = [
    ("Published ΔmAP@0.5\n(median, IQR; n=174)", 3.80, 2.12, 6.85, C["blue"], "3.80 [2.12, 6.85]"),
    ("Grouped-split span (10 splits)\n(SD=2.61, 82.8% var.)", 2.61, 1.19, 8.87, C["amber"], "8.87 span (SD 2.61)"),
    ("GC10 10-seed range\n(5 dets., 50 runs)", 5.00, 4.02, 6.57, C["teal"], "5.00 [4.02, 6.57]"),
    ("NEU 10-seed range\n(5 dets., 50 runs)", 3.20, 2.58, 3.87, C["teal"], "3.20 [2.58, 3.87]"),
    ("K=10 Leakage on T_leak\n(100 runs, df=9 CI)", cf50["mean"], cf50["ci95_lo"], cf50["ci95_hi"], C["red"], f"+{cf50['mean']:.2f} [{cf50['ci95_lo']:.2f}, {cf50['ci95_hi']:.2f}]"),
    ("CBAM seed cherry-pick\n(within-plat. max-min)", 2.93, 2.80, 3.06, C["red"], "+2.80 / +3.06"),
    ("Paired BiFPN (10 seeds)\n(448 tensors, df=9 CI)", e7_bifpn["mean"], e7_bifpn["ci95_lo"], e7_bifpn["ci95_hi"], C["navy"], f"+{e7_bifpn['mean']:.2f} [{e7_bifpn['ci95_lo']:+.2f}, {e7_bifpn['ci95_hi']:+.2f}]"),
    ("Paired CBAM (10 seeds)\n(448 tensors, df=9 CI)", e7_cbam["mean"], e7_cbam["ci95_lo"], e7_cbam["ci95_hi"], C["navy"], f"+{e7_cbam['mean']:.2f} [{e7_cbam['ci95_lo']:+.2f}, {e7_cbam['ci95_hi']:+.2f}]"),
    ("Paired EMA (10 seeds)\n(448 tensors, df=9 CI)", e7_ema["mean"], e7_ema["ci95_lo"], e7_ema["ci95_hi"], C["navy"], f"+{e7_ema['mean']:.2f} [{e7_ema['ci95_lo']:+.2f}, {e7_ema['ci95_hi']:+.2f}]"),
]

ax2.axvline(0, color=C["line"], lw=0.8, ls=":", zorder=1)
for idx, (label_s, pt, lo, hi, col, val_txt) in enumerate(items):
    ax2.plot([lo, hi], [idx, idx], color=col, lw=2.2, solid_capstyle="round", zorder=3)
    ax2.plot(pt, idx, "o", ms=5.0, mfc=col, mec="white", mew=0.75, zorder=4)
    ax2.text(hi + 0.28, idx, val_txt, va="center", ha="left", fontsize=6.6, fontweight="bold", color=C["ink"])

ax2.set_yticks(range(len(items)))
ax2.set_yticklabels([it[0] for it in items], color=C["ink"], fontsize=6.8)
ax2.invert_yaxis()
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.set_xlim(-1.4, 15.6)
ax2.set_xlabel("Magnitude / effect size on mAP@0.5 (pp)")
ax2.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax2.set_axisbelow(True)

save(fig, "fig09_claims_vs_noise")
print("Saved fig09_claims_vs_noise.pdf and .png")
