"""
Fig. 8: K=10 Multi-Split Counterfactual Partner-Replacement (100 runs) and E5 Dose-Response (45 runs).
Reads verified statistics from results/cloud_297_comprehensive_stats.json.
"""
from pathlib import Path
import json
import numpy as np
from style import *

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATS = json.loads((ROOT_DIR / "results" / "cloud_297_comprehensive_stats.json").read_text(encoding="utf-8"))

cf = STATS["counterfactual_k10"]
e5 = STATS["e5_dose_response"]

fig = plt.figure(figsize=(W2 * MM, 76 * MM))
gs = fig.add_gridspec(1, 3, left=0.075, right=0.975, bottom=0.19, top=0.83,
                      wspace=0.56, width_ratios=[1.0, 1.32, 1.25])

# (a) K=10 split-level trajectories (C vs L on T_leak and T_clean, 100 runs)
ax = fig.add_subplot(gs[0])
panel(ax, "(a)", x=-0.22, y=1.07)
sp50 = cf["split_level_k10"]["map50"]
c_leak = sp50["C_leak"]["values"]
l_leak = sp50["L_leak"]["values"]
c_cln = sp50["C_clean"]["values"]
l_cln = sp50["L_clean"]["values"]

for cl, ll in zip(c_leak, l_leak):
    ax.plot([0, 1], [cl, ll], color=C["red"], lw=0.85, alpha=0.55, zorder=2)
ax.plot([0, 1], [np.mean(c_leak), np.mean(l_leak)], color=C["red"], lw=2.3, zorder=5)
ax.plot(0, np.mean(c_leak), "o", ms=5.5, mfc=C["navy"], mec="white", mew=0.8, zorder=6)
ax.plot(1, np.mean(l_leak), "o", ms=5.5, mfc=C["red"], mec="white", mew=0.8, zorder=6)

for cc, lc in zip(c_cln, l_cln):
    ax.plot([0, 1], [cc, lc], color=C["teal"], lw=0.7, alpha=0.35, ls="--", zorder=1)
ax.plot([0, 1], [np.mean(c_cln), np.mean(l_cln)], color=C["teal"], lw=2.0, ls="--", zorder=4)
ax.plot(0, np.mean(c_cln), "s", ms=4.8, mfc=C["teal"], mec="white", mew=0.8, zorder=6)
ax.plot(1, np.mean(l_cln), "s", ms=4.8, mfc=C["teal"], mec="white", mew=0.8, zorder=6)

dl = sp50["delta_leak"]
dc = sp50["delta_clean"]
ax.text(0.5, 0.97, f"T_leak: Δ = +{dl['mean']:.2f} pp\n(10/10 pos, p={dl['p_wilcoxon']:.4f})",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.5, fontweight="bold", color=C["red"])
ax.text(0.5, 0.04, f"T_clean control:\n{dc['mean']:+.2f} pp (p={dc['p_wilcoxon']:.2f})",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color=C["teal"], fontweight="bold")

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(54, 96)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Clean (C)\n0% leaked", "Leaked (L)\n100% partners"], color=C["ink"], fontsize=7.2)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Macro mAP@0.5 (%, K=10 splits)")
ax.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.set_title("K=10 paired split shift", fontsize=7.8, fontweight="bold", pad=6)

# (b) Per-class Test AP@0.5 on T_leak across K=10 splits (300 leaked image evaluations)
ax2 = fig.add_subplot(gs[1])
panel(ax2, "(b)", x=-0.42, y=1.07)
cls_order = [
    ("rolled-in_scale", "Rolled-in scale"),
    ("inclusion", "Inclusion"),
    ("patches", "Patches"),
    ("crazing", "Crazing"),
    ("scratches", "Scratches"),
    ("pitted_surface", "Pitted surface"),
]
pc = cf["per_class_ap50"]
for i, (ck, lbl) in enumerate(cls_order):
    cm = pc[ck]["C_leak"]["mean"]
    lm = pc[ck]["L_leak"]["mean"]
    dm = pc[ck]["delta_leak"]["mean"]
    ax2.plot([cm, lm], [i, i], color=C["line"], lw=1.6, zorder=2)
    ax2.plot(cm, i, "o", ms=5.0, mfc=C["navy"], mec="white", mew=0.7, zorder=4)
    ax2.plot(lm, i, "o", ms=5.0, mfc=C["amber"], mec="white", mew=0.7, zorder=4)
    ax2.text(101.5, i, f"{dm:+.2f}", va="center", ha="left", fontsize=6.9, fontweight="bold", color=C["ink"])

ax2.text(101.5, -0.75, "Δ (pp)", fontsize=6.5, color=C["mute"], ha="left", va="center")
ax2.set_yticks(range(len(cls_order)))
ax2.set_yticklabels([c[1] for c in cls_order], fontsize=7.2, color=C["ink"])
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.set_ylim(len(cls_order) - 0.4, -0.55)
ax2.set_xlim(52, 100)
ax2.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.set_xlabel("T_leak AP@0.5 (%, K=10 mean)")
ax2.set_title("Per-class T_leak gain (300 evals)", fontsize=7.8, fontweight="bold", pad=6)
hs2 = [
    plt.Line2D([], [], marker="o", ls="", ms=4.8, mfc=C["navy"], mec="white", mew=0.7, label="Clean (C)"),
    plt.Line2D([], [], marker="o", ls="", ms=4.8, mfc=C["amber"], mec="white", mew=0.7, label="Leaked (L)"),
]
ax2.legend(handles=hs2, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, fontsize=6.6, handletextpad=0.2, columnspacing=0.8)

# (c) E5 Leakage Dose-Response Curve (0%, 25%, 50%, 75%, 100%, 45 runs)
ax3 = fig.add_subplot(gs[2])
panel(ax3, "(c)", x=-0.20, y=1.07)
conds = ["C", "D25", "D50", "D75", "L"]
xs = [0, 25, 50, 75, 100]

def get_delta_series(metric_key):
    base = e5["by_dose"]["C"][metric_key]["mean"]
    means = [0.0] + [e5["by_dose"][c][f"delta_vs_C_{metric_key}"]["mean"] for c in conds[1:]]
    ses = [0.0] + [e5["by_dose"][c][f"delta_vs_C_{metric_key}"]["sd"] / np.sqrt(9) for c in conds[1:]]
    return np.array(means), np.array(ses)

m_mic, s_mic = get_delta_series("leak_micro")
m_rec, s_rec = get_delta_series("leak_inst_rec")
m_cln, s_cln = get_delta_series("clean_map50")

ax3.axhline(0, color=C["line"], lw=0.8, ls=":", zorder=1)
ax3.errorbar(xs, m_mic, yerr=s_mic, fmt="-o", ms=4.5, lw=1.6, color=C["red"], ecolor=C["red"], elinewidth=0.9, capsize=2, label="T_leak Micro-AP@0.5", zorder=4)
ax3.errorbar(xs, m_rec, yerr=s_rec, fmt="-^", ms=4.5, lw=1.6, color=C["amber"], ecolor=C["amber"], elinewidth=0.9, capsize=2, label="T_leak Inst. Recall", zorder=4)
ax3.errorbar(xs, m_cln, yerr=s_cln, fmt="--s", ms=4.0, lw=1.3, color=C["teal"], ecolor=C["teal"], elinewidth=0.8, capsize=2, label="T_clean Control", zorder=3)

ax3.text(94, m_mic[-1] - 0.72, f"+{m_mic[-1]:.2f}", ha="right", va="top", fontsize=6.6, color=C["red"], fontweight="bold")
ax3.text(94, m_rec[-1] + 0.55, f"+{m_rec[-1]:.2f}", ha="right", va="bottom", fontsize=6.6, color=C["amber"], fontweight="bold")

ax3.set_xticks(xs)
ax3.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.0, color=C["ink"])
ax3.set_xlim(-8, 108)
ax3.set_ylim(-1.6, 5.4)
ax3.set_xlabel("Retained duplicate partners (%)")
ax3.set_ylabel("Δ vs. 0% Clean (pp, mean ± SE)")
ax3.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax3.set_axisbelow(True)
ax3.set_title("Leakage dose–response", fontsize=7.8, fontweight="bold", pad=6)
ax3.legend(loc="upper left", fontsize=6.4, frameon=True, facecolor="white", edgecolor="none")

save(fig, "fig08_leakage_comparison")
print("Saved fig08_leakage_comparison.pdf and .png")
