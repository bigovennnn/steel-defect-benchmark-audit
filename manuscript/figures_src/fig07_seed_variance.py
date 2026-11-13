"""
Fig. 7: Stochastic Seed Variance and E2 Grouped-Split Variance Decomposition.
Reads verified statistics from results/cloud_297_comprehensive_stats.json.
"""
from pathlib import Path
import json
import numpy as np
from style import *

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATS = json.loads((ROOT_DIR / "results" / "cloud_297_comprehensive_stats.json").read_text(encoding="utf-8"))

fig = plt.figure(figsize=(W2 * MM, 118 * MM))
gs = fig.add_gridspec(2, 2, left=0.145, right=0.965, bottom=0.085, top=0.915,
                      hspace=0.45, wspace=0.52, height_ratios=[1.15, 0.95], width_ratios=[1.0, 1.0])

# (a) 10-seed per-class Test AP@0.5 dispersion for YOLO11n on neu_grouped_v6
ax = fig.add_subplot(gs[0, 0])
panel(ax, "(a)", x=-0.28, y=1.06)
cls_map = [
    ("scratches", "Scratches"),
    ("patches", "Patches"),
    ("inclusion", "Inclusion"),
    ("pitted_surface", "Pitted surface"),
    ("rolled-in_scale", "Rolled-in scale"),
    ("crazing", "Crazing"),
]
y11_cls = STATS["baseline_neu"]["architectures"]["yolo11n"]["per_class_test_ap50"]
y11_all = STATS["baseline_neu"]["architectures"]["yolo11n"]["test_map50"]
rows = [(lbl, y11_cls[k]["values"], y11_cls[k]["range"], False) for k, lbl in cls_map] + [
    ("All classes (mAP)", y11_all["values"], y11_all["range"], True)
]

for i, (name, vals, rng, is_last) in enumerate(rows):
    y = i + (0.45 if is_last else 0.0)
    col = C["navy"] if is_last else C["blue"]
    ax.plot([min(vals), max(vals)], [y, y], color=col, lw=3.2, alpha=0.25, solid_capstyle="round", zorder=2)
    jitter = np.linspace(-0.12, 0.12, len(vals))
    for v, j in zip(vals, jitter):
        ax.plot(v, y + j * 0.4, "o", ms=3.5, mfc=col, mec="white", mew=0.45, zorder=4)
    ax.plot([np.mean(vals)] * 2, [y - 0.25, y + 0.25], color=C["ink"], lw=1.2, zorder=5)
    ax.text(101.5, y, f"{rng:.2f}", va="center", ha="left", fontsize=7.0,
            fontweight="bold" if is_last else "normal", color=C["ink"])
    ax.text(30.5, y, name, va="center", ha="right", fontsize=7.2,
            fontweight="bold" if is_last else "normal", color=C["ink"])

ax.axhline(len(rows) - 1.28, color=C["grid"], lw=0.7)
ax.text(101.5, -0.78, "Range (pp)", fontsize=6.6, color=C["mute"], ha="left", va="center")
ax.set_ylim(len(rows) - 0.05, -0.55)
ax.set_xlim(32, 100)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("YOLO11n Test AP@0.5 (%) across 10 seeds")
ax.set_title("Per-class 10-seed dispersion (NEU-DET)", fontsize=8.0, fontweight="bold", pad=6)

# (b) 5-architecture x 10-seed Test mAP@0.5 mean +- SD & min-max range (NEU-DET & GC10-DET)
ax2 = fig.add_subplot(gs[0, 1])
panel(ax2, "(b)", x=-0.16, y=1.06)
arches = [("yolo11n", "YOLO11n"), ("yolov8n", "YOLOv8n"), ("yolo11s", "YOLO11s"), ("yolov8s", "YOLOv8s"), ("rtdetr-l", "RT-DETR-L")]

for idx, (ak, lbl) in enumerate(arches):
    neu = STATS["baseline_neu"]["architectures"][ak]["test_map50"]
    gc = STATS["baseline_gc10"]["architectures"][ak]["test_map50"]
    # Plot NEU (teal/navy) and GC10 (blue)
    for ddict, y_off, col, ds_lbl in [(neu, idx - 0.16, C["navy"], "NEU-DET"), (gc, idx + 0.16, C["teal"], "GC10-DET")]:
        ax2.plot([ddict["min"], ddict["max"]], [y_off, y_off], color=col, lw=1.4, alpha=0.45, zorder=2)
        ax2.plot([ddict["mean"] - ddict["sd"], ddict["mean"] + ddict["sd"]], [y_off, y_off], color=col, lw=3.8, alpha=0.85, zorder=3)
        ax2.plot(ddict["mean"], y_off, "o", ms=4.2, mfc="white", mec=col, mew=1.1, zorder=4)
        ax2.text(ddict["max"] + 0.7, y_off, f"R={ddict['range']:.2f}", va="center", ha="left", fontsize=6.4, color=col)

ax2.set_yticks(range(len(arches)))
ax2.set_yticklabels([a[1] for a in arches], fontsize=7.4, color=C["ink"])
ax2.invert_yaxis()
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.set_xlim(52, 87.5)
ax2.set_ylim(len(arches) - 0.3, -0.75)
ax2.set_xlabel("Held-out Test mAP@0.5 (%, 10 seeds)")
ax2.xaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax2.set_axisbelow(True)
ax2.set_title("5-detector 10-seed baselines", fontsize=8.0, fontweight="bold", pad=16)
hs2 = [
    plt.Line2D([], [], color=C["navy"], lw=3.0, label="NEU-DET"),
    plt.Line2D([], [], color=C["teal"], lw=3.0, label="GC10-DET"),
]
ax2.legend(handles=hs2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.6, frameon=False, handlelength=1.4, columnspacing=1.0)

# (c) E2 Zero-Leakage Grouped-Split Variance Decomposition (10 splits x 3 seeds = 30 runs)
ax3 = fig.add_subplot(gs[1, :])
panel(ax3, "(c)", x=-0.125, y=1.08)
e2 = STATS["e2_variance_decomposition"]
splits = sorted(e2["per_split"].keys())
anova50 = e2["variance_decomposition"]["test_map50"]
grand_mean = anova50["overall_summary"]["mean"]

ax3.axhline(grand_mean, color=C["line"], lw=1.0, ls="--", zorder=1)
ax3.axhspan(grand_mean - anova50["sd_split"], grand_mean + anova50["sd_split"], color=C["amber"], alpha=0.14, zorder=0)

for i, sp in enumerate(splits):
    sp_data = e2["per_split"][sp]["test_map50"]
    vals = sp_data["values"]
    m = sp_data["mean"]
    ax3.plot([i, i], [min(vals), max(vals)], color=C["blue"], lw=1.6, alpha=0.5, zorder=2)
    for j, v in enumerate(vals):
        ax3.plot(i + (j - 1) * 0.09, v, "o", ms=3.8, mfc=C["blue"], mec="white", mew=0.5, zorder=4)
    ax3.plot([i - 0.22, i + 0.22], [m, m], color=C["navy"], lw=1.7, zorder=5)
    ax3.text(i, max(vals) + 0.45, f"{m:.1f}", ha="center", va="bottom", fontsize=6.6, color=C["ink"])

ax3.set_xticks(range(len(splits)))
ax3.set_xticklabels([f"G{k}" + ("*" if k == 0 else "") for k in range(len(splits))], fontsize=7.4, color=C["ink"])
ax3.set_xlim(-0.6, len(splits) - 0.4)
ax3.set_ylim(67.2, 83.2)
ax3.set_ylabel("Test mAP@0.5 (%)")
ax3.set_xlabel("Zero-leakage grouped split realization (G0* = primary NEU-Grouped; 3 seeds per split)")
ax3.yaxis.grid(True, color=C["grid"], lw=0.5, zorder=0)
ax3.set_axisbelow(True)
ax3.set_title(
    f"Grouped-split ANOVA (30 runs): Split span = {anova50['split_means_summary']['range']:.2f} pp | "
    f"SD_split = {anova50['sd_split']:.2f} pp ({anova50['pct_split']:.1f}%) vs. SD_seed = {anova50['sd_seed']:.2f} pp ({anova50['pct_seed']:.1f}%)",
    fontsize=7.3, fontweight="bold", pad=6
)

save(fig, "fig07_seed_variance")
print("Saved fig07_seed_variance.pdf and .png")
