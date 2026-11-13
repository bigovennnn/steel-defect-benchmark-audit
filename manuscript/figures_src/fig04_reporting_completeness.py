from style_jim import *
import numpy as np

N = 182
# (code, label, reported, ambiguous, not reported) -- aligned with Table 2 (F04 = 3/182 = 1.6%)
GROUPS = [
    ("Data and split", [("F01", "Dataset version / source", 173, 5, 4), ("F02", "Split ratio or counts", 127, 3, 52),
                        ("F04", "Duplicate / leakage audit", 3, 1, 178), ("F03", "Split manifest / image list", 0, 1, 181)]),
    ("Training configuration", [("F10", "Epoch budget", 150, 7, 25), ("F11", "Batch size", 118, 6, 58),
                                ("F09", "Input resolution", 116, 11, 55), ("F05", "Pretrained-weight source", 39, 16, 127),
                                ("F12", "Random seed", 5, 5, 172)]),
    ("Selection and evaluation", [("F14", "Baseline provenance", 74, 78, 30), ("F15", "Hardware and latency scope", 69, 49, 64),
                                  ("F07", "Evaluation-split identity", 29, 77, 76), ("F13", "Multiple runs with dispersion", 18, 2, 162),
                                  ("F06", "Checkpoint-selection criterion", 14, 9, 159), ("F08", "Validation / test separation", 10, 9, 163)]),
    ("Released artifacts", [("F16", "Working code repository", 31, 11, 140), ("F17", "Trained weights", 1, 0, 181)]),
]
COL = [C["blue"], C["amber"], C["none"]]
LAB = ["Reported with verbatim evidence", "Ambiguous or inconsistent", "Not reported"]


def wilson_ci_pct(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return max(0.0, 100.0 * (center - half)), min(100.0, 100.0 * (center + half))


fig = plt.figure(figsize=(W2 * MM, 114 * MM))
ax = fig.add_axes([0.455, 0.095, 0.41, 0.805])
y = 0
bh = 0.64
for gname, items in GROUPS:
    y0 = y
    for code, lab, r, a, nr in items:
        left = 0
        for v, c in zip((r, a, nr), COL):
            p = 100 * v / N
            if p > 0:
                ax.barh(y, p, left=left, height=bh, color=c, edgecolor="white", linewidth=0.9, zorder=3)
            left += p
        pr = 100 * r / N
        lo, hi = wilson_ci_pct(r, N)
        ax.plot([lo, hi], [y, y], color=C["navy"], lw=1.1, zorder=5)
        ax.plot([lo, lo], [y - 0.14, y + 0.14], color=C["navy"], lw=0.9, zorder=5)
        ax.plot([hi, hi], [y - 0.14, y + 0.14], color=C["navy"], lw=0.9, zorder=5)
        ax.text(102.0, y, f"{pr:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color=C["ink"])
        ax.text(119.5, y, f"({r})", va="center", ha="left", fontsize=8.0, color=C["mute"])
        ax.text(-2.0, y, lab, va="center", ha="right", fontsize=8.0, color=C["ink"])
        ax.text(-62.5, y, code, va="center", ha="left", fontsize=8.0, color=C["blue"], fontweight="bold")
        y += 1
    ax.plot([-65.5, -65.5], [y0 - 0.33, y - 1 + 0.33], color=C["navy"], lw=1.4, clip_on=False, solid_capstyle="butt")
    ax.text(
        -68.0,
        (y0 + y - 1) / 2,
        gname.replace(" and ", "\nand ").replace(" configuration", "\nconfiguration").replace(" artifacts", "\nartifacts"),
        va="center",
        ha="right",
        fontsize=8.0,
        fontweight="bold",
        color=C["navy"],
        linespacing=1.08,
    )
    y += 0.55
ax.set_ylim(y - 0.9, -0.75)
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.grid(True, color="white", lw=0.6, zorder=4)
ax.set_xlabel(f"Share of benchmark evaluations (%), N = {N} in 131 papers (whiskers: Wilson 95% CI)", fontsize=8.5)
ax.text(102.0, -0.88, "Reported", fontsize=8.0, color=C["mute"], ha="left", va="bottom")
h = [Rectangle((0, 0), 1, 1, fc=c, ec="none") for c in COL]
fig.legend(
    h,
    LAB,
    loc="upper center",
    bbox_to_anchor=(0.54, 0.992),
    ncol=3,
    fontsize=8.0,
    handlelength=1.1,
    handleheight=0.9,
    columnspacing=1.4,
    handletextpad=0.5,
    borderaxespad=0,
)
save(fig, "fig04_reporting_completeness")
save(fig, "fig1_reporting_completeness")
