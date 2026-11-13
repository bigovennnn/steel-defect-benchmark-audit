from style_jim import *
import numpy as np, statistics as st, math

D = load(); cf = D["cf"]; ps = D["cf_per_split"]
fig = plt.figure(figsize=(W2 * MM, 88 * MM))

# (a) paired split-level scores, leaked and clean test images
ax = fig.add_axes([0.07, 0.17, 0.25, 0.60]); fig.text(0.005, 0.975, "(a)", fontsize=10, fontweight="bold", va="top")
keys = sorted(ps)
Cl = [ps[k]["C_leak_map50"] for k in keys]; Ll = [ps[k]["L_leak_map50"] for k in keys]
Cc = cf["map50"]["C_clean"]["values"]; Lc = cf["map50"]["L_clean"]["values"]
for a, b in zip(Cl, Ll):
    ax.plot([0, 1], [a, b], color=C["amber"], lw=0.9, alpha=0.75, zorder=2)
for a, b in zip(Cc, Lc):
    ax.plot([2.1, 3.1], [a, b], color=C["line"], lw=0.9, zorder=2)
for x, v, col in ((0, Cl, C["amber"]), (1, Ll, C["amber"]), (2.1, Cc, C["mute"]), (3.1, Lc, C["mute"])):
    ax.scatter([x] * len(v), v, s=14, c=col, edgecolors="white", linewidths=0.4, zorder=3)
    ax.plot([x - 0.22, x + 0.22], [st.mean(v)] * 2, color=C["ink"], lw=1.4, zorder=4)
ax.set_xticks([0, 1, 2.1, 3.1]); ax.set_xticklabels(["Without", "With", "Without", "With"], color=C["ink"])
ax.set_xlabel("Training partners of the leaked images")
ax.tick_params(axis="x", length=0); ax.set_xlim(-0.5, 3.6); ax.set_ylim(55, 92); grid(ax)
ax.set_ylabel("Test mAP@0.5 (%), 5-seed mean per split")
ax.text(0.5, 1.04, "Leaked images", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["amber"])
ax.text(2.6, 1.04, "Other images", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["mute"])

# (b) effect by evaluation subset
ax = fig.add_axes([0.505, 0.17, 0.17, 0.60]); fig.text(0.365, 0.975, "(b)", fontsize=10, fontweight="bold", va="top")
rows = [("Leaked\nimages", "delta_leak"), ("Other\nimages", "delta_clean"), ("Full\ntest set", "delta_all")]
for i, (lab, k) in enumerate(rows):
    for m, off, col, mk in (("map50", -0.17, C["navy"], "o"), ("map50_95", 0.17, C["teal"], "s")):
        d = cf[m][k]
        ax.plot([d["ci95_lo"], d["ci95_hi"]], [i + off] * 2, color=col, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot(d["mean"], i + off, mk, ms=6, mfc=col, mec="white", mew=0.6, zorder=4)
ax.axvline(0, color=C["line"], lw=0.8)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], color=C["ink"]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(-1.5, 7); ax.set_xticks([0, 2, 4, 6]); ax.set_xlabel("Effect of partners (pp),\nmean and 95% CI over 10 splits"); grid(ax, "x")
h = [plt.Line2D([], [], marker="o", color=C["navy"], lw=1.6, ms=6, mec="white", mew=0.6, label="mAP@0.5"),
     plt.Line2D([], [], marker="s", color=C["teal"], lw=1.6, ms=6, mec="white", mew=0.6, label="mAP@[0.5:0.95]")]
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.55, 1.0), ncol=1, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)

# (c) dose series
ax = fig.add_axes([0.775, 0.17, 0.21, 0.60]); fig.text(0.685, 0.975, "(c)", fontsize=10, fontweight="bold", va="top")
dose = D["dose"]; xs = [0, 25, 50, 75, 100]; order = ["C", "D25", "D50", "D75", "L"]
series = [("Leaked, macro mAP@0.5", "delta_vs_C_leak_map50", C["amber"], "o"),
          ("Leaked, pooled AP@0.5", "delta_vs_C_leak_micro", C["red"], "D"),
          ("Other, macro mAP@0.5", "delta_vs_C_clean_map50", C["mute"], "s")]
for j, (lab, k, col, mk) in enumerate(series):
    m = [0] + [dose[o][k]["mean"] for o in order[1:]]
    se = [0] + [dose[o][k]["sd"] / math.sqrt(dose[o][k]["n"]) for o in order[1:]]
    xo = np.array(xs) + (j - 1) * 2.2
    ax.errorbar(xo, m, yerr=se, color=col, lw=1.4, marker=mk, ms=5.5, mec="white", mew=0.6, elinewidth=0.9, capsize=0, zorder=3 + j, label=lab)
ax.axhline(0, color=C["line"], lw=0.8)
ax.set_xticks(xs); ax.set_xlim(-8, 108); ax.set_xlabel("Training partners retained (%)"); ax.set_ylabel("Change from 0% (pp), mean ± SE")
grid(ax)
ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 1.0), ncol=1, handlelength=1.6, labelspacing=0.25, borderaxespad=0.2)
save(fig, "figE_counterfactual")
