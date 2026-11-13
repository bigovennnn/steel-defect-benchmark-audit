from style_jim import *
import numpy as np

SIZES = {2: 110, 3: 26, 4: 8, 6: 1, 7: 1, 8: 1}                   # component size -> count
CLS_LIST = [("Scratches", 149, 299), ("Pitted surface", 64, 299), ("Inclusion", 47, 300),
            ("Crazing", 44, 300), ("Rolled-in scale", 33, 300), ("Patches", 14, 299)]   # images in a component / images in class
EDGES = [("Train–test", 32), ("Train–val", 35), ("Val–test", 7)]   # naive split; grouped split has 0 in every row
TOTAL_EDGES = 223

fig = plt.figure(figsize=(W2 * MM, 72 * MM))

# (a) component sizes
ax = fig.add_axes([0.075, 0.21, 0.20, 0.65])
fig.text(0.005, 0.96, "(a)", fontsize=10, fontweight="bold", va="top")
xs = list(range(2, 9)); ys = [SIZES.get(k, 0) for k in xs]
ax.bar(xs, ys, width=0.66, color=C["navy"], zorder=3)
for x, v in zip(xs, ys):
    if v:
        ax.text(x, v + 2.5, str(v), ha="center", va="bottom", fontsize=8.0, color=C["ink"])
ax.set_xticks(xs); ax.set_xlabel("Images per component"); ax.set_ylabel("Number of components")
ax.set_ylim(0, 128); grid(ax, "y")
ax.text(0.97, 0.96, "147 components\n351 images", transform=ax.transAxes, ha="right", va="top",
        fontsize=8.0, color=C["mute"], bbox=dict(fc="white", ec="none", pad=1.2), zorder=4)

# (b) share of each class involved
ax = fig.add_axes([0.408, 0.21, 0.218, 0.65])
fig.text(0.295, 0.96, "(b)", fontsize=10, fontweight="bold", va="top")
names = [c[0] for c in CLS_LIST]; pct = [100 * c[1] / c[2] for c in CLS_LIST]
yy = np.arange(len(CLS_LIST))
ax.barh(yy, pct, height=0.62, color=C["teal"], zorder=3)
for y, p, c in zip(yy, pct, CLS_LIST):
    ax.text(p + 1.5, y, f"{p:.1f}% ({c[1]})", va="center", ha="left", fontsize=8.0, color=C["ink"])
ax.set_yticks(yy); ax.set_yticklabels(names, color=C["ink"]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0, 102); ax.set_xticks([0, 25, 50, 75]); ax.set_xlabel("Images in a component (%)")
grid(ax, "x")

# (c) cross-split edges
ax = fig.add_axes([0.735, 0.21, 0.252, 0.65])
fig.text(0.662, 0.96, "(c)", fontsize=10, fontweight="bold", va="top")
lab = [e[0] for e in EDGES]; v = [e[1] for e in EDGES]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2 - 0.02, v, width=w, color=C["amber"], zorder=3, label="Naive random split")
ax.bar(x + w / 2 + 0.02, [0.6] * 3, width=w, color=C["navy"], zorder=3, label="Grouped MIP split")
for xi, vi in zip(x, v):
    ax.text(xi - w / 2 - 0.02, vi + 1.0, str(vi), ha="center", va="bottom", fontsize=8.0, color=C["ink"])
    ax.text(xi + w / 2 + 0.02, 1.5, "0", ha="center", va="bottom", fontsize=8.0, color=C["ink"])
ax.set_xticks(x); ax.set_xticklabels(lab, color=C["ink"]); ax.tick_params(axis="x", length=0)
ax.set_ylabel("Cross-subset candidate edges"); ax.set_ylim(0, 56); grid(ax, "y")
ax.legend(loc="lower right", bbox_to_anchor=(1.02, 1.01), ncol=1, handlelength=1.1,
          handleheight=0.8, borderaxespad=0, labelspacing=0.25)

save(fig, "fig06_duplicate_statistics")
