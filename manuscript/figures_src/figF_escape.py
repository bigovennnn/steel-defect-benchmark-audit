from style_jim import *
import numpy as np

D = load(); E = D["escape"]
fig = plt.figure(figsize=(W2 * MM, 68 * MM))
rows = [("Leaked images:\nimage escape rate", "leak_img_escape"), ("Other images:\nimage escape rate", "clean_img_escape"),
        ("Leaked images:\ninstance miss rate", "leak_inst_miss"), ("Other images:\ninstance miss rate", "clean_inst_miss")]
for p, (key, lab) in enumerate((("fppi_0.15", "0.15 false positives per image"), ("fppi_0.25", "0.25 false positives per image"))):
    ax = fig.add_axes([0.20 + p * 0.405, 0.19, 0.375, 0.60])
    fig.text(0.005 if p == 0 else 0.595, 0.975, "(a)" if p == 0 else "(b)", fontsize=10, fontweight="bold", va="top")
    for i, (rl, k) in enumerate(rows):
        c, l = E[key][k]["C"], E[key][k]["L"]
        ax.plot([c["mean"], l["mean"]], [i, i], color=C["line"], lw=1.6, zorder=2)
        for d, col, off in ((c, C["navy"], -0.16), (l, C["amber"], 0.16)):
            ax.plot([d["ci95_lo"], d["ci95_hi"]], [i + off] * 2, color=col, lw=1.3, zorder=3, solid_capstyle="butt")
            ax.plot(d["mean"], i + off, "o", ms=6, mfc=col, mec="white", mew=0.6, zorder=4)
        dl = E[key][k]["delta_L_minus_C"]
        ax.text(61, i, ("0.0" if abs(dl["mean"]) < 0.05 else f"{dl['mean']:+.1f}"), ha="right", va="center", fontsize=8, fontweight="bold" if dl["ci95_hi"] < 0 or dl["ci95_lo"] > 0 else "normal")
    ax.axhline(1.5, color=C["grid"], lw=0.8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows] if p == 0 else [], color=C["ink"]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 62); ax.set_xticks([0, 10, 20, 30, 40, 50]); ax.set_xlabel(f"Rate (%) at {lab}"); grid(ax, "x")
    ax.text(61, -0.72, "Change (pp)", ha="right", va="center", fontsize=8, color=C["mute"])
    if p == 0:
        h = [plt.Line2D([], [], marker="o", color=C["navy"], lw=1.3, ms=6, mec="white", mew=0.6, label="No partners in training"),
             plt.Line2D([], [], marker="o", color=C["amber"], lw=1.3, ms=6, mec="white", mew=0.6, label="Partners in training")]
        ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.5, 1.03), ncol=2, handlelength=1.6, columnspacing=1.5, borderaxespad=0)
save(fig, "figF_escape")
print({k: {kk: (round(E[k][kk]["C"]["mean"], 2), round(E[k][kk]["L"]["mean"], 2), round(E[k][kk]["delta_L_minus_C"]["mean"], 2), round(E[k][kk]["delta_L_minus_C"]["ci95_lo"], 2), round(E[k][kk]["delta_L_minus_C"]["ci95_hi"], 2)) for kk in ("leak_img_escape", "clean_img_escape", "leak_inst_miss", "clean_inst_miss")} for k in ("fppi_0.15", "fppi_0.25")})
