from style_jim import *
from matplotlib.patches import Polygon

SHOW_RATES = True   # set False to draw the taxonomy without audit percentages
N_EVAL = 182
STAGES = [
    ("Dataset", [("F01", "Dataset and\nversion", 95.1), ("F04", "Duplicate\naudit", 1.6)]),
    ("Data split", [("F02", "Split ratio\nor counts", 69.8), ("F03", "Split\nmanifest", 0.0)]),
    ("Training", [("F05", "Pretrained\nweights", 21.4), ("F09", "Input\nresolution", 63.7),
                  ("F10", "Epoch\nbudget", 82.4), ("F11", "Batch\nsize", 64.8), ("F12", "Random\nseed", 2.7)]),
    ("Model\nselection", [("F06", "Checkpoint\ncriterion", 7.7), ("F08", "Val / test\nseparation", 5.5)]),
    ("Evaluation", [("F07", "Eval. split\nidentity", 15.9), ("F13", "Multi-run\ndispersion", 9.9),
                    ("F14", "Baseline\nprovenance", 40.7), ("F15", "Hardware\nand latency", 37.9)]),
    ("Release", [("F16", "Code\nrepository", 17.0), ("F17", "Trained\nweights", 0.5)]),
]

W, H = W2, 114
fig, ax = canvas(W, H)
n = len(STAGES)
gap = 2.0
tip = 2.0
cw = (W - 4.0 - tip - gap * (n - 1)) / n
hh = 15.0     # header chevron height
ytop = H - 2.0
ch, cg = 14.8, 2.2   # chip height / gap

for i, (name, fields) in enumerate(STAGES):
    x = 2.0 + i * (cw + gap)
    y0 = ytop - hh
    pts = [(x, ytop), (x + cw, ytop), (x + cw + tip, y0 + hh / 2), (x + cw, y0), (x, y0)]
    if i:
        pts.append((x + tip, y0 + hh / 2))
    ax.add_patch(Polygon(pts, closed=True, fc=C["navy"], ec="white", lw=1.2, zorder=4 + (n - i), joinstyle="round"))
    cx = x + cw / 2 + (tip / 2 if i else 0.0)
    if "\n" in name:
        txt(ax, cx, y0 + hh - 5.2, name, size=8.4, weight="bold", color="white", z=20, linespacing=1.02)
        txt(ax, cx, y0 + 2.6, f"Stage {i + 1}", size=8.0, color="#B9C7D8", z=20)
    else:
        txt(ax, cx, y0 + hh - 4.8, name, size=8.6, weight="bold", color="white", z=20)
        txt(ax, cx, y0 + 3.4, f"Stage {i + 1}", size=8.0, color="#B9C7D8", z=20)
    # vertical guide line
    ax.plot([x + cw / 2] * 2, [y0 - 0.2, y0 - 3.0 - len(fields) * (ch + cg) + cg + ch / 2],
            color=C["grid"], lw=0.8, zorder=1)
    for j, (code, lab, pct) in enumerate(fields):
        y = y0 - 3.0 - ch - j * (ch + cg)
        box(ax, x, y, cw, ch, fc="white", ec=C["blue_m"], lw=0.75, r=1.4, z=2)
        txt(ax, x + 2.0, y + ch - 3.2, code, size=8.0, weight="bold", color=C["blue"], ha="left")
        if SHOW_RATES:
            txt(ax, x + cw - 2.0, y + ch - 3.2, f"{pct:.1f}%", size=8.0, weight="bold", ha="right", color=C["ink"])
        txt(ax, x + 2.0, y + 6.4, lab, size=8.0, color=C["ink"], ha="left", va="center", linespacing=1.06)
        if SHOW_RATES:
            bw = cw - 4.0
            ax.add_patch(FancyBboxPatch((x + 2.0, y + 1.6), bw, 1.3, boxstyle="round,pad=0,rounding_size=0.65",
                                        fc=C["none"], ec="none", zorder=3))
            if pct > 0:
                ax.add_patch(FancyBboxPatch((x + 2.0, y + 1.6), max(bw * pct / 100, 1.3), 1.3,
                                            boxstyle="round,pad=0,rounding_size=0.65", fc=C["blue"], ec="none", zorder=4))

if SHOW_RATES:
    yl = 3.8
    ax.add_patch(FancyBboxPatch((2.0, yl - 0.65), 10.0, 1.3, boxstyle="round,pad=0,rounding_size=0.65", fc=C["none"], ec="none"))
    ax.add_patch(FancyBboxPatch((2.0, yl - 0.65), 5.0, 1.3, boxstyle="round,pad=0,rounding_size=0.65", fc=C["blue"], ec="none"))
    txt(ax, 13.8, yl, f"Share of the {N_EVAL} audited benchmark evaluations reporting the item with verbatim evidence",
        size=8.0, color=C["mute"], ha="left")

save(fig, "fig03_protocol_pipeline")
