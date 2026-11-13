from style_jim import *
import numpy as np
from matplotlib.patches import Arc

W, H = W2, 86
fig, ax = canvas(W, H)
n, gap = 6, 2.6
cw = (W - 4.0 - gap * (n - 1)) / n
ch = 66.0
y0 = 3.0
ph = 30.0          # pictogram zone height
T = C["teal"]; TL = C["teal_l"]; TM = C["teal_m"]

cards = [
    ("All image\npairs", "NEU-DET,\n200 × 200 px,\ngrayscale"),
    ("Coarse\nscreening", "16 × 16 grid,\nFFT corr.,\n4 rotations"),
    ("Fine\nregistration", "64 × 64 phase,\n200 × 200 check,\nHanning win."),
    ("Dual NCC\ntest", "pixel NCC ≥ 0.95\nand grad.\nNCC ≥ 0.90"),
    ("Duplicate\ngraph", "223 candidate\nedges, 351 imgs,\n147 comps."),
    ("Grouped\nMIP split", "1,437/180/180,\nclass-balanced,\n0 cross-edges"),
]


def bracket(x0, x1, label, color):
    y = y0 + ch + 5.2
    ax.plot([x0, x0, x1, x1], [y - 1.6, y, y, y - 1.6], color=color, lw=0.9, solid_joinstyle="miter")
    ax.text((x0 + x1) / 2, y + 1.4, label, fontsize=8.8, fontweight="bold", color=color, ha="center", va="bottom")


xs = [2.0 + i * (cw + gap) for i in range(n)]
bracket(xs[0], xs[3] + cw, "Candidate detection", C["teal"])
bracket(xs[4], xs[5] + cw, "Leak-mitigated partition", C["navy"])

rng = np.random.default_rng(3)
for i, (t, d) in enumerate(cards):
    x = xs[i]
    k = C["teal"] if i < 4 else C["navy"]
    km = TM if i < 4 else C["blue_m"]
    kl = TL if i < 4 else C["blue_l"]
    box(ax, x, y0, cw, ch, fc="white", ec=km, lw=0.8, r=1.8)
    box(ax, x + 1.5, y0 + ch - ph - 1.5, cw - 3.0, ph, fc=kl, ec="none", r=1.2, z=2)
    badge(ax, x + 1.8, y0 + ch - 1.8, str(i + 1), k, r=2.6, size=8.5)
    txt(ax, x + 2.0, y0 + ch - ph - 3.5, t, size=8.4, weight="bold", ha="left", va="top", linespacing=1.06)
    txt(ax, x + 2.0, y0 + ch - ph - 14.5, d, size=8.0, color=C["mute"], ha="left", va="top", linespacing=1.18)
    if i < n - 1:
        arrow(ax, (x + cw + 0.35, y0 + ch - ph / 2 - 1.5),
              (x + cw + gap - 0.35, y0 + ch - ph / 2 - 1.5), color=C["mute"], lw=1.0, head=4.5)

    cx, cy = x + cw / 2, y0 + ch - ph / 2 - 1.5
    if i == 0:   # stack of image tiles
        for j, (dx, dy) in enumerate([(-3.2, 2.8), (-1.6, 1.4), (0, 0)]):
            ax.add_patch(Rectangle((cx - 5.5 + dx + 1.6, cy - 6.8 + dy), 11.5, 11.5,
                                   fc=["#C9D3DC", "#AEBBC7", "#8E9DAC"][j], ec="white", lw=0.8, zorder=4 + j))
        ax.plot([cx - 2.2, cx + 5.6], [cy - 3.4, cy - 1.0], color="white", lw=1.1, zorder=8)
        ax.plot([cx - 1.6, cx + 3.4], [cy + 1.4, cy + 2.5], color="white", lw=0.75, zorder=8)
    elif i == 1:  # coarse grid + rotation arc
        g = 2.25
        vals = rng.uniform(0.25, 0.95, (4, 4))
        for a in range(4):
            for b in range(4):
                v = vals[a, b]
                col = (0.165 + (1 - v) * 0.74, 0.616 + (1 - v) * 0.33, 0.561 + (1 - v) * 0.38)
                ax.add_patch(Rectangle((cx - 2 * g - 3.8 + a * g, cy - 2 * g + b * g), g, g,
                                       fc=col, ec="white", lw=0.5, zorder=4))
        ax.add_patch(Arc((cx + 6.4, cy), 7.6, 7.6, theta1=25, theta2=305, color=T, lw=1.05, zorder=5))
        arrow(ax, (cx + 8.1, cy - 3.3), (cx + 9.2, cy - 1.8), color=T, lw=1.0, head=4)
        txt(ax, cx + 6.4, cy, "×4", size=8.0, weight="bold", color=T)
    elif i == 2:  # two overlapping frames
        ax.add_patch(Rectangle((cx - 8.6, cy - 2.2), 11.0, 11.0, fc="#AEBBC7", ec="white", lw=0.8, zorder=4))
        ax.add_patch(Rectangle((cx - 2.6, cy - 5.8), 11.0, 11.0, fc="#8E9DAC", ec="white", lw=0.8, zorder=5, alpha=0.95))
        ax.add_patch(Rectangle((cx - 2.6, cy - 2.2), 5.0, 7.4, fc=T, ec="white", lw=0.8, zorder=6))
        txt(ax, cx - 0.1, cy + 1.5, "Ω", size=9.2, weight="bold", color="white", z=7)
        arrow(ax, (cx - 8.2, cy - 5.2), (cx - 3.3, cy - 5.2), color=T, lw=0.95, head=4)
        txt(ax, cx - 5.5, cy - 9.2, "Δx, Δy", size=8.0, color=T)
    elif i == 3:  # threshold quadrant
        ox, oy, s = cx - 4.8, cy - 5.2, 13.8
        ax.add_patch(Rectangle((ox, oy), s, s, fc="white", ec=C["line"], lw=0.65, zorder=4))
        th = 0.6
        ax.add_patch(Rectangle((ox + s * th, oy + s * th), s * (1 - th), s * (1 - th), fc=TM, ec="none", zorder=5))
        ax.plot([ox + s * th] * 2, [oy, oy + s], color=T, lw=0.75, ls=(0, (2, 1.5)), zorder=6)
        ax.plot([ox, ox + s], [oy + s * th] * 2, color=T, lw=0.75, ls=(0, (2, 1.5)), zorder=6)
        pts = rng.uniform(0.05, 0.95, (26, 2))
        pts = np.vstack([pts * 0.62, rng.uniform(0.66, 0.96, (7, 2))])
        for px, py in pts:
            inq = px > th and py > th
            ax.add_patch(Circle((ox + s * px, oy + s * py), 0.42, fc=T if inq else "#AEBBC7", ec="none", zorder=7))
        txt(ax, ox + s / 2, oy - 3.6, "pixel NCC", size=8.0, color=C["mute"])
        txt(ax, ox - 3.4, oy + s / 2, "grad. NCC", size=8.0, color=C["mute"], rotation=90)
    elif i == 4:  # small graph with components
        comps = [[(-7, 4), (-3.4, 6)], [(-6.6, -2.6), (-3.2, -5.6), (-7.4, -6.6)],
                 [(1.6, 5.4), (5.2, 6.6), (6.6, 3.0), (2.8, 2.0)], [(1.0, -3.4), (4.6, -5.8)], [(7.4, -2.2)], [(-1.2, 0.4)]]
        edges = [[(0, 1)], [(0, 1), (1, 2), (0, 2)], [(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 1)], [], []]
        for cp, ed in zip(comps, edges):
            single = len(cp) == 1
            for a, b in ed:
                ax.plot([cx + cp[a][0], cx + cp[b][0]], [cy + cp[a][1], cy + cp[b][1]], color=C["navy"], lw=0.8, zorder=4)
            for px, py in cp:
                ax.add_patch(Circle((cx + px, cy + py), 1.05, fc="#AEBBC7" if single else C["navy"], ec="white", lw=0.6, zorder=5))
    elif i == 5:  # three partitions with intact components
        widths = [10.4, 4.4, 4.8]; labs = ["train", "val", "test"]; cols = [C["navy"], C["blue"], C["amber"]]
        gap_box = 0.7
        xx = cx - (sum(widths) + gap_box * 2) / 2
        for wdt, lab, col in zip(widths, labs, cols):
            ax.add_patch(FancyBboxPatch((xx, cy - 3.0), wdt, 9.8, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc="white", ec=col, lw=0.9, zorder=4))
            txt(ax, xx + wdt / 2, cy - 6.8, lab, size=8.0, color=col, weight="bold")
            xx += wdt + gap_box
        for (a, b) in [((-8.6, 4.2), (-6.3, 2.7)), ((-4.0, 4.4), (-1.8, 3.8)), ((-8.4, -1.0), (-6.0, -0.6))]:
            ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=C["navy"], lw=0.8, zorder=5)
            for p in (a, b):
                ax.add_patch(Circle((cx + p[0], cy + p[1]), 0.8, fc=C["navy"], ec="white", lw=0.4, zorder=6))
        ax.plot([cx - 3.8, cx - 1.4, cx - 2.6, cx - 3.8], [cy - 1.5, cy - 0.7, cy + 1.1, cy - 1.5], color=C["navy"], lw=0.8, zorder=5)
        for p in [(-3.8, -1.5), (-1.4, -0.7), (-2.6, 1.1)]:
            ax.add_patch(Circle((cx + p[0], cy + p[1]), 0.8, fc=C["navy"], ec="white", lw=0.4, zorder=6))
        for xo, col in [(2.7, C["blue"]), (8.0, C["amber"])]:
            ax.plot([cx + xo] * 2, [cy - 0.5, cy + 3.6], color=col, lw=0.8, zorder=5)
            for py in (-0.5, 3.6):
                ax.add_patch(Circle((cx + xo, cy + py), 0.8, fc=col, ec="white", lw=0.4, zorder=6))

save(fig, "fig05_duplicate_pipeline")
