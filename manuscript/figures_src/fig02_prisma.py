from style_jim import *

# ---------------- EDIT THE NUMBERS HERE ----------------
D = dict(
    identified=("Records identified", "Europe PMC", 433),
    removed=None,
    screened=("Records screened", "title and abstract", 433),
    excluded=("Records excluded", [("outside steel / metal scope", 16),
                                   ("classification or segmentation only", 50),
                                   ("no NEU-DET / GC10-DET evaluation", 171)], 237),
    sought=("Reports sought for retrieval", "full text", 196),
    not_retrieved=("Reports not retrieved", [("no PMC full-text copy", 59),
                                             ("download failure", 2)], 61),
    assessed=("Reports assessed for eligibility", "full text", 135),
    ft_excluded=("Reports excluded", [("classification or segmentation only", 2),
                                      ("no required dataset experiment", 2)], 4),
    included=("Studies included in the audit", 131),
    evals=[("Benchmark evaluations", 182), ("on NEU-DET", 125), ("on GC10-DET", 57)],
)
# --------------------------------------------------------

W, H = W2, 126
fig, ax = canvas(W, H)
xm, wm = 13.5, 77.5        # main column (13.5 .. 91.0)
xr, wr = 98.5, 73.5        # side column (98.5 .. 172.0)
bh = 16.5
rows = {"identified": 107.0, "screened": 82.0, "sought": 56.5, "assessed": 32.0}
yinc, ih = 2.0, 19.5


def main(key, y):
    t, sub, n = D[key]
    box(ax, xm, y, wm, bh, fc="white", ec=C["blue"], lw=0.8)
    txt(ax, xm + 3.5, y + bh - 5.3, t, size=8.6, weight="bold", ha="left")
    txt(ax, xm + 3.5, y + 4.4, sub, size=8.0, color=C["mute"], ha="left")
    pw = 21.5
    box(ax, xm + wm - pw - 2.5, y + 3.2, pw, bh - 6.4, fc=C["blue_l"], ec="none", r=1.4, z=3)
    txt(ax, xm + wm - pw / 2 - 2.5, y + bh / 2 - 0.1, f"n = {n:,}", size=8.6, weight="bold", color=C["navy"])


def side(key, y_main):
    t, items, n = D[key]
    step = 4.4
    h = 9.0 + step * len(items) + 2.0
    y = y_main + bh / 2 - h / 2
    box(ax, xr, y, wr, h, fc=C["bg"], ec=C["line"], lw=0.65)
    txt(ax, xr + 3.2, y + h - 4.2, t, size=8.3, weight="bold", ha="left")
    txt(ax, xr + wr - 3.2, y + h - 4.2, f"n = {n:,}", size=8.3, weight="bold", ha="right", color=C["navy"])
    ax.plot([xr + 3.2, xr + wr - 3.2], [y + h - 7.3] * 2, color=C["grid"], lw=0.7, zorder=3)
    for i, (lab, k) in enumerate(items):
        yy = y + h - 10.6 - step * i
        txt(ax, xr + 3.2, yy, lab, size=8.0, color=C["mute"], ha="left")
        txt(ax, xr + wr - 3.2, yy, f"{k:,}", size=8.0, color=C["ink"], ha="right")
    arrow(ax, (xm + wm + 0.4, y_main + bh / 2), (xr - 0.4, y_main + bh / 2), color=C["line"], lw=0.95)


main("identified", rows["identified"])
main("screened", rows["screened"]);  side("excluded", rows["screened"])
main("sought", rows["sought"]);      side("not_retrieved", rows["sought"])
main("assessed", rows["assessed"]);  side("ft_excluded", rows["assessed"])
if D["removed"]:
    side("removed", rows["identified"])

order = ["identified", "screened", "sought", "assessed"]
for a, b in zip(order[:-1], order[1:]):
    arrow(ax, (xm + wm / 2, rows[a] - 0.4), (xm + wm / 2, rows[b] + bh + 0.4), color=C["blue"], lw=1.0)
arrow(ax, (xm + wm / 2, rows["assessed"] - 0.4), (xm + wm / 2, yinc + ih + 0.4), color=C["blue"], lw=1.0)

# included (full-width emphasis)
box(ax, xm, yinc, W - xm - 2.0, ih, fc=C["navy"], ec=C["navy"], r=1.8)
txt(ax, xm + 4.0, yinc + ih - 6.0, D["included"][0], size=9.0, weight="bold", color="white", ha="left")
txt(ax, xm + 4.0, yinc + 5.6, f"n = {D['included'][1]:,} papers", size=8.4, color="#C9D6E5", ha="left")
cx = 76.5
gap = 2.4
tot = W - 4.5 - cx - gap * 2
wids = [42.0, (tot - 42.0) / 2, (tot - 42.0) / 2]
for i, (lab, k) in enumerate(D["evals"]):
    x = cx + sum(wids[:i]) + gap * i
    cwid = wids[i]
    box(ax, x, yinc + 2.6, cwid, ih - 5.2, fc="#2B4E77" if i else "#35608F", ec="none", r=1.3, z=3)
    txt(ax, x + cwid / 2, yinc + ih - 6.4, f"{k:,}", size=9.8, weight="bold", color="white")
    txt(ax, x + cwid / 2, yinc + 5.6, lab, size=8.0, color="#C9D6E5")

# phase tabs (heights chosen so 8 pt vertical labels fit with padding)
phases = [("Identification", rows["identified"] - 1.5, rows["identified"] + bh + 1.5),
          ("Screening", rows["sought"] - 0.5, rows["screened"] + bh + 0.5),
          ("Eligibility", rows["assessed"] - 1.5, rows["assessed"] + bh + 1.5),
          ("Included", yinc, yinc + ih)]
for lab, y0, y1 in phases:
    box(ax, 2.0, y0, 8.8, y1 - y0, fc=C["blue_m"], ec="none", r=1.4)
    txt(ax, 6.4, (y0 + y1) / 2, lab, size=8.0, weight="bold", color=C["navy"], rotation=90)

save(fig, "fig02_prisma")
