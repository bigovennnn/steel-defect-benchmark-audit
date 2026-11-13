from style_jim import *

W, H = W2, 112
fig, ax = canvas(W, H)

box(ax, 2, H - 15, W - 4, 13, fc=C["navy"], ec=C["navy"], r=2)
txt(ax, W / 2, H - 6.2, "Question", size=8, color="#B9C7D8", weight="bold")
txt(ax, W / 2, H - 11.2, "How large must a reported mAP gain be before a single run can be trusted?",
    size=9.5, color="white", weight="bold")

cards = [
    ("blue", "1", "How large are\npublished gains?", "Can they be checked?",
     ["131 papers,\n182 evaluations", "17 reporting items", "claimed gains\nextracted"]),
    ("teal", "2", "How much do scores\nmove on their own?", "Nothing changed but\nseed or split",
     ["5 detectors ×\n10 seeds", "2 datasets", "10 grouped splits\n× 3 seeds"]),
    ("amber", "3", "How much does\nleakage add?", "Near-duplicates across\ntrain and test",
     ["registration graph,\nhuman-checked", "10 paired\ncounterfactual splits", "5 partner doses"]),
    ("red", "4", "Do common modules\nsurvive a re-test?", "Matched weights,\nmatched seeds",
     ["3 modules", "10 paired seeds", "same split and\ntraining budget"]),
]
LIGHT = {"blue": C["blue_l"], "teal": C["teal_l"], "amber": C["amber_l"], "red": C["red_l"]}
MID = {"blue": C["blue_m"], "teal": C["teal_m"], "amber": C["amber_m"], "red": "#EBC0B5"}
n, gap = 4, 5.0
cw = (W - 4 - gap * (n - 1)) / n
top, ch = H - 20, 63
for i, (key, num, q, sub, items) in enumerate(cards):
    x = 2 + i * (cw + gap)
    k = C[key]
    box(ax, x, top - ch, cw, ch, fc=LIGHT[key], ec=MID[key], r=2.2, z=1)
    badge(ax, x + 5.4, top - 5.8, num, k, r=2.7, size=9)
    txt(ax, x + 3, top - 14.6, q, size=8.8, weight="bold", ha="left")
    txt(ax, x + 3, top - 24.6, sub, size=8, color=C["mute"], ha="left", style="italic")
    y = top - 30.5
    for it in items:
        nl = it.count("\n") + 1
        h = 4.4 + 3.6 * nl - 1.2
        box(ax, x + 3, y - h, cw - 6, h, fc="white", ec=MID[key], r=1.2)
        ax.add_patch(Rectangle((x + 3, y - h + 1.1), 0.9, h - 2.2, fc=k, ec="none", zorder=3))
        txt(ax, x + 5.6, y - h / 2, it, size=8, ha="left")
        y -= h + 1.8
    if i < n - 1:
        arrow(ax, (x + cw + 0.6, top - 5.8), (x + cw + gap - 0.6, top - 5.8), color=C["mute"], lw=1.1, head=6)
    ax.plot([x + cw / 2] * 2, [top - ch, top - ch - 4.2], color=C["mute"], lw=0.9, zorder=3)

# yardstick bar
yb = top - ch - 4.2
box(ax, 2, yb - 10.5, W - 4, 10.5, fc="white", ec=C["navy"], lw=1.0, r=2)
txt(ax, 6, yb - 5.25, "Common yardstick", size=8.6, weight="bold", color=C["navy"], ha="left")
txt(ax, 39, yb - 5.25, "the smallest difference in test mAP@0.5 that one run per model can resolve", size=8.4, ha="left")
arrow(ax, (W / 2, yb - 10.5), (W / 2, yb - 14.3), color=C["mute"], lw=1.0)

outs = ["Reporting checklist", "Seeds-needed rule", "Released splits, logs and code"]
ow = (W - 4 - 2 * 4) / 3
for i, o in enumerate(outs):
    x = 2 + i * (ow + 4)
    box(ax, x, 2, ow, 8.6, fc=C["bg"], ec=C["line"], r=1.6)
    txt(ax, x + ow / 2, 6.3, o, size=8.4, weight="bold", color=C["ink"])
save(fig, "figA_framework")
