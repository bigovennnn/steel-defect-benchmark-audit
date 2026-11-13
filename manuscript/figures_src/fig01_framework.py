from style import *

W, H = W2, 122
fig, ax = canvas(W, H)

# ---------- top: problem statement ----------
box(ax, 2, H - 15, W - 4, 13, fc=C["navy"], ec=C["navy"], r=2)
txt(ax, W / 2, H - 6.4, "Research problem", size=6.5, color="#B9C7D8", weight="bold")
txt(ax, W / 2, H - 11.0,
    "Are reported detector improvements on NEU-DET and GC10-DET larger than split, seed, and duplicate-leakage noise?",
    size=7.9, color="white", weight="bold")

# ---------- three modules ----------
cols = [
    dict(key="blue", tag="A", title="Literature evidence audit",
         sub="131 papers, 182 evaluations",
         steps=[("Systematic PRISMA search", "433 Europe PMC records;\n131 PMC journal studies"),
                ("17-field protocol audit", "F01–F17 verbatim coding\nwith Wilson 95% CIs"),
                ("Claimed-gain extraction", "174 ΔmAP@0.5 comparisons\n(median +3.80 pp)")],
         out=("Reporting compliance gaps", "0% manifests; 89% no SD")),
    dict(key="teal", tag="B", title="Controlled benchmark",
         sub="170 clean cloud runs",
         steps=[("Duplicate & coil grouping", "223 SE(2) edges (NEU-DET);\n467 coils (GC10-DET)"),
                ("10-seed detector baselines", "5 detectors × 10 seeds on\nzero-leakage splits (100 runs)"),
                ("Split ANOVA & modules", "10 grouped splits × 3 seeds +\n10-seed CBAM/EMA/BiFPN")],
         out=("Split & seed noise floors", "8.87 pp split; 2.6–6.6 pp seed")),
    dict(key="amber", tag="C", title="Counterfactual leakage",
         sub="127 CF runs + escape audit",
         steps=[("K=10 partner replacement", "hold test images fixed;\nswap only 31 train partners"),
                ("5-dose leakage scaling", "0% to 100% partner doses\n(matched K=0..2 × 3 seeds)"),
                ("Precision & escape metrics", "Pooled Micro-AP, Recall &\nmatched-FP escape rate")],
         out=("Causal leakage inflation", "+4.08 pp mAP; −3.0 pp escape")),
]
gap = 4
cw = (W - 4 - 2 * gap) / 3
top = H - 20
col_h = 78
for i, c in enumerate(cols):
    x0 = 2 + i * (cw + gap)
    k, kl, km = C[c["key"]], C[c["key"] + "_l"], C[c["key"] + "_m"]
    box(ax, x0, top - col_h, cw, col_h, fc=kl, ec=km, lw=0.7, r=2.2, z=1)
    badge(ax, x0 + 5.2, top - 6, c["tag"], k)
    txt(ax, x0 + 9.5, top - 5.6, c["title"], size=8.1, weight="bold", ha="left", va="baseline")
    txt(ax, x0 + 9.5, top - 8.9, c["sub"], size=6.7, ha="left", va="baseline", color=C["mute"], style="italic")
    bh, bg_, y = 13.4, 3.6, top - 12.6
    for j, (t, d) in enumerate(c["steps"]):
        yb = y - bh
        box(ax, x0 + 3, yb, cw - 6, bh, fc="white", ec=km, lw=0.7, r=1.4)
        ax.add_patch(Rectangle((x0 + 3, yb + 1.4), 0.9, bh - 2.8, fc=k, ec="none", zorder=3))
        txt(ax, x0 + 6, yb + bh - 3.4, t, size=7.4, weight="bold", ha="left")
        txt(ax, x0 + 6, yb + 4.7, d, size=6.5, ha="left", color=C["mute"])
        arrow(ax, (x0 + cw / 2, yb - 0.3), (x0 + cw / 2, yb - bg_ + 0.3), color=k, lw=0.9)
        y = yb - bg_
    oh = 11.2
    yb = y - oh
    box(ax, x0 + 3, yb, cw - 6, oh, fc=k, ec=k, r=1.4)
    txt(ax, x0 + cw / 2, yb + oh - 3.9, c["out"][0], size=7.3, weight="bold", color="white")
    txt(ax, x0 + cw / 2, yb + 3.1, c["out"][1], size=6.4, color="white")
    c["xc"] = x0 + cw / 2
    c["ybot"] = top - col_h

# ---------- converge ----------
ys = 18.0
bus = 21.5
for c in cols:
    ax.plot([c["xc"], c["xc"]], [c["ybot"], bus], color=C["mute"], lw=0.9, zorder=3, solid_capstyle="butt")
ax.plot([cols[0]["xc"], cols[2]["xc"]], [bus, bus], color=C["mute"], lw=0.9, zorder=3)
arrow(ax, (W / 2, bus), (W / 2, ys + 0.2), color=C["mute"])

box(ax, 2, 2, W - 4, ys - 2, fc="white", ec=C["navy"], lw=0.9, r=2)
segs = [
    ("Evidence synthesis", "38.5% of claims ≤ 3.0 pp;\n10-seed modules null (+0.01–0.33 pp)"),
    ("Inspection practice", "fixed-FP escape drops −21% to −25%\nunder cross-split leakage"),
    ("Open benchmark kit", "grouped manifests, SHA-256 hashes,\n297-run logs & audit code"),
]
sw = (W - 4) / 3
for i, (t, d) in enumerate(segs):
    xc = 2 + sw * i + sw / 2
    txt(ax, xc, ys - 4.0, t, size=7.7, weight="bold", color=C["navy"])
    txt(ax, xc, 7.2, d, size=6.5, color=C["mute"])
    if i:
        ax.plot([2 + sw * i] * 2, [3.8, ys - 1.8], color=C["grid"], lw=0.8, zorder=3)

save(fig, "fig01_framework")
