from style_jim import *
import numpy as np, statistics as st, math

D = load()
ARCH = ["yolo11n", "yolov8n", "yolo11s", "yolov8s", "rtdetr-l"]
Z = 1.96 * math.sqrt(2)
sd = {ds: [st.stdev([r[2] for r in D["base"][ds][a]]) for a in ARCH] for ds in ("NEU-DET", "GC10-DET")}
thr = {ds: (Z * min(v), Z * max(v)) for ds, v in sd.items()}            # single-run resolvable difference, same split
thr_cross = Z * math.sqrt(D["e2_var"]["var_split"] + D["e2_var"]["var_seed"])  # different splits, YOLO11n on NEU-DET
claims = {ds: [float(c["delta"]) for c in D["claims"] if c["dataset"] == ds and c["delta"] not in (None, "", "None")] for ds in ("NEU-DET", "GC10-DET")}
allc = sorted(claims["NEU-DET"] + claims["GC10-DET"])
q = lambda p: float(np.percentile(allc, p))
print("thresholds", thr, "cross-split", round(thr_cross, 2), "claims n", len(allc), "median", q(50), "IQR", q(25), q(75))
for ds in claims:
    v = claims[ds]; print(ds, len(v), "below lower thr: %.1f%%  below upper thr: %.1f%%" % (100 * np.mean(np.array(v) < thr[ds][0]), 100 * np.mean(np.array(v) < thr[ds][1])))

fig = plt.figure(figsize=(W2 * MM, 134 * MM))
COL = {"NEU-DET": C["navy"], "GC10-DET": C["teal"]}

# (a) published gains against what one run can resolve
ax = fig.add_axes([0.075, 0.60, 0.50, 0.32]); fig.text(0.005, 0.985, "(a)", fontsize=10, fontweight="bold", va="top")
bins = np.arange(-2, 27, 1)
ax.hist([np.clip(claims["NEU-DET"], -2, 26), np.clip(claims["GC10-DET"], -2, 26)], bins=bins, stacked=True,
        color=[COL["NEU-DET"], COL["GC10-DET"]], edgecolor="white", linewidth=0.8, zorder=3)
ymax = ax.get_ylim()[1] * 1.05
for ds, y0, y1 in (("NEU-DET", 0.52, 1.0), ("GC10-DET", 0.0, 0.48)):
    pass
ax.axvspan(*thr["NEU-DET"], color=C["blue_m"], alpha=0.55, lw=0, zorder=1)
ax.axvspan(*thr["GC10-DET"], color=C["teal_m"], alpha=0.55, lw=0, zorder=1)
ax.set_ylim(0, ymax); ax.set_xlim(-2, 26); ax.set_xticks(range(0, 27, 5))
ax.set_xlabel("Claimed gain over baseline, mAP@0.5 (pp)"); ax.set_ylabel("Number of evaluations"); grid(ax)
h = [Rectangle((0, 0), 1, 1, fc=COL["NEU-DET"], ec="none"), Rectangle((0, 0), 1, 1, fc=COL["GC10-DET"], ec="none"),
     Rectangle((0, 0), 1, 1, fc=C["blue_m"], alpha=0.55, ec="none"), Rectangle((0, 0), 1, 1, fc=C["teal_m"], alpha=0.55, ec="none")]
ax.legend(h, ["Claims on NEU-DET", "Claims on GC10-DET", "Smallest gain one run resolves, NEU-DET", "Same, GC10-DET"],
          loc="upper right", handlelength=1.0, handleheight=0.8, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)

# (c) seeds needed
ax = fig.add_axes([0.70, 0.60, 0.285, 0.32]); fig.text(0.625, 0.985, "(b)", fontsize=10, fontweight="bold", va="top")
d = np.linspace(0.5, 6, 200); K = 2 * (1.96 + 0.84) ** 2
for ds in ("GC10-DET", "NEU-DET"):
    lo, hi = K * min(sd[ds]) ** 2 / d ** 2, K * max(sd[ds]) ** 2 / d ** 2
    ax.fill_between(d, np.maximum(lo, 1), np.maximum(hi, 1), color=COL[ds], alpha=0.28, lw=0, zorder=2)
    ax.plot(d, np.maximum(np.sqrt(lo * hi), 1), color=COL[ds], lw=1.6, zorder=3, label=ds)
ax.set_yscale("log"); ax.set_ylim(1, 300); ax.set_xlim(0.5, 6)
ax.set_yticks([1, 3, 10, 30, 100, 300]); ax.set_yticklabels(["1", "3", "10", "30", "100", "300"]); ax.minorticks_off()
ax.set_xticks([1, 2, 3, 4, 5, 6]); ax.set_xlabel("True gain to detect, mAP@0.5 (pp)"); ax.set_ylabel("Seeds needed per model")
grid(ax); grid(ax, "x"); ax.legend(loc="upper right", handlelength=1.4, borderaxespad=0.2)

# (b) one ruler for every quantity
ax = fig.add_axes([0.455, 0.085, 0.53, 0.37]); fig.text(0.005, 0.50, "(c)", fontsize=10, fontweight="bold", va="top")
cf = D["cf"]["map50"]
rows = [
    ("Published gains, median and interquartile range", "iqr", (q(25), q(50), q(75)), C["mute"]),
    ("Smallest gain one run per model resolves, NEU-DET", "range", thr["NEU-DET"], C["navy"]),
    ("Smallest gain one run per model resolves, GC10-DET", "range", thr["GC10-DET"], C["teal"]),
    ("Same, if the two models use different splits (NEU-DET)", "pt", (thr_cross,), C["navy"]),
    ("Leakage effect on the full test set", "ci", (cf["delta_all"]["ci95_lo"], cf["delta_all"]["mean"], cf["delta_all"]["ci95_hi"]), C["amber"]),
    ("Leakage effect on leaked test images only", "ci", (cf["delta_leak"]["ci95_lo"], cf["delta_leak"]["mean"], cf["delta_leak"]["ci95_hi"]), C["amber"]),
]
for i, (lab, kind, v, col) in enumerate(rows):
    if kind == "range":
        ax.barh(i, v[1] - v[0], left=v[0], height=0.42, color=col, zorder=3)
        ax.text(v[1] + 0.2, i, f"{v[0]:.1f}–{v[1]:.1f}", va="center", ha="left", fontsize=8)
    elif kind == "pt":
        ax.plot(v[0], i, "D", ms=6.5, mfc=col, mec="white", mew=0.6, zorder=4)
        ax.text(v[0] + 0.25, i, f"{v[0]:.1f}", va="center", ha="left", fontsize=8)
    else:
        ax.plot([v[0], v[2]], [i, i], color=col, lw=1.8, zorder=3, solid_capstyle="butt")
        ax.plot(v[1], i, "o" if kind == "ci" else "s", ms=6.5, mfc=col, mec="white", mew=0.6, zorder=4)
        ax.text(v[2] + 0.2, i, f"{v[1]:+.2f} [{v[0]:+.2f}, {v[2]:+.2f}]" if kind == "ci" else f"{v[1]:.1f} [{v[0]:.1f}, {v[2]:.1f}]",
                va="center", ha="left", fontsize=8)
ax.axvline(0, color=C["line"], lw=0.8)
for y in (0.5, 3.5):
    ax.axhline(y, color=C["grid"], lw=0.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=C["ink"]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(-1, 10.5); ax.set_xticks(range(0, 11, 2)); ax.set_xlabel("Difference in test mAP@0.5 (pp)"); grid(ax, "x")
save(fig, "figD_ruler")
