from style_jim import *
import numpy as np, statistics as st

D = load()
ARCH = ["yolo11n", "yolov8n", "yolo11s", "yolov8s", "rtdetr-l"]
fig = plt.figure(figsize=(W2 * MM, 132 * MM))

# (a) ten seeds per detector, two datasets
ax = fig.add_axes([0.125, 0.585, 0.44, 0.36]); fig.text(0.005, 0.985, "(a)", fontsize=10, fontweight="bold", va="top")
rng = np.random.default_rng(1)
for i, a in enumerate(ARCH):
    for ds, col, off in (("NEU-DET", C["navy"], -0.19), ("GC10-DET", C["teal"], 0.19)):
        v = [r[2] for r in D["base"][ds][a]]
        y = i + off + rng.uniform(-0.09, 0.09, len(v))
        ax.scatter(v, y, s=13, c=col, edgecolors="white", linewidths=0.4, zorder=3, alpha=0.9)
        ax.plot([st.mean(v)] * 2, [i + off - 0.16, i + off + 0.16], color=C["ink"], lw=1.2, zorder=4)
        ax.text(max(v) + 0.7, i + off, f"SD {st.stdev(v):.2f}", fontsize=8, va="center", ha="left", color=C["ink"])
ax.set_yticks(range(len(ARCH))); ax.set_yticklabels([NICE[a] for a in ARCH], color=C["ink"]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(54, 84.5); ax.set_xlabel("Test mAP@0.5 (%), one dot per seed"); grid(ax, "x")
h = [plt.Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec="white", mew=0.5, label=l) for l, c in (("GC10-DET", C["teal"]), ("NEU-DET", C["navy"]))]
h.append(plt.Line2D([], [], marker="|", ls="", ms=9, mec=C["ink"], mew=1.2, label="Mean"))
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.1, columnspacing=1.2, borderaxespad=0.2)

# (b) per-class seed SD on NEU-DET
ax = fig.add_axes([0.745, 0.585, 0.24, 0.36]); fig.text(0.61, 0.985, "(b)", fontsize=10, fontweight="bold", va="top")
CL = ["scratches", "patches", "inclusion", "pitted_surface", "rolled-in_scale", "crazing"]
M = np.array([[st.stdev(D["per_class"][a][c]) for a in ARCH] for c in CL])
im = ax.imshow(M, cmap=mpl.colors.LinearSegmentedColormap.from_list("b", ["#F3F7FB", C["blue_m"], C["blue"], C["navy"]]), vmin=0.5, vmax=4.3, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 2.6 else C["ink"])
ax.set_xticks(range(len(ARCH))); ax.set_xticklabels(["11n", "v8n", "11s", "v8s", "RT-\nDETR"], color=C["ink"])
ax.set_yticks(range(len(CL))); ax.set_yticklabels([CLS[c] for c in CL], color=C["ink"])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xticks(np.arange(-.5, len(ARCH)), minor=True); ax.set_yticks(np.arange(-.5, len(CL)), minor=True)
ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
ax.set_title("Seed SD of class AP@0.5 (pp), NEU-DET", fontsize=8, loc="right", pad=5, color=C["mute"])

# (c) ten grouped splits x three seeds
ax = fig.add_axes([0.085, 0.09, 0.90, 0.345]); fig.text(0.005, 0.50, "(c)", fontsize=10, fontweight="bold", va="top")
keys = sorted(D["e2"]); allv = []
for i, k in enumerate(keys):
    v = D["e2"][k]["test_map50"]["values"]; allv += v
    ax.plot([i - 0.24, i + 0.24], [st.mean(v)] * 2, color=C["navy"], lw=2.0, zorder=3, solid_capstyle="butt")
    ax.scatter(i + np.array([-0.12, 0, 0.12]), v, s=18, c=C["blue"], edgecolors="white", linewidths=0.5, zorder=4)
    ax.text(i, max(v) + 0.55, f"{st.mean(v):.1f}", ha="center", va="bottom", fontsize=8)
gm = st.mean(allv); sdt = (D["e2_var"]["var_split"] + D["e2_var"]["var_seed"]) ** 0.5
ax.axhline(gm, color=C["mute"], lw=0.8, ls=(0, (4, 2)), zorder=1)
ax.axhspan(gm - D["e2_var"]["sd_seed"], gm + D["e2_var"]["sd_seed"], color=C["amber_m"], alpha=0.55, lw=0, zorder=0)
ax.axhspan(gm - sdt, gm + sdt, color=C["amber_l"], lw=0, zorder=-1)
ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"G{i}" for i in range(len(keys))], color=C["ink"]); ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, len(keys) - 0.4); ax.set_ylim(67.5, 82.5)
ax.set_ylabel("Test mAP@0.5 (%)"); ax.set_xlabel("Registration-grouped split of NEU-DET (YOLO11n, three seeds each)"); grid(ax)
h = [plt.Line2D([], [], marker="o", ls="", ms=5.5, mfc=C["blue"], mec="white", mew=0.5, label="Seed"),
     plt.Line2D([], [], color=C["navy"], lw=2, label="Split mean"),
     Rectangle((0, 0), 1, 1, fc=C["amber_m"], alpha=0.55, ec="none", label="Grand mean ± seed SD"),
     Rectangle((0, 0), 1, 1, fc=C["amber_l"], ec="none", label="± total SD (split and seed)")]
ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handletextpad=0.4, columnspacing=1.3, handlelength=1.3, borderaxespad=0.2)
save(fig, "figC_noise")
