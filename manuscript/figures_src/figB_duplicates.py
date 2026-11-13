from style_jim import *
import numpy as np, collections

D = load(); P = D["pairs"]
A = [p for p in P if p["source_pool"].startswith("Pool_A")]
VERD = {2: ("Both annotators confirm", C["navy"], "o"), 1: ("One confirms", C["amber"], "s"), 0: ("Both reject", C["red"], "X")}

fig = plt.figure(figsize=(W2 * MM, 74 * MM))
R = [[0.075, 0.175, 0.235, 0.53], [0.405, 0.175, 0.235, 0.53], [0.795, 0.175, 0.19, 0.53]]
def plab(s, x):
    fig.text(x, 0.975, s, fontsize=10, fontweight="bold", ha="left", va="top")

# (a) pixel vs gradient NCC by human verdict
ax = fig.add_axes(R[0]); plab("(a)", 0.005)
for v in (2, 1, 0):
    q = [p for p in A if p["c1"] + p["c2"] == v]
    ax.scatter([p["pixel_ncc"] for p in q], [p["gradient_ncc"] for p in q], s=13 if v == 2 else 24, marker=VERD[v][2],
               c=VERD[v][1], edgecolors="white", linewidths=0.4, zorder=3 + (2 - v), label=f"{VERD[v][0]} ({len(q)})")
ax.axvline(0.95, color=C["line"], lw=0.8, ls=(0, (3, 2))); ax.axhline(0.90, color=C["line"], lw=0.8, ls=(0, (3, 2)))
ax.set_xlim(0.945, 1.003); ax.set_ylim(0.893, 1.006)
ax.set_xticks([0.95, 0.97, 0.99]); ax.set_yticks([0.90, 0.95, 1.00])
ax.set_xlabel("Pixel NCC after registration"); ax.set_ylabel("Gradient NCC"); grid(ax); grid(ax, "x")
ax.legend(loc="lower left", bbox_to_anchor=(-0.06, 1.03), ncol=1, handletextpad=0.2, labelspacing=0.25, borderaxespad=0)

# (b) registered displacement, by rotation
ax = fig.add_axes(R[1]); plab("(b)", 0.335)
rot0 = [p for p in A if p["rot_deg"] == 0]; rotq = [p for p in A if p["rot_deg"] != 0]
ax.scatter([p["dx"] for p in rot0], [p["dy"] for p in rot0], s=14, c=C["blue"], edgecolors="white", linewidths=0.4, zorder=3,
           label=f"Shift only ({len(rot0)})")
ax.scatter([p["dx"] for p in rotq], [p["dy"] for p in rotq], s=20, marker="D", c=C["teal"], edgecolors="white", linewidths=0.4,
           zorder=4, label=f"Quarter-turn + shift ({len(rotq)})")
ax.axhline(0, color=C["line"], lw=0.6); ax.axvline(0, color=C["line"], lw=0.6)
ax.set_xlim(-56, 56); ax.set_ylim(-56, 56)
ax.set_xticks([-50, -25, 0, 25, 50]); ax.set_yticks([-50, -25, 0, 25, 50])
ax.set_xlabel("Horizontal offset (px)"); ax.set_ylabel("Vertical offset (px)"); grid(ax); grid(ax, "x")
ax.legend(loc="lower left", bbox_to_anchor=(-0.06, 1.03), ncol=1, handletextpad=0.2, labelspacing=0.25, borderaxespad=0)

# (c) human verdict by candidate source
ax = fig.add_axes(R[2]); plab("(c)", 0.665)
pools = [("Registration\ngraph (223)", "Pool_A"), ("Below\nthreshold (26)", "Pool_B"), ("ResNet-50 /\nORB (100)", "Pool_C")]
for i, (lab, key) in enumerate(pools):
    q = [p for p in P if p["source_pool"].startswith(key)]
    cnt = collections.Counter(p["c1"] + p["c2"] for p in q); left = 0
    for v in (2, 1, 0):
        w = 100 * cnt[v] / len(q)
        ax.barh(i, w, left=left, height=0.6, color=VERD[v][1], edgecolor="white", linewidth=1.0, zorder=3)
        if w > 9:
            ax.text(left + w / 2, i, str(cnt[v]), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=5)
        left += w
ax.set_yticks(range(3)); ax.set_yticklabels([p[0] for p in pools], color=C["ink"]); ax.invert_yaxis()
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("Share of candidate pairs (%)")
h = [Rectangle((0, 0), 1, 1, fc=VERD[v][1], ec="none") for v in (2, 1, 0)]
ax.legend(h, ["Both confirm", "One confirms", "Both reject"], loc="lower left", bbox_to_anchor=(-0.02, 1.03), ncol=1,
          handlelength=1.0, handleheight=0.8, handletextpad=0.4, labelspacing=0.25, borderaxespad=0)
save(fig, "figB_duplicates")
