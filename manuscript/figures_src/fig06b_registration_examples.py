"""Fig. 6b: Pixel-verified SE(2) near-duplicate registration examples from NEU-DET."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.patches as patches
from style_jim import *

IMG_ROOT = Path(r"benchmark\datasets\neu_grouped_v6\images")
IMG_MAP = {p.name: p for p in IMG_ROOT.rglob("*.jpg")}


def load_gray(name: str) -> np.ndarray:
    return np.asarray(Image.open(IMG_MAP[name]).convert("L"), dtype=np.float64)


def best_se2_alignment(img_a: np.ndarray, img_b0: np.ndarray, min_overlap: float = 0.60) -> dict:
    H_img, W_img = img_a.shape
    best = {"ncc": -2.0}
    for k in range(4):
        b = np.rot90(img_b0, k)
        for dy in range(-50, 51):
            y1a, y2a = max(0, dy), min(H_img, H_img + dy)
            y1b, y2b = y1a - dy, y2a - dy
            for dx in range(-50, 51):
                x1a, x2a = max(0, dx), min(W_img, W_img + dx)
                x1b, x2b = x1a - dx, x2a - dx
                area = (y2a - y1a) * (x2a - x1a)
                if area < min_overlap * H_img * W_img:
                    continue
                pa = img_a[y1a:y2a, x1a:x2a]
                pb = b[y1b:y2b, x1b:x2b]
                pa0 = pa - pa.mean()
                pb0 = pb - pb.mean()
                denom = np.sqrt(np.sum(pa0 * pa0) * np.sum(pb0 * pb0))
                ncc = float(np.sum(pa0 * pb0) / denom) if denom > 0 else 0.0
                if ncc > best["ncc"]:
                    b_on_a = np.full_like(img_a, np.nan, dtype=np.float64)
                    b_on_a[y1a:y2a, x1a:x2a] = pb
                    diff = np.full_like(img_a, np.nan, dtype=np.float64)
                    diff[y1a:y2a, x1a:x2a] = np.abs(pa - pb)
                    best = {
                        "ncc": ncc,
                        "rot_deg": k * 90,
                        "dx": dx,
                        "dy": dy,
                        "overlap": area / float(H_img * W_img),
                        "mae": float(np.mean(np.abs(pa - pb))),
                        "box_a": (x1a, y1a, x2a - x1a, y2a - y1a),
                        "box_b_rot": (x1b, y1b, x2b - x1b, y2b - y1b),
                        "b_rot": b,
                        "b_on_a": b_on_a,
                        "diff": diff,
                    }
    return best


pairs = [
    ("Pair 1 (0° crop)", "scratches_254.jpg", "scratches_249.jpg"),
    ("Pair 2 (90° rot.)", "scratches_157.jpg", "scratches_279.jpg"),
]

W, H = W2, 95
fig = plt.figure(figsize=(W * MM, H * MM))

col_x = [0.165, 0.357, 0.549, 0.741]
tile_w = 0.172                        # 0.172 * 174 = 29.93 mm
tile_h = tile_w * W / H               # exact square in mm
row_y = [0.60, 0.21]

col_xlabels = [
    "Image A (200×200)\nwith overlap Ω",
    "Image B (200×200)\naligned / rotated",
    "Composite overlay\n(A red / B cyan)",
    "Pixel residual\non Ω (ADU)",
]

last_im = None
for row_idx, (row_label, name_a, name_b) in enumerate(pairs):
    img_a = load_gray(name_a)
    img_b0 = load_gray(name_b)
    reg = best_se2_alignment(img_a, img_b0)
    assert reg["ncc"] >= 0.99, f"Unexpected low NCC for {name_a}/{name_b}: {reg['ncc']}"

    y_bot = row_y[row_idx]
    row_mid_y = y_bot + tile_h / 2
    fig.text(
        0.008,
        row_mid_y,
        f"{row_label}\nNCC = {reg['ncc']:.4f}\nΩ = {reg['overlap']*100:.0f}%",
        ha="left",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=C["navy"],
        linespacing=1.25,
    )

    axes_row = []
    for c_idx in range(4):
        ax = fig.add_axes([col_x[c_idx], y_bot, tile_w, tile_h])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(C["line"])
            s.set_linewidth(0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        axes_row.append(ax)

    ax1, ax2, ax3, ax4 = axes_row

    # Col 1: Image A with overlap box Omega
    ax1.imshow(img_a, cmap="gray", vmin=0, vmax=255)
    xa, ya, wa, ha = reg["box_a"]
    ax1.add_patch(patches.Rectangle((xa, ya), wa - 1, ha - 1, lw=1.3, ec=C["teal"], fc="none", ls="--"))
    cap1 = name_a.replace(".jpg", "")

    # Col 2: Image B (rotated if rot != 0)
    ax2.imshow(reg["b_rot"], cmap="gray", vmin=0, vmax=255)
    xb, yb, wb, hb = reg["box_b_rot"]
    ax2.add_patch(patches.Rectangle((xb, yb), wb - 1, hb - 1, lw=1.3, ec=C["amber"], fc="none", ls="--"))
    cap2 = name_b.replace(".jpg", "") + (f" ({reg['rot_deg']}°)" if reg["rot_deg"] != 0 else "")

    # Col 3: False-color RGB alignment overlay (R = A, G = B_reg, B = B_reg)
    b_fill = np.where(np.isnan(reg["b_on_a"]), 0.0, reg["b_on_a"])
    rgb = np.stack([img_a / 255.0, b_fill / 255.0, b_fill / 255.0], axis=-1)
    ax3.imshow(np.clip(rgb, 0.0, 1.0))
    ax3.add_patch(patches.Rectangle((xa, ya), wa - 1, ha - 1, lw=1.1, ec="white", fc="none", ls=":"))
    cap3 = f"Δ = ({reg['dx']}, {reg['dy']}) px"

    # Col 4: Absolute residual |I_A - I_B| inside Omega
    cmap_res = plt.get_cmap("magma").copy()
    cmap_res.set_bad(color="#E8EEF3")
    last_im = ax4.imshow(reg["diff"], cmap=cmap_res, vmin=0.0, vmax=12.0)
    cap4 = f"MAE = {reg['mae']:.3f} / 255"

    caps = [(ax1, cap1, C["ink"], "normal"), (ax2, cap2, C["ink"], "normal"),
            (ax3, cap3, C["ink"], "normal"), (ax4, cap4, C["teal"], "bold")]
    for ax_i, cap_txt, col, wt in caps:
        ax_i.text(0.5, -0.055, cap_txt, transform=ax_i.transAxes, ha="center", va="top",
                  fontsize=8.0, fontweight=wt, color=col)

    if row_idx == 1:
        for ax_i, xlab in zip(axes_row, col_xlabels):
            ax_i.text(0.5, -0.22, xlab, transform=ax_i.transAxes, ha="center", va="top",
                      fontsize=8.2, fontweight="bold", color=C["navy"], linespacing=1.12)

cax = fig.add_axes([0.928, row_y[1], 0.015, (row_y[0] + tile_h) - row_y[1]])
cb = fig.colorbar(last_im, cax=cax)
cb.outline.set_edgecolor(C["line"])
cb.outline.set_linewidth(0.6)
cb.set_label("|I_A − I_B| (ADU, 0–255)", fontsize=8.0, color=C["ink"], labelpad=4)
cb.ax.tick_params(labelsize=8.0, color=C["mute"], labelcolor=C["ink"])

save(fig, "fig06b_registration_examples")
print("Saved fig06b_registration_examples.pdf and .png")
