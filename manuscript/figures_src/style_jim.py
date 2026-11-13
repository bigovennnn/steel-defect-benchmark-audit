"""Shared style for all figures (Springer / JIM-friendly).
Change FONT to 'Arial' or 'Helvetica' on your own machine if you prefer.
Widths: single column 84 mm, double column 174 mm."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

FONT = ["Arial", "Helvetica", "Inter", "DejaVu Sans"]
MM = 1 / 25.4
W2 = 174  # double-column width, mm
W1 = 84   # single-column width, mm

C = dict(
    navy="#1B3A5C", blue="#2F6DA8", blue_l="#EAF1F8", blue_m="#BCD3EA",
    teal="#2A9D8F", teal_l="#E7F5F3", teal_m="#B5E0DA",
    amber="#D98A1F", amber_l="#FCF3E3", amber_m="#F2D29C",
    red="#C8553D", red_l="#FAECE8",
    ink="#1F2933", mute="#5F6B7A", line="#9AA5B1", grid="#E4E7EB",
    bg="#F5F7FA", none="#DDE2E8", white="#FFFFFF",
)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6, "axes.edgecolor": C["line"], "axes.labelcolor": C["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": C["mute"], "ytick.color": C["mute"],
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8,
    "text.color": C["ink"], "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 600, "figure.dpi": 150, "hatch.linewidth": 0.5,
})


def canvas(w, h):
    """Blank drawing canvas whose data units are millimetres."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="#FFFFFF", ec=None, lw=0.7, r=1.6, z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec or C["line"], lw=lw, zorder=z, ls=ls)
    ax.add_patch(p); return p


def txt(ax, x, y, s, size=8, color=None, weight="normal", ha="center", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color or C["ink"], fontweight=weight,
                   ha=ha, va=va, zorder=z, **{"linespacing":1.02, **kw})


def arrow(ax, p0, p1, color=None, lw=0.9, conn="arc3,rad=0", head=5, z=3, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={head*0.55},head_width={head*0.32}",
                        mutation_scale=1.6, color=color or C["mute"], lw=lw,
                        connectionstyle=conn, zorder=z, shrinkA=0, shrinkB=0, ls=ls,
                        joinstyle="miter", capstyle="butt")
    ax.add_patch(a); return a


def badge(ax, x, y, s, color, r=2.3, size=7.5):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=6))
    txt(ax, x, y - 0.05, s, size=size, color="white", weight="bold", z=7)


def panel(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom", color=C["ink"])


import json, os
from pathlib import Path

_FIG_OUT = Path(__file__).resolve().parents[1] / "figures"
_STATS_PATH = Path(__file__).resolve().parent / "stats_jim.json"

def save(fig, name):
    _FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIG_OUT / f"{name}.pdf")
    fig.savefig(_FIG_OUT / f"{name}.png", dpi=300)
    plt.close(fig)

def load():
    with open(_STATS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=C["grid"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)

NICE = {"yolo11n": "YOLO11n", "yolov8n": "YOLOv8n", "yolo11s": "YOLO11s", "yolov8s": "YOLOv8s", "rtdetr-l": "RT-DETR-L"}
CLS = {"scratches": "Scratches", "patches": "Patches", "inclusion": "Inclusion", "pitted_surface": "Pitted surface",
       "rolled-in_scale": "Rolled-in scale", "crazing": "Crazing"}
