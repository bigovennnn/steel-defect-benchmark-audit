#!/usr/bin/env python3
"""
Generate formal PRISMA 2020 Flow Diagram PDF (manuscript/figures/fig_prisma_flow.pdf)
with exact reconciled counts across Identification, Screening, Retrieval (61 non-XML breakdown),
Eligibility, and Included cohorts.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = ROOT / "manuscript" / "figures" / "fig_prisma_flow.pdf"


def draw_box(ax, x, y, w, h, title, body, bg="#f8fafc", edge="#334155", title_color="#0f172a"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=bg, edgecolor=edge, linewidth=1.2
    )
    ax.add_patch(box)
    ax.text(
        x + 0.016, y + h - 0.024, title,
        fontsize=8.6, fontweight="bold", color=title_color,
        va="top", ha="left", family="sans-serif"
    )
    ax.text(
        x + 0.016, y + h - 0.056, body,
        fontsize=7.9, color="#1e293b",
        va="top", ha="left", family="sans-serif", linespacing=1.26
    )


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.35", lw=1.25, color="#334155")
    )


def main():
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Left stage banners
    stages = [
        ("Identification", 0.81, 0.15, "#e0f2fe", "#0369a1"),
        ("Screening", 0.61, 0.16, "#e0e7ff", "#3730a3"),
        ("Retrieval", 0.36, 0.21, "#fef3c7", "#92400e"),
        ("Eligibility", 0.19, 0.13, "#fce7f3", "#9d174d"),
        ("Included", 0.02, 0.13, "#dcfce7", "#166534"),
    ]
    for label, y, h, bg, fg in stages:
        b = FancyBboxPatch(
            (0.01, y), 0.11, h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=bg, edgecolor=fg, linewidth=1.1
        )
            # Add stage label
        ax.add_patch(b)
        ax.text(
            0.065, y + h / 2, label,
            fontsize=9.0, fontweight="bold", color=fg,
            va="center", ha="center", rotation=90 if h > 0.14 else 0
        )

    # Main column boxes (x=0.14, w=0.38)
    draw_box(
        ax, 0.14, 0.81, 0.38, 0.15,
        "Records identified from Europe PMC (n = 433)",
        "Search date: Oct 30, 2026 (2020-01-01 to 2026-10-30)\n"
        "• q1: steel AND defect AND yolo*\n"
        "• q2: steel AND defect AND (detect* OR inspection)\n"
        "• q3: \"NEU-DET\" OR \"GC10-DET\" (DOI/ID deduplicated)"
    )

    draw_box(
        ax, 0.14, 0.61, 0.38, 0.16,
        "Records screened by title & abstract (n = 433)",
        "Retained for full-text retrieval: n = 196\n"
        "• Include for full-text assessment: n = 154\n"
        "• Unclear at abstract level (seek full text): n = 42\n"
        "(137 records with PMCID; 59 without PMCID)"
    )

    draw_box(
        ax, 0.14, 0.36, 0.38, 0.21,
        "Full-text retrieval sought (n = 196)",
        "Programmatic Europe PMC JATS XML ingestion:\n"
        "• PMC XML download attempts: n = 137\n"
        "• PMC XML successfully retrieved: n = 135\n"
        "• Outside successful PMC XML set: n = 61\n"
        "  (47 with OA metadata links; 14 without OA links)"
    )

    draw_box(
        ax, 0.14, 0.19, 0.38, 0.13,
        "Full-text JATS XML assessed for eligibility (n = 135)",
        "Evaluated against Frozen Coding Manual v1.0–v1.4\n"
        "eligibility criteria (bounding-box object detection\n"
        "experiments on NEU-DET or GC10-DET)."
    )

    draw_box(
        ax, 0.14, 0.02, 0.38, 0.13,
        "Studies included in protocol audit (n = 131)",
        "• 131 peer-reviewed PMC journal articles (0 preprints)\n"
        "• 182 benchmark evaluations audited across F01–F17\n"
        "  (125 NEU-DET evaluations; 57 GC10-DET evaluations)",
        bg="#f0fdf4", edge="#15803d", title_color="#14532d"
    )

    # Right column exclusion / accounting boxes (x=0.57, w=0.41)
    draw_box(
        ax, 0.57, 0.61, 0.41, 0.16,
        "Records excluded at title/abstract stage (n = 237)",
        "Recorded screening exclusions:\n"
        "• Outside steel/metal surface defect scope: n = 16\n"
        "• Classification or segmentation only (no bbox): n = 50\n"
        "• No NEU-DET or GC10-DET experiment: n = 171",
        bg="#fffbeb", edge="#b45309", title_color="#78350f"
    )

    draw_box(
        ax, 0.57, 0.35, 0.41, 0.22,
        "Records outside PMC JATS XML set (n = 61)",
        "Among 47 records with OA metadata links:\n"
        "• Duplicate/superseded records (E4): n = 8 (2 duplicate MED\n"
        "  records + 3 preprints of included PMC papers; 3 other E4)\n"
        "• External preprints without PMC JATS XML (E5): n = 37\n"
        "• PMC XML API HTTP 500 retrieval failure (E5): n = 2\n"
        "Among 14 records without OA metadata links:\n"
        "• MEDLINE-only journal records without OA/XML (E5): n = 14",
        bg="#fffbeb", edge="#b45309", title_color="#78350f"
    )

    draw_box(
        ax, 0.57, 0.19, 0.41, 0.13,
        "Full-text articles excluded (n = 4)",
        "• Classification / segmentation only (E2): n = 2\n"
        "• No NEU-DET or GC10-DET detection experiment (E3): n = 2",
        bg="#fff1f2", edge="#be123c", title_color="#881337"
    )

    # Vertical arrows
    draw_arrow(ax, 0.33, 0.81, 0.33, 0.77)
    draw_arrow(ax, 0.33, 0.61, 0.33, 0.57)
    draw_arrow(ax, 0.33, 0.36, 0.33, 0.32)
    draw_arrow(ax, 0.33, 0.19, 0.33, 0.15)

    # Horizontal arrows
    draw_arrow(ax, 0.52, 0.69, 0.57, 0.69)
    draw_arrow(ax, 0.52, 0.46, 0.57, 0.46)
    draw_arrow(ax, 0.52, 0.255, 0.57, 0.255)

    plt.tight_layout(pad=0.2)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PRISMA 2020 flow diagram to {FIG_PATH}")


if __name__ == "__main__":
    main()
