#!/usr/bin/env python3
"""
Step 2.2: Precision-Recall Curve Comparison Across Similarity Methods (Ours vs. Li 2026 Baseline)
Part of AGENT EXECUTION PLAYBOOK (Phase 2).
"""

import os
import argparse
import sys
import glob
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
HUMAN_AUDIT_DIR = os.path.join(RESULTS_DIR, "human_audit")
FIGURES_DIR = os.path.join(ROOT_DIR, "manuscript", "figures")

RAW_IMG_DIRS = [
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "IMAGES"),
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "Validation_Images"),
]

def load_image_map():
    img_map = {}
    for d in RAW_IMG_DIRS:
        for f in glob.glob(os.path.join(d, "*.jpg")):
            stem = os.path.splitext(os.path.basename(f))[0]
            img_map[stem] = f
            img_map[os.path.basename(f)] = f
    return img_map

def compute_li2026_thumbnail_feature(img_path):
    # Li (2026) code/analyze_group_structure.py lines 80-86
    buf = np.fromfile(img_path, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(im, (128, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t_norm = (t - t.mean()) / (t.std() + 1e-6)
    t_unit = t_norm / (np.linalg.norm(t_norm) + 1e-12)
    return t_unit

def select_evaluation_items(pool_items, labels, label_mode):
    """Never mix pool-derived proxy labels with independently supplied ratings."""
    ids = [item['audit_id'] for item in pool_items]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate audit IDs in candidate pool')
    if set(labels) - set(ids):
        raise ValueError('Human labels contain unknown audit IDs')
    if label_mode == 'proxy':
        return [(item, int(item['source_pool'] == 'Pool_A_Registration_High_NCC_ge_085'))
                for item in pool_items]
    selected = [(item, labels[item['audit_id']]) for item in pool_items
                if item['audit_id'] in labels]
    if len({label for _, label in selected}) < 2:
        raise ValueError('Human-only PR analysis requires both positive and negative ratings; no proxy fallback')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', choices=['proxy', 'human'], default='proxy',
                        help='Proxy diagnostic by default; human uses rated pairs only. Neither establishes method superiority on this selected pool.')
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    mapping_path = os.path.join(HUMAN_AUDIT_DIR, "internal_pool_mapping.json")
    user_labels_path = os.path.join(HUMAN_AUDIT_DIR, "pair_labels_to_fill.csv")

    with open(mapping_path, "r", encoding="utf-8") as f:
        pool_items = json.load(f)

    # Check if user has labeled pairs
    user_labels = {}
    if os.path.exists(user_labels_path):
        df_u = pd.read_csv(user_labels_path, dtype=str, keep_default_na=False)
        for _, row in df_u.iterrows():
            v = row.get('is_visually_overlapping', '').strip()
            if not v:
                continue
            if v not in ['0', '1']:
                raise ValueError(f'Invalid binary rating for {row["audit_id"]}: {v!r}')
            if row['audit_id'] in user_labels:
                raise ValueError(f'Duplicate human rating: {row["audit_id"]}')
            user_labels[row['audit_id']] = int(v)

    selected = select_evaluation_items(pool_items, user_labels, args.labels)

    print(f"Total pool candidate pairs: {len(pool_items)}")
    print(f"User human labels available: {len(user_labels)}")

    img_map = load_image_map()

    # Label source is explicit for the whole analysis: either pool-membership
    # proxy labels or human-rated pairs only. Pool membership is not gold truth.
    y_true = []
    method1_scores = []  # Li 2026: Unaligned 128x64 Thumbnail Correlation
    method2_scores = []  # Method 2 (Ablation): 2D Registered Pixel Intensity NCC only
    method3_scores = []  # Method 3 (Ours): 2D Registered Joint Pixel Intensity + Gradient Field NCC

    pair_eval_records = []

    for item, gt in selected:
        aid = item["audit_id"]
        a_file = item["image_a"]
        b_file = item["image_b"]
        path_a = img_map[a_file]
        path_b = img_map[b_file]

        # Labels are selected once per analysis, never replaced with proxy fallbacks.

        # Method 1: Li 2026 Unaligned 128x64 Thumbnail Correlation
        feat_a = compute_li2026_thumbnail_feature(path_a)
        feat_b = compute_li2026_thumbnail_feature(path_b)
        s_li = float(np.dot(feat_a, feat_b))

        # Method 2: 2D Registered Pixel Intensity NCC
        s_pixel = float(item.get("pixel_ncc", 0.0))

        # Method 3: 2D Registered Joint Intensity + Gradient NCC
        s_joint = min(float(item.get("pixel_ncc", 0.0)), float(item.get("gradient_ncc", 0.0)))

        y_true.append(gt)
        method1_scores.append(s_li)
        method2_scores.append(s_pixel)
        method3_scores.append(s_joint)

        pair_eval_records.append({
            "audit_id": aid,
            "image_a": a_file,
            "image_b": b_file,
            "ground_truth": gt,
            "li2026_thumbnail_corr": round(s_li, 4),
            "registered_pixel_ncc": round(s_pixel, 4),
            "registered_joint_ncc": round(s_joint, 4),
            "source_pool": item["source_pool"]
        })

    y_true = np.array(y_true)
    s1 = np.array(method1_scores)
    s2 = np.array(method2_scores)
    s3 = np.array(method3_scores)

    print(f"Ground-truth positive pairs: {int(y_true.sum())} / {len(y_true)}")

    # Compute PR Curves and Average Precision
    p1, r1, t1 = precision_recall_curve(y_true, s1)
    p2, r2, t2 = precision_recall_curve(y_true, s2)
    p3, r3, t3 = precision_recall_curve(y_true, s3)

    ap1 = average_precision_score(y_true, s1)
    ap2 = average_precision_score(y_true, s2)
    ap3 = average_precision_score(y_true, s3)

    print(f"Average Precision (AP):")
    print(f"  Method 1 (Li 2026 Baseline: Unaligned 128x64 Thumbnail): AP = {ap1:.4f}")
    print(f"  Method 2 (Ablation: Registered Pixel Intensity NCC):     AP = {ap2:.4f}")
    print(f"  Method 3 (Ours: Registered Joint Pixel + Gradient NCC):  AP = {ap3:.4f}")

    # Recall at fixed high precision (Precision >= 0.95)
    def recall_at_precision(p_arr, r_arr, target_p=0.95):
        valid = r_arr[p_arr >= target_p]
        return float(np.max(valid)) if len(valid) > 0 else 0.0

    r95_1 = recall_at_precision(p1, r1, 0.95)
    r95_2 = recall_at_precision(p2, r2, 0.95)
    r95_3 = recall_at_precision(p3, r3, 0.95)

    # Performance at default thresholds (0.99 for Li, 0.85 for Ours)
    pred_li_default = (s1 >= 0.99).astype(int)
    pred_ours_default = (s3 >= 0.85).astype(int)

    tp_li = int(np.sum((pred_li_default == 1) & (y_true == 1)))
    fp_li = int(np.sum((pred_li_default == 1) & (y_true == 0)))
    fn_li = int(np.sum((pred_li_default == 0) & (y_true == 1)))

    tp_ours = int(np.sum((pred_ours_default == 1) & (y_true == 1)))
    fp_ours = int(np.sum((pred_ours_default == 1) & (y_true == 0)))
    fn_ours = int(np.sum((pred_ours_default == 0) & (y_true == 1)))

    prec_li = tp_li / (tp_li + fp_li) if (tp_li + fp_li) > 0 else 0.0
    rec_li = tp_li / (tp_li + fn_li) if (tp_li + fn_li) > 0 else 0.0

    prec_ours = tp_ours / (tp_ours + fp_ours) if (tp_ours + fp_ours) > 0 else 0.0
    rec_ours = tp_ours / (tp_ours + fn_ours) if (tp_ours + fn_ours) > 0 else 0.0

    print(f"\nDefault Threshold Performance:")
    print(f"  Li 2026 (threshold=0.99): Detected={int(pred_li_default.sum())}, Prec={prec_li:.2%}, Rec={rec_li:.2%}")
    print(f"  Ours    (threshold=0.85): Detected={int(pred_ours_default.sum())}, Prec={prec_ours:.2%}, Rec={rec_ours:.2%}")

    # Output Summary JSON
    comparison_summary = {
        "evaluation_dataset": "NEU-DET Targeted Candidate Pool",
        "total_pairs_evaluated": len(y_true),
        "ground_truth_positives": int(y_true.sum()),
        "ground_truth_source": 'pool_membership_proxy' if args.labels == 'proxy' else 'human_ratings_only_not_necessarily_adjudicated',
        "label_mode": args.labels,
        "candidate_pool_size": len(pool_items),
        "human_labels_available": len(user_labels),
        "human_label_coverage": len(user_labels) / len(pool_items),
        "human_labels_used": 0 if args.labels == 'proxy' else len(selected),
        "proxy_labels_used": len(selected) if args.labels == 'proxy' else 0,
        "supports_method_superiority": False,
        "limitations": [
            'Pool selected by registered NCC and ResNet/ORB; lacks thumbnail-method top candidates and random controls.',
            'Proxy labels are derived from pool construction and are circular for NCC evaluation.',
            'Even human-only metrics describe this selected pool, not population recall or comparative method superiority.'
        ],
        "average_precision": {
            "method1_li2026_thumbnail_corr": round(float(ap1), 4),
            "method2_registered_pixel_ncc": round(float(ap2), 4),
            "method3_registered_joint_ncc_ours": round(float(ap3), 4)
        },
        "recall_at_precision_95pct": {
            "method1_li2026": round(r95_1, 4),
            "method2_pixel_ncc": round(r95_2, 4),
            "method3_joint_ncc_ours": round(r95_3, 4)
        },
        "default_threshold_metrics": {
            "li2026_threshold_099": {
                "threshold": 0.99,
                "detected_count": int(pred_li_default.sum()),
                "true_positives": tp_li,
                "false_positives": fp_li,
                "precision": round(prec_li, 4),
                "recall": round(rec_li, 4)
            },
            "ours_threshold_085": {
                "threshold": 0.85,
                "detected_count": int(pred_ours_default.sum()),
                "true_positives": tp_ours,
                "false_positives": fp_ours,
                "precision": round(prec_ours, 4),
                "recall": round(rec_ours, 4)
            }
        }
    }

    suffix = '' if args.labels == 'proxy' else '_human_only'
    out_json = os.path.join(RESULTS_DIR, f"step2_method_comparison{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Saved method comparison summary to: {out_json}")

    # Plot PR Curves
    plt.figure(figsize=(6.5, 5.2), dpi=300)
    plt.plot(r1, p1, label=f"Li (2026) Unaligned Thumbnail (AP = {ap1:.3f})", color="#d62728", linestyle="--", linewidth=2.0)
    plt.plot(r2, p2, label=f"2D Registered Pixel NCC (AP = {ap2:.3f})", color="#ff7f0e", linestyle="-.", linewidth=2.0)
    plt.plot(r3, p3, label=f"2D Registered Joint Intensity + Grad (Ours, AP = {ap3:.3f})", color="#1f77b4", linewidth=2.4)

    # Plot default operating points
    plt.scatter([rec_li], [prec_li], color="#d62728", s=80, marker="o", zorder=5, label=f"Li default (0.99): Rec={rec_li:.1%}")
    plt.scatter([rec_ours], [prec_ours], color="#1f77b4", s=100, marker="*", zorder=5, label=f"Ours default (0.85): Rec={rec_ours:.1%}")

    plt.xlabel("Recall (Detection Rate of Visually Correlated Pairs)", fontsize=10, fontweight="bold")
    plt.ylabel('Precision', fontsize=10, fontweight='bold')
    plt.title(('Pool-membership proxy diagnostic (circular NCC labels)' if args.labels == 'proxy'
               else 'Human-rated subset of a method-selected pool'), fontsize=10, fontweight='bold')
    plt.figtext(0.5, 0.01, 'Exploratory only: selected pool does not establish method superiority.', ha='center', fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=9, frameon=True)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig_path = os.path.join(FIGURES_DIR, f"fig_similarity_pr_curve{suffix}.pdf")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved PR curve figure to: {fig_path}")

    # Copy to results/
    import shutil
    shutil.copy(fig_path, os.path.join(RESULTS_DIR, f"fig_similarity_pr_curve{suffix}.pdf"))

if __name__ == "__main__":
    main()
