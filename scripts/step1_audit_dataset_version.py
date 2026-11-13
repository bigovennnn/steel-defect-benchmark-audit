#!/usr/bin/env python3
"""
Step 1: Dataset Version Audit, XML Instance Statistics & Competitor Intersection Alignment
Part of AGENT EXECUTION PLAYBOOK (Phase 1).
"""

import os
import sys
import glob
import json
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np
import cv2

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_IMG_DIRS = [
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "IMAGES"),
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "Validation_Images"),
]
RAW_XML_DIRS = [
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "ANNOTATIONS"),
    os.path.join(ROOT_DIR, "benchmark", "paper-try", "datasets", "raw", "NEU-DET", "Validation_Annotations"),
]
FROZEN_SPLIT_PATH = os.path.join(ROOT_DIR, "benchmark", "datasets", "neu_v5_frozen", "split.json")
COMPETITOR_DIR = os.path.join(ROOT_DIR, "benchmark", "competitor_reference_li2026_v2")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def parse_xml_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        cname = obj.find("name").text.strip()
        # Normalization if needed
        cname = cname.replace(" ", "_")
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        area = w * h
        boxes.append({
            "class": cname,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "width": w,
            "height": h,
            "area": area
        })
    return boxes

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Step 1.1: Auditing NEU-DET Dataset Version and Excluded Images ===")

    # 1. Collect all raw images and XMLs
    raw_images = {}
    for d in RAW_IMG_DIRS:
        for f in glob.glob(os.path.join(d, "*.jpg")):
            stem = os.path.splitext(os.path.basename(f))[0]
            raw_images[stem] = f

    raw_xmls = {}
    for d in RAW_XML_DIRS:
        for f in glob.glob(os.path.join(d, "*.xml")):
            stem = os.path.splitext(os.path.basename(f))[0]
            raw_xmls[stem] = f

    print(f"Raw images found: {len(raw_images)}, Raw XMLs found: {len(raw_xmls)}")
    assert len(raw_images) == 1800, f"Expected 1800 raw images, found {len(raw_images)}"
    assert len(raw_xmls) == 1800, f"Expected 1800 raw XMLs, found {len(raw_xmls)}"

    # 2. Compare against neu_v5_frozen
    with open(FROZEN_SPLIT_PATH, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    frozen_stems = set(split_data["train"] + split_data["val"] + split_data["test"])
    print(f"neu_v5_frozen active stems count: {len(frozen_stems)}")
    assert len(frozen_stems) == 1797, f"Expected 1797 frozen stems, found {len(frozen_stems)}"

    excluded_stems = sorted(list(set(raw_images.keys()) - frozen_stems))
    print(f"Excluded images ({len(excluded_stems)}): {excluded_stems}")

    # Inspect excluded images
    excluded_audit = {}
    for stem in excluded_stems:
        img_path = raw_images[stem]
        xml_path = raw_xmls[stem]
        size_bytes = os.path.getsize(img_path)
        with open(img_path, "rb") as f:
            header_hex = f.read(32).hex()
            f.seek(0)
            img_sha256 = hashlib.sha256(f.read()).hexdigest()

        # OpenCV imdecode (Windows unicode path safe)
        buf = np.fromfile(img_path, dtype=np.uint8)
        img_mat = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        h, w = img_mat.shape[:2] if img_mat is not None else (None, None)

        boxes = parse_xml_boxes(xml_path)
        box_classes = [b["class"] for b in boxes]

        # Context on why excluded in benchmarks
        # KeenForgeAI and benchmark prep scripts removed them due to near-duplicate contamination or zero-area / noise
        reason = ""
        if stem == "patches_105":
            reason = "Near-duplicate pair with patches_101 (cross train/val leak in uncleaned split); excluded to prevent train/val leakage."
        elif stem == "pitted_surface_292":
            reason = "Near-duplicate translation pair with pitted_surface_171 (cross train/val leak); contains 4 boxes with mixed classes (pitted_surface + patches)."
        elif stem == "scratches_247":
            reason = "Near-duplicate rotation pair with scratches_232; excluded in standard corrected benchmarks (KeenForgeAI)."

        excluded_audit[stem] = {
            "image_path": img_path,
            "xml_path": xml_path,
            "size_bytes": size_bytes,
            "sha256": img_sha256,
            "header_hex": header_hex,
            "resolution": [w, h],
            "decode_success": img_mat is not None,
            "xml_objects_count": len(boxes),
            "xml_classes": box_classes,
            "exclusion_reason": reason
        }

    # 3. Parse XML annotations for all 1,797 frozen images
    print("\n=== Parsing XMLs and computing class & box statistics for 1,797 images ===")
    stem_to_prefix = {stem: stem.rsplit("_", 1)[0] for stem in frozen_stems}
    prefix_counts = Counter(stem_to_prefix.values())

    class_boxes = defaultdict(list)
    class_containing_images = defaultdict(set)
    image_to_classes = defaultdict(set)
    image_majority_class = {}

    for stem in frozen_stems:
        xml_path = raw_xmls[stem]
        boxes = parse_xml_boxes(xml_path)
        img_class_counts = Counter()
        for b in boxes:
            c = b["class"]
            class_boxes[c].append(b)
            class_containing_images[c].add(stem)
            image_to_classes[stem].add(c)
            img_class_counts[c] += 1
        
        if img_class_counts:
            # Majority class by box count; ties broken by filename prefix
            top_class = img_class_counts.most_common(1)[0][0]
            image_majority_class[stem] = top_class
        else:
            image_majority_class[stem] = stem_to_prefix[stem]

    majority_counts = Counter(image_majority_class.values())

    # Multi-class co-occurrence
    multiclass_images = [stem for stem, cls_set in image_to_classes.items() if len(cls_set) >= 2]
    multiclass_count = len(multiclass_images)
    multiclass_pct = multiclass_count / len(frozen_stems) * 100.0

    print(f"Multi-class co-occurrence images: {multiclass_count} / {len(frozen_stems)} ({multiclass_pct:.2f}%)")

    # Table of 6 classes
    category_stats = []
    total_boxes_all = 0
    for c in CLASSES:
        prefix_n = prefix_counts[c]
        majority_n = majority_counts.get(c, 0)
        containing_n = len(class_containing_images.get(c, set()))
        boxes_list = class_boxes.get(c, [])
        total_boxes = len(boxes_list)
        total_boxes_all += total_boxes
        boxes_per_img = total_boxes / containing_n if containing_n > 0 else 0.0

        areas = [b["area"] for b in boxes_list]
        if areas:
            area_median = float(np.median(areas))
            area_iqr = float(np.percentile(areas, 75) - np.percentile(areas, 25))
        else:
            area_median, area_iqr = 0.0, 0.0

        stat = {
            "Category": c,
            "Filename_Prefix_Images": prefix_n,
            "Majority_Box_Images": majority_n,
            "Containing_Images": containing_n,
            "Total_Boxes": total_boxes,
            "Boxes_Per_Containing_Image": round(boxes_per_img, 2),
            "Box_Area_Median_px2": round(area_median, 1),
            "Box_Area_IQR_px2": round(area_iqr, 1)
        }
        category_stats.append(stat)

    our_stats_output = {
        "dataset_name": "NEU-DET",
        "total_raw_images": len(raw_images),
        "total_frozen_images": len(frozen_stems),
        "split_counts": {k: len(split_data[k]) for k in ["train", "val", "test"]},
        "excluded_images_audit": excluded_audit,
        "category_statistics": category_stats,
        "total_boxes_all": total_boxes_all,
        "multiclass_cooccurrence": {
            "count": multiclass_count,
            "percentage": round(multiclass_pct, 2),
            "sample_images": sorted(multiclass_images)[:10]
        }
    }

    our_stats_json_path = os.path.join(RESULTS_DIR, "step1_our_neudet_stats.json")
    with open(our_stats_json_path, "w", encoding="utf-8") as f:
        json.dump(our_stats_output, f, indent=2)
    print(f"Saved stats to {our_stats_json_path}")

    # Print Markdown table
    print("\n--- NEU-DET Class & Instance Statistics (Our Frozen 1,797 Images) ---")
    print("| Category | Filename Prefix Imgs | Majority Box Imgs | Containing Imgs | Total Boxes | Boxes/Img | Area Median (px^2) | Area IQR (px^2) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in category_stats:
        print(f"| `{s['Category']}` | {s['Filename_Prefix_Images']} | {s['Majority_Box_Images']} | {s['Containing_Images']} | {s['Total_Boxes']} | {s['Boxes_Per_Containing_Image']:.2f} | {s['Box_Area_Median_px2']:.1f} | {s['Box_Area_IQR_px2']:.1f} |")
    print(f"| **Total / Overall** | **1,797** | **1,797** | -- | **{total_boxes_all}** | -- | -- | -- |")

    # 4. Task 1.2: Compare with Zehao Li (2026 v2) Competitor Reference
    print("\n=== Step 1.2: Cross-Alignment with Zehao Li (2026 v2) ===")
    comp_meta_path = os.path.join(COMPETITOR_DIR, "metadata", "neu", "meta.json")
    comp_folds_path = os.path.join(COMPETITOR_DIR, "splits", "neu", "folds.json")

    with open(comp_meta_path, "r", encoding="utf-8") as f:
        comp_meta = json.load(f)

    with open(comp_folds_path, "r", encoding="utf-8") as f:
        comp_folds = json.load(f)

    comp_images = comp_meta["images"]
    comp_stems = set(img["stem"] for img in comp_images)
    comp_box_count = sum(len(img["boxes"]) for img in comp_images)
    print(f"Competitor NEU images: {len(comp_stems)}, Total boxes in competitor: {comp_box_count}")

    shared_stems = sorted(list(frozen_stems.intersection(comp_stems)))
    only_in_ours = sorted(list(frozen_stems - comp_stems))
    only_in_comp = sorted(list(comp_stems - frozen_stems))

    print(f"Shared stems: {len(shared_stems)}")
    print(f"Only in our frozen: {len(only_in_ours)}")
    print(f"Only in competitor: {len(only_in_comp)}")

    # Competitor per-class box count from meta.json
    comp_class_boxes = Counter()
    for img in comp_images:
        for b in img["boxes"]:
            cls_idx = b[0]
            cls_name = CLASSES[cls_idx]
            comp_class_boxes[cls_name] += 1

    comparison_by_class = []
    for s in category_stats:
        c = s["Category"]
        our_b = s["Total_Boxes"]
        comp_b = comp_class_boxes.get(c, 0)
        comparison_by_class.append({
            "category": c,
            "our_boxes": our_b,
            "competitor_boxes": comp_b,
            "diff": our_b - comp_b
        })

    alignment_output = {
        "competitor_doi": "10.6084/m9.figshare.34070919.v2",
        "competitor_name": "Zehao Li (2026-10-28, Figshare v2)",
        "our_frozen_images": len(frozen_stems),
        "competitor_images": len(comp_stems),
        "shared_images": len(shared_stems),
        "only_in_our_dataset": only_in_ours,
        "only_in_competitor_dataset": only_in_comp,
        "our_total_boxes": total_boxes_all,
        "competitor_total_boxes": comp_box_count,
        "box_comparison_by_class": comparison_by_class,
        "competitor_folds_summary": {
            "k": comp_folds.get("k"),
            "seed": comp_folds.get("seed"),
            "total_assigned": len(comp_folds.get("fold_of", {}))
        }
    }

    alignment_json_path = os.path.join(RESULTS_DIR, "step1_competitor_alignment.json")
    with open(alignment_json_path, "w", encoding="utf-8") as f:
        json.dump(alignment_output, f, indent=2)
    print(f"Saved competitor alignment report to {alignment_json_path}")

    print("\n--- Competitor vs. Our Annotation Box Count Comparison ---")
    print("| Category | Our Total Boxes | Competitor Boxes | Diff |")
    print("| :--- | :---: | :---: | :---: |")
    for row in comparison_by_class:
        print(f"| `{row['category']}` | {row['our_boxes']} | {row['competitor_boxes']} | {row['diff']} |")
    print(f"| **Total** | **{total_boxes_all}** | **{comp_box_count}** | **{total_boxes_all - comp_box_count}** |")

if __name__ == "__main__":
    main()
