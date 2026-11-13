#!/usr/bin/env python3
"""
Step 2.1: Build Verification Pool, Targeted False-Negative Audit & Generate Human Review Package
Part of AGENT EXECUTION PLAYBOOK (Phase 2).
"""

import os
import sys
import glob
import json
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDITS_DIR = os.path.join(ROOT_DIR, "benchmark", "audits")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
HUMAN_AUDIT_DIR = os.path.join(RESULTS_DIR, "human_audit")
IMAGES_DIR = os.path.join(HUMAN_AUDIT_DIR, "pair_images")

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

def apply_registration(img_b, rot_deg, dx, dy):
    # Rotate B counter-clockwise
    if rot_deg == 90:
        b_rot = cv2.rotate(img_b, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot_deg == 180:
        b_rot = cv2.rotate(img_b, cv2.ROTATE_180)
    elif rot_deg == 270:
        b_rot = cv2.rotate(img_b, cv2.ROTATE_90_CLOCKWISE)
    else:
        b_rot = img_b.copy()

    H, W = img_b.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    b_reg = cv2.warpAffine(b_rot, M, (W, H), borderValue=(0, 0, 0))
    mask_b = cv2.warpAffine(np.ones((H, W), dtype=np.uint8), M, (W, H), borderValue=0)
    return b_reg, (mask_b > 0)

def create_pair_composite(path_a, path_b, rot_deg=0, dx=0, dy=0):
    img_a = cv2.imdecode(np.fromfile(path_a, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.fromfile(path_b, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    H, W = img_a.shape[:2]
    b_reg, mask_overlap = apply_registration(img_b, rot_deg, dx, dy)

    # Compute difference heatmap
    diff = np.abs(img_a.astype(np.float32) - b_reg.astype(np.float32)).mean(axis=2)
    # Normalize diff for visualization (0 to 60 intensity range mapped to 0-255)
    diff_norm = np.clip(diff * 3.5, 0, 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    diff_color[~mask_overlap] = [30, 30, 30]

    # Draw border on images
    for im in [img_a, b_reg, diff_color]:
        cv2.rectangle(im, (0, 0), (W-1, H-1), (80, 80, 80), 1)

    # Canvas with header
    header_h = 36
    total_w = W * 3 + 20
    total_h = H + header_h + 10
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8) + 245

    # Paste images
    y_off = header_h
    canvas[y_off:y_off+H, 5:5+W] = img_a
    canvas[y_off:y_off+H, 10+W:10+2*W] = b_reg
    canvas[y_off:y_off+H, 15+2*W:15+3*W] = diff_color

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Image A (Ref)", (10, 24), font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Image B (Reg: r={rot_deg}, d=({dx},{dy}))", (15+W, 24), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Absolute Diff Heatmap", (20+2*W, 24), font, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

    return canvas

def main():
    os.makedirs(HUMAN_AUDIT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    img_map = load_image_map()
    print(f"Total mapped raw images: {len(img_map)}")

    # 1. Output Geometric Registration Definition
    geom_def = {
        "transformation_group": "SE(2) (Special Euclidean Group in 2D)",
        "degrees_of_freedom": 3,
        "parameters": {
            "rotation_theta": "Discrete quarter turns {0, 90, 180, 270} degrees CCW (rigid planar, no reflections/mirroring)",
            "translation_dx_dy": "Continuous 2D translation (dx, dy) within [-50, +50] pixels (25% of image width/height)",
            "min_overlap_ratio_alpha_min": 0.60
        },
        "search_stages": {
            "coarse": "16x16 FFT phase correlation on downsampled representations across 4 rotation angles",
            "refine": "64x64 sub-pixel phase correlation with Hanning windowing",
            "verification": "Original 200x200 resolution joint intensity NCC (>= 0.85) and Sobel gradient field NCC (>= 0.85)"
        }
    }
    with open(os.path.join(RESULTS_DIR, "step2_geometric_registration_def.json"), "w", encoding="utf-8") as f:
        json.dump(geom_def, f, indent=2)

    # 2. Collect Candidate Pools
    print("\n--- Constructing Candidate Pools ---")
    
    # Pool A: 223 candidate edges
    pool_a_csv = os.path.join(AUDITS_DIR, "neu_candidate_edges_223_20261101.csv")
    df_pool_a = pd.read_csv(pool_a_csv)
    print(f"Pool A (Primary High-NCC >= 0.85): {len(df_pool_a)} pairs")
    
    pool_a_pairs = []
    seen_pair_keys = set()
    for _, row in df_pool_a.iterrows():
        a = row["a_file"]
        b = row["b_file"]
        pair_key = tuple(sorted([a, b]))
        seen_pair_keys.add(pair_key)
        pool_a_pairs.append({
            "image_a": a,
            "image_b": b,
            "rot_deg": int(row["rotation_b_ccw_degrees"]),
            "dx": int(row["shift_b_dx_pixels"]),
            "dy": int(row["shift_b_dy_pixels"]),
            "pixel_ncc": float(row["pixel_ncc"]),
            "gradient_ncc": float(row["gradient_ncc"]),
            "source_pool": "Pool_A_Registration_High_NCC_ge_085"
        })

    # Pool B: Registration threshold boundary [0.70, 0.85)
    # Load candidate_audit.json
    cand_audit_json = os.path.join(AUDITS_DIR, "neu_clean_v2_rigid_20261031", "candidate_audit.json")
    pool_b_candidates = []
    if os.path.exists(cand_audit_json):
        with open(cand_audit_json, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        for item in cdata.get("all_intermediate_passing_pairs_with_verification", []):
            a_file = item["a"]["file"]
            b_file = item["b"]["file"]
            pkey = tuple(sorted([a_file, b_file]))
            if pkey in seen_pair_keys:
                continue
            p_ncc = item.get("pixel_ncc", 0.0)
            g_ncc = item.get("gradient_ncc", 0.0)
            min_ncc = min(p_ncc, g_ncc)
            if 0.70 <= min_ncc < 0.85:
                pool_b_candidates.append({
                    "image_a": a_file,
                    "image_b": b_file,
                    "rot_deg": int(item.get("rotation_b_ccw_degrees", 0)),
                    "dx": int(item.get("shift_b_xy_pixels", [0, 0])[0]),
                    "dy": int(item.get("shift_b_xy_pixels", [0, 0])[1]),
                    "pixel_ncc": float(p_ncc),
                    "gradient_ncc": float(g_ncc),
                    "min_ncc": min_ncc,
                    "source_pool": "Pool_B_Registration_Boundary_070_085"
                })

    print(f"Pool B raw eligible boundary pairs: {len(pool_b_candidates)}")
    # Sample 100 stratified pairs
    if len(pool_b_candidates) > 100:
        # Stratify into [0.70, 0.75), [0.75, 0.80), [0.80, 0.85)
        bin1 = [p for p in pool_b_candidates if 0.70 <= p["min_ncc"] < 0.75]
        bin2 = [p for p in pool_b_candidates if 0.75 <= p["min_ncc"] < 0.80]
        bin3 = [p for p in pool_b_candidates if 0.80 <= p["min_ncc"] < 0.85]
        random.shuffle(bin1)
        random.shuffle(bin2)
        random.shuffle(bin3)
        sample_b = bin1[:33] + bin2[:33] + bin3[:34]
        # Fill if needed
        if len(sample_b) < 100:
            remaining = [p for p in pool_b_candidates if p not in sample_b]
            random.shuffle(remaining)
            sample_b.extend(remaining[:100 - len(sample_b)])
    else:
        sample_b = pool_b_candidates

    for p in sample_b:
        seen_pair_keys.add(tuple(sorted([p["image_a"], p["image_b"]])))
    print(f"Pool B sampled: {len(sample_b)} pairs")

    # Pool C: Independent Feature Retrieval (ResNet-50 Cosine + ORB Inliers)
    print("\n--- Computing Pool C Independent Features (ResNet-50 & ORB) ---")
    split_path = os.path.join(ROOT_DIR, "benchmark", "datasets", "neu_v5_frozen", "split.json")
    with open(split_path, "r", encoding="utf-8") as f:
        sdata = json.load(f)
    frozen_stems = sorted(list(set(sdata["train"] + sdata["val"] + sdata["test"])))
    print(f"Total frozen stems for embedding: {len(frozen_stems)}")

    # 1. ResNet-50 Feature Extraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for ResNet-50 extraction")
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights).to(device)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    preprocess = weights.transforms()
    feats = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(frozen_stems), batch_size):
            batch_stems = frozen_stems[i:i+batch_size]
            batch_tensors = []
            for stem in batch_stems:
                p = img_map[stem]
                img_pil = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img_pil))
            inp = torch.stack(batch_tensors).to(device)
            out = resnet(inp)
            feats.append(out.cpu().numpy())
    
    feats = np.concatenate(feats, axis=0)
    # L2 normalize
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    
    # Cosine similarity matrix
    sim_mat = np.dot(feats, feats.T)
    # Zero diagonal
    np.fill_diagonal(sim_mat, -1.0)

    # Find top 50 pairs from ResNet cosine outside seen_pair_keys
    triu_i, triu_j = np.triu_indices(len(frozen_stems), k=1)
    sim_scores = sim_mat[triu_i, triu_j]
    sorted_order = np.argsort(sim_scores)[::-1]

    resnet_top_pairs = []
    for idx in sorted_order:
        i, j = triu_i[idx], triu_j[idx]
        stem_a, stem_b = frozen_stems[i], frozen_stems[j]
        file_a, file_b = f"{stem_a}.jpg", f"{stem_b}.jpg"
        pkey = tuple(sorted([file_a, file_b]))
        if pkey not in seen_pair_keys:
            seen_pair_keys.add(pkey)
            resnet_top_pairs.append({
                "image_a": file_a,
                "image_b": file_b,
                "rot_deg": 0,
                "dx": 0,
                "dy": 0,
                "pixel_ncc": 0.0,
                "gradient_ncc": 0.0,
                "cosine_sim": float(sim_scores[idx]),
                "source_pool": "Pool_C_ResNet50_Cosine_Top50"
            })
            if len(resnet_top_pairs) >= 50:
                break
    print(f"Pool C (ResNet-50 Cosine Top 50) extracted: {len(resnet_top_pairs)} pairs")

    # 2. ORB Keypoints + RANSAC Inliers
    print("Extracting ORB keypoints & computing RANSAC inliers for intra-class candidates...")
    orb = cv2.ORB_create(nfeatures=500)
    img_descs = {}
    img_kps = {}
    
    for stem in frozen_stems:
        p = img_map[stem]
        buf = np.fromfile(p, dtype=np.uint8)
        gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        kps, des = orb.detectAndCompute(gray, None)
        img_kps[stem] = kps
        img_descs[stem] = des

    # Group stems by class to search for intra-class keypoint matches
    class_stems = defaultdict(list)
    for stem in frozen_stems:
        c = stem.rsplit("_", 1)[0]
        class_stems[c].append(stem)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    orb_candidates = []

    for c, stems in class_stems.items():
        for i in range(len(stems)):
            stem_a = stems[i]
            des_a = img_descs[stem_a]
            if des_a is None or len(des_a) < 10:
                continue
            for j in range(i + 1, len(stems)):
                stem_b = stems[j]
                file_a, file_b = f"{stem_a}.jpg", f"{stem_b}.jpg"
                pkey = tuple(sorted([file_a, file_b]))
                if pkey in seen_pair_keys:
                    continue
                des_b = img_descs[stem_b]
                if des_b is None or len(des_b) < 10:
                    continue
                
                # KNN match
                matches = bf.knnMatch(des_a, des_b, k=2)
                good = []
                for m_n in matches:
                    if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                        good.append(m_n[0])
                
                if len(good) >= 12:
                    # Estimate Affine or Homography with RANSAC
                    src_pts = np.float32([img_kps[stem_a][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([img_kps[stem_b][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    _, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
                    if inlier_count >= 8:
                        orb_candidates.append({
                            "image_a": file_a,
                            "image_b": file_b,
                            "inlier_count": inlier_count,
                            "rot_deg": 0,
                            "dx": 0,
                            "dy": 0,
                            "pixel_ncc": 0.0,
                            "gradient_ncc": 0.0,
                            "source_pool": "Pool_C_ORB_RANSAC_Top50"
                        })

    orb_candidates.sort(key=lambda x: x["inlier_count"], reverse=True)
    orb_top_pairs = []
    for cand in orb_candidates:
        pkey = tuple(sorted([cand["image_a"], cand["image_b"]]))
        if pkey not in seen_pair_keys:
            seen_pair_keys.add(pkey)
            orb_top_pairs.append(cand)
            if len(orb_top_pairs) >= 50:
                break

    print(f"Pool C (ORB RANSAC Inliers Top 50) extracted: {len(orb_top_pairs)} pairs")

    # Combine all candidate pairs
    all_pairs = pool_a_pairs + sample_b + resnet_top_pairs + orb_top_pairs
    print(f"\nTotal combined verification pool: {len(all_pairs)} pairs")
    
    # Shuffle randomly with fixed seed for blind audit
    random.shuffle(all_pairs)
    
    # Assign blind IDs: AUDIT_001 ...
    audit_sheet_rows = []
    internal_mapping = []

    print("\nGenerating composite visualization images...")
    for idx, pair in enumerate(all_pairs):
        audit_id = f"AUDIT_{idx+1:04d}"
        path_a = img_map[pair["image_a"]]
        path_b = img_map[pair["image_b"]]
        rot = pair["rot_deg"]
        dx = pair["dx"]
        dy = pair["dy"]

        # Generate composite image
        composite = create_pair_composite(path_a, path_b, rot, dx, dy)
        img_out_path = os.path.join(IMAGES_DIR, f"{audit_id}.png")
        cv2.imencode('.png', composite)[1].tofile(img_out_path)

        # Blind sheet row
        audit_sheet_rows.append({
            "audit_id": audit_id,
            "image_a": pair["image_a"],
            "image_b": pair["image_b"],
            "is_visually_overlapping": "",  # 1 = True Overlap, 0 = Distinct
            "annotator_notes": ""
        })

        # Internal mapping
        internal_mapping.append({
            "audit_id": audit_id,
            "image_a": pair["image_a"],
            "image_b": pair["image_b"],
            "rot_deg": rot,
            "dx": dx,
            "dy": dy,
            "pixel_ncc": pair.get("pixel_ncc", 0.0),
            "gradient_ncc": pair.get("gradient_ncc", 0.0),
            "source_pool": pair["source_pool"]
        })

    # Save CSV for human annotation
    sheet_csv_path = os.path.join(HUMAN_AUDIT_DIR, "pair_labels_to_fill.csv")
    pd.DataFrame(audit_sheet_rows).to_csv(sheet_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved blank annotation sheet: {sheet_csv_path}")

    # Save internal mapping
    mapping_json_path = os.path.join(HUMAN_AUDIT_DIR, "internal_pool_mapping.json")
    with open(mapping_json_path, "w", encoding="utf-8") as f:
        json.dump(internal_mapping, f, indent=2)
    print(f"Saved internal pool mapping: {mapping_json_path}")

    # 4. Generate Interactive HTML Review App
    print("Generating interactive HTML review application...")
    html_content = generate_html_review_app(audit_sheet_rows)
    html_path = os.path.join(HUMAN_AUDIT_DIR, "duplicate_pair_review.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated interactive review application: {html_path}")

def generate_html_review_app(items):
    items_json = json.dumps(items)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NEU-DET Visual Near-Duplicate Pair Blind Review</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; background: #1e293b; padding: 15px 24px; border-radius: 8px; margin-bottom: 20px; }}
  .header h1 {{ margin: 0; font-size: 20px; font-weight: 600; color: #38bdf8; }}
  .progress-bar {{ background: #334155; height: 10px; border-radius: 5px; width: 300px; overflow: hidden; }}
  .progress-fill {{ background: #10b981; height: 100%; width: 0%; transition: width 0.2s; }}
  .controls {{ display: flex; gap: 12px; align-items: center; }}
  .btn {{ padding: 8px 16px; border-radius: 6px; border: none; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.2s; }}
  .btn-primary {{ background: #38bdf8; color: #0f172a; }}
  .btn-primary:hover {{ background: #0ea5e9; }}
  .btn-success {{ background: #10b981; color: white; }}
  .btn-success:hover {{ background: #059669; }}
  .btn-danger {{ background: #ef4444; color: white; }}
  .btn-danger:hover {{ background: #dc2626; }}
  .btn-secondary {{ background: #475569; color: white; }}
  .btn-secondary:hover {{ background: #334155; }}
  .viewer {{ background: #1e293b; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
  .pair-meta {{ display: flex; gap: 24px; margin-bottom: 16px; font-size: 15px; color: #cbd5e1; }}
  .pair-meta span strong {{ color: #f1f5f9; }}
  .image-container {{ position: relative; margin-bottom: 20px; max-width: 100%; overflow: auto; background: #020617; padding: 12px; border-radius: 8px; border: 1px solid #334155; }}
  .image-container img {{ max-width: 100%; height: auto; display: block; }}
  .action-bar {{ display: flex; gap: 20px; align-items: center; margin-top: 10px; }}
  .vote-btn {{ padding: 14px 28px; font-size: 16px; font-weight: 600; border-radius: 8px; border: none; cursor: pointer; display: flex; align-items: center; gap: 8px; }}
  .vote-1 {{ background: #059669; color: white; }}
  .vote-1:hover {{ background: #10b981; }}
  .vote-1.active {{ outline: 4px solid #34d399; }}
  .vote-0 {{ background: #dc2626; color: white; }}
  .vote-0:hover {{ background: #ef4444; }}
  .vote-0.active {{ outline: 4px solid #f87171; }}
  .shortcuts {{ margin-top: 20px; color: #94a3b8; font-size: 13px; text-align: center; }}
  .kbd {{ background: #334155; color: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-weight: 600; }}
  .notes-input {{ width: 450px; padding: 10px 14px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: white; font-size: 14px; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>NEU-DET Visual Near-Duplicate Pair Blind Review</h1>
    <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">Targeted Multi-Proposal Precision & Recall Audit (Blind Mode)</div>
  </div>
  <div class="controls">
    <div style="font-size: 14px;">Progress: <strong id="progress-text">0 / {len(items)}</strong></div>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <button class="btn btn-secondary" onclick="prevPair()">Previous (K / Left)</button>
    <button class="btn btn-secondary" onclick="nextPair()">Next (J / Right)</button>
    <button class="btn btn-danger" onclick="resetProgress()">Reset Progress</button>
    <button class="btn btn-primary" onclick="exportCSV()">Export CSV</button>
  </div>
</div>

<div class="viewer">
  <div class="pair-meta">
    <span>Audit ID: <strong id="pair-id">--</strong></span>
    <span>Image A: <strong id="img-a">--</strong></span>
    <span>Image B: <strong id="img-b">--</strong></span>
  </div>

  <div class="image-container">
    <img id="pair-img" src="" alt="Pair Composite">
  </div>

  <div class="action-bar">
    <button class="vote-btn vote-1" id="btn-vote-1" onclick="vote(1)">
      <span>[1] True Overlapping Pair</span>
    </button>
    <button class="vote-btn vote-0" id="btn-vote-0" onclick="vote(0)">
      <span>[0] Distinct / False Positive</span>
    </button>
    <input type="text" class="notes-input" id="pair-notes" placeholder="Optional notes / rationale..." onchange="saveNotes()">
  </div>

  <div class="shortcuts">
    Shortcuts: <span class="kbd">1</span> = True Overlap | <span class="kbd">0</span> = Distinct | <span class="kbd">J</span> / <span class="kbd">Right</span> = Next | <span class="kbd">K</span> / <span class="kbd">Left</span> = Previous
  </div>
</div>

<script>
const items = {items_json};
let currentIndex = 0;
let userVotes = JSON.parse(localStorage.getItem('neu_duplicate_votes') || '{{}}');
let userNotes = JSON.parse(localStorage.getItem('neu_duplicate_notes') || '{{}}');

function updateUI() {{
  const cur = items[currentIndex];
  document.getElementById('pair-id').textContent = cur.audit_id;
  document.getElementById('img-a').textContent = cur.image_a;
  document.getElementById('img-b').textContent = cur.image_b;
  document.getElementById('pair-img').src = 'pair_images/' + cur.audit_id + '.png';
  document.getElementById('pair-notes').value = userNotes[cur.audit_id] || '';

  const v = userVotes[cur.audit_id];
  document.getElementById('btn-vote-1').classList.toggle('active', v === 1);
  document.getElementById('btn-vote-0').classList.toggle('active', v === 0);

  const votedCount = Object.keys(userVotes).length;
  document.getElementById('progress-text').textContent = votedCount + ' / ' + items.length;
  document.getElementById('progress-fill').style.width = (votedCount / items.length * 100) + '%';
}}

function vote(val) {{
  const cur = items[currentIndex];
  userVotes[cur.audit_id] = val;
  localStorage.setItem('neu_duplicate_votes', JSON.stringify(userVotes));
  updateUI();
  setTimeout(() => {{
    if (currentIndex < items.length - 1) {{
      currentIndex++;
      updateUI();
    }}
  }}, 150);
}}

function saveNotes() {{
  const cur = items[currentIndex];
  userNotes[cur.audit_id] = document.getElementById('pair-notes').value;
  localStorage.setItem('neu_duplicate_notes', JSON.stringify(userNotes));
}}

function nextPair() {{
  if (currentIndex < items.length - 1) {{
    currentIndex++;
    updateUI();
  }}
}}

function prevPair() {{
  if (currentIndex > 0) {{
    currentIndex--;
    updateUI();
  }}
}}

function exportCSV() {{
  let csv = "audit_id,image_a,image_b,is_visually_overlapping,annotator_notes\\n";
  for (const it of items) {{
    const v = userVotes[it.audit_id] !== undefined ? userVotes[it.audit_id] : "";
    const n = (userNotes[it.audit_id] || "").replace(/"/g, '""');
    csv += `${{it.audit_id}},${{it.image_a}},${{it.image_b}},${{v}},"${{n}}"\\n`;
  }}
  const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'pair_labels_to_fill.csv';
  a.click();
}}

function resetProgress() {{
  const votedCount = Object.keys(userVotes).length;
  const noteCount = Object.keys(userNotes).filter(k => (userNotes[k] || '').trim()).length;
  if (votedCount === 0 && noteCount === 0) {{
    alert('当前没有任何已保存的标注进度，无需重置。');
    return;
  }}
  if (!confirm(`确定要清空全部已有标注进度吗？（当前已标注 ${{votedCount}} / ${{items.length}} 对，清空后不可恢复）`)) {{
    return;
  }}
  userVotes = {{}};
  userNotes = {{}};
  localStorage.removeItem('neu_duplicate_votes');
  localStorage.removeItem('neu_duplicate_notes');
  currentIndex = 0;
  updateUI();
  alert('已清空全部标注进度！');
}}

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') vote(1);
  if (e.key === '0') vote(0);
  if (e.key === 'ArrowRight' || e.key === 'j') nextPair();
  if (e.key === 'ArrowLeft' || e.key === 'k') prevPair();
}});

updateUI();
</script>
</body>
</html>
"""
    return html

if __name__ == "__main__":
    main()
