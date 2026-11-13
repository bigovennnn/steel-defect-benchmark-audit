"""Read-only NEU candidate audit for quarter-turn rotations and translations.

Every unordered image pair is screened, regardless of split or class, on a
16x16 area-resized grid. Zero-padded FFT correlation and rectangle sums give
overlap-only normalized cross correlation (NCC) for every allowed integer grid
translation and each quarter turn. Passing pairs receive original-resolution
phase-correlation checks and pixel/gradient evidence. These are candidates,
not duplicate adjudications or evidence that leakage has been excluded.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time

import cv2
import numpy as np
import scipy
from scipy.fft import irfft2, rfft2


CAUTION = (
    "Candidate evidence only; no images, annotations or partitions are changed. "
    "Only quarter-turn rotations and bounded translations are screened. "
    "Coarse resampling, thresholds and phase estimation can miss true pairs. "
    "Similar texture can produce false positives. No model/test performance is read. "
    "A negative result does not establish absence of duplicates or capture-group leakage."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pool(root: Path):
    split_path = root / "split.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    records, images, seen = [], [], set()
    for part in ("train", "val", "test"):
        listed = sorted(splits[part])
        actual = {p.stem for p in (root / "images" / part).glob("*.jpg")}
        if len(listed) != len(set(listed)) or set(listed) != actual:
            raise ValueError(f"Manifest/file mismatch in {part}")
        for stem in listed:
            if stem in seen:
                raise ValueError(f"Overlapping image name: {stem}")
            seen.add(stem)
            path = root / "images" / part / f"{stem}.jpg"
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (200, 200):
                raise ValueError(f"This NEU-specific audit expects 200x200 images: {path}")
            records.append({"split": part, "file": path.name, "sha256": sha256(path)})
            images.append(image)
    return records, images, sha256(split_path)


def grid_geometry(size, max_fraction, min_overlap):
    bound = int(np.floor(size * max_fraction))
    shifts = [(dx, dy) for dy in range(-bound, bound + 1)
              for dx in range(-bound, bound + 1)
              if (size - abs(dx)) * (size - abs(dy)) / size ** 2 >= min_overlap]
    counts = np.array([(size - abs(dx)) * (size - abs(dy)) for dx, dy in shifts], np.float32)
    reverse = np.array([shifts.index((-dx, -dy)) for dx, dy in shifts])
    return shifts, counts, reverse


def rectangles(images, shifts):
    size = images.shape[-1]
    sums, energies = [], []
    # Values are centered per image before this function, improving float32 stability.
    for dx, dy in shifts:
        region = images[:, max(0, -dy):min(size, size - dy), max(0, -dx):min(size, size - dx)]
        sums.append(region.sum(axis=(1, 2)))
        energies.append(np.square(region).sum(axis=(1, 2)))
    return np.stack(sums, axis=1), np.stack(energies, axis=1)


def coarse_features(images, size, shifts):
    small = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA) for x in images]).astype(np.float32)
    small -= small.mean(axis=(1, 2), keepdims=True)
    rotated = np.stack([np.rot90(small, k, axes=(1, 2)) for k in range(4)])
    spectra = rfft2(rotated, s=(size * 2, size * 2), axes=(-2, -1), workers=1)
    sums, energies = zip(*(rectangles(x, shifts) for x in rotated))
    return spectra, np.stack(sums), np.stack(energies)


def correlation_values(fa, fb, suma, sumb, energy_a, energy_b, counts, reverse, size, yi, xi):
    corr = irfft2(np.conj(fa) * fb, s=(size * 2, size * 2), axes=(-2, -1), workers=1)
    cross = corr[..., yi, xi]
    sb = sumb[..., reverse]
    variance_a = np.maximum(energy_a - suma * suma / counts, 0)
    variance_b = np.maximum(energy_b[..., reverse] - sb * sb / counts, 0)
    denom = np.sqrt(variance_a * variance_b)
    return np.divide(cross - suma * sb / counts, denom,
                     out=np.full_like(cross, -1.0), where=denom > 1e-5).clip(-1, 1)


def pair_coarse(a, b, settings):
    size = settings["grid_size"]
    shifts, counts, reverse = grid_geometry(size, settings["max_shift_fraction"], settings["min_overlap"])
    spectra, sums, energies = coarse_features([a, b], size, shifts)
    yi = np.array([dy % (2 * size) for dx, dy in shifts])
    xi = np.array([dx % (2 * size) for dx, dy in shifts])
    best = (-2.0, 0, 0, 0)
    for k in range(4):
        values = correlation_values(spectra[0, 0], spectra[k, 1:2], sums[0, 0], sums[k, 1:2],
                                    energies[0, 0], energies[k, 1:2], counts, reverse, size, yi, xi)[0]
        p = int(values.argmax())
        if float(values[p]) > best[0]:
            best = (float(values[p]), k, *shifts[p])
    return {"coarse_ncc": best[0], "coarse_rotation_b_ccw_degrees": best[1] * 90,
            "coarse_shift_b_grid_xy": [best[2], best[3]]}


def screen_pool(images, settings):
    size = settings["grid_size"]
    shifts, counts, reverse = grid_geometry(size, settings["max_shift_fraction"], settings["min_overlap"])
    spectra, sums, energies = coarse_features(images, size, shifts)
    yi = np.array([dy % (2 * size) for dx, dy in shifts])
    xi = np.array([dx % (2 * size) for dx, dy in shifts])
    n = len(images)
    candidates = []
    best_matrix = np.full((n, n), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for i in range(n - 1):
        best = np.full(n - i - 1, -2, dtype=np.float32)
        best_shift = np.zeros(n - i - 1, dtype=np.int16)
        best_rotation = np.zeros(n - i - 1, dtype=np.uint8)
        for k in range(4):
            values = correlation_values(
                spectra[0, i], spectra[k, i + 1:], sums[0, i], sums[k, i + 1:],
                energies[0, i], energies[k, i + 1:], counts, reverse, size, yi, xi)
            p = values.argmax(axis=1)
            current = values[np.arange(len(p)), p]
            changed = current > best
            best[changed] = current[changed]
            best_shift[changed] = p[changed]
            best_rotation[changed] = k
        best_matrix[i, i + 1:] = best
        for relative in np.flatnonzero(best >= settings["coarse_ncc_threshold"]):
            j = i + 1 + int(relative)
            dx, dy = shifts[int(best_shift[relative])]
            candidates.append({"i": i, "j": j, "coarse_ncc": float(best[relative]),
                               "coarse_rotation_b_ccw_degrees": int(best_rotation[relative]) * 90,
                               "coarse_shift_b_grid_xy": [dx, dy]})
        if (i + 1) % 200 == 0 or i == n - 2:
            print(f"coarse {i+1}/{n-1}; passing {len(candidates)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    return candidates, best_matrix, len(shifts)


def ncc(a, b):
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    aa -= aa.mean()
    bb -= bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.clip(np.dot(aa, bb) / denom, -1, 1)) if denom > 1e-8 else -1.0


def overlap_regions(a, b, dx, dy):
    h, w = a.shape
    ax0, ax1 = max(0, -dx), min(w, w - dx)
    ay0, ay1 = max(0, -dy), min(h, h - dy)
    return a[ay0:ay1, ax0:ax1], b[ay0+dy:ay1+dy, ax0+dx:ax1+dx]


def verify_pair(a, b, settings, coarse, fast=False):
    h, w = a.shape
    bound_x = int(w * settings["max_shift_fraction"])
    bound_y = int(h * settings["max_shift_fraction"])
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    af = a.astype(np.float32)
    best = None
    for k in range(4):
        br = np.ascontiguousarray(np.rot90(b, k))
        bf = br.astype(np.float32)
        seeds = {(0, 0)}
        responses = []
        for use_window in (False, True):
            # phaseCorrelate may mutate its inputs when applying a supplied window.
            shift, response = cv2.phaseCorrelate(af.copy(), bf.copy(), window.copy() if use_window else None)
            responses.append({"hanning": use_window, "shift_xy": list(map(float, shift)), "response": float(response)})
            if all(np.isfinite(shift)):
                x, y = [int(round(t)) for t in shift]
                offsets = (0,) if fast else (-1, 0, 1)
                seeds.update((x+sx, y+sy) for sx in offsets for sy in offsets)
        if coarse["coarse_rotation_b_ccw_degrees"] == k * 90:
            x, y = coarse["coarse_shift_b_grid_xy"]
            seeds.add((int(round(x * w / settings["grid_size"])), int(round(y * h / settings["grid_size"]))))
        for dx, dy in sorted(seeds):
            if abs(dx) > bound_x or abs(dy) > bound_y:
                continue
            overlap = (w - abs(dx)) * (h - abs(dy)) / (w * h)
            if overlap < settings["min_overlap"]:
                continue
            aa, bb = overlap_regions(a, br, dx, dy)
            score = ncc(aa, bb)
            if best is None or score > best["pixel_ncc"]:
                best = {"rotation_b_ccw_degrees": k * 90, "shift_b_xy_pixels": [dx, dy],
                        "overlap_fraction": overlap, "pixel_ncc": score, "phase_estimates": responses}
    if best is None:
        raise RuntimeError("Zero translation must be eligible")
    if fast:
        return best
    br = np.ascontiguousarray(np.rot90(b, best["rotation_b_ccw_degrees"] // 90))
    aa, bb = overlap_regions(a, br, *best["shift_b_xy_pixels"])
    # Interior derivatives avoid the artificial crop-boundary derivative.
    ga = np.stack([cv2.Sobel(aa, cv2.CV_32F, 1, 0), cv2.Sobel(aa, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
    gb = np.stack([cv2.Sobel(bb, cv2.CV_32F, 1, 0), cv2.Sobel(bb, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
    best["gradient_ncc"] = ncc(ga, gb)
    best["mean_absolute_pixel_error_0_255"] = float(np.abs(aa.astype(np.float32) - bb).mean())
    best["passes_review_threshold"] = bool(best["pixel_ncc"] >= settings["pixel_ncc_threshold"]
                                          and best["gradient_ncc"] >= settings["gradient_ncc_threshold"])
    return best


def intermediate_check(a, b, settings, coarse):
    size = settings["refine_size"]
    small = [cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA) for image in (a, b)]
    evidence = verify_pair(*small, settings, coarse, fast=True)
    return {"intermediate_pixel_ncc": evidence["pixel_ncc"],
            "intermediate_pass": evidence["pixel_ncc"] >= settings["refine_ncc_threshold"]}


def sensitivity_controls(reference_root, records, images, settings):
    refs, ref_images, split_hash = load_pool(reference_root)
    lookup = {r["file"]: (r, image) for r, image in zip(refs, ref_images)}
    known = []
    for an, bn in [("scratches_247.jpg", "scratches_232.jpg"), ("pitted_surface_292.jpg", "pitted_surface_171.jpg")]:
        ar, a = lookup[an]
        br, b = lookup[bn]
        coarse = pair_coarse(a, b, settings)
        middle = intermediate_check(a, b, settings, coarse)
        fine = verify_pair(a, b, settings, coarse)
        known.append({"a": ar, "b": br, **coarse, **middle, **fine,
                      "coarse_pass": coarse["coarse_ncc"] >= settings["coarse_ncc_threshold"]})
    # One train source per filename prefix; selection is deterministic and label-free.
    selected = {}
    for record, image in zip(records, images):
        if record["split"] == "train":
            selected.setdefault(record["file"].rsplit("_", 1)[0], (record, image))
    positives, negatives = [], []
    rng = np.random.default_rng(20261031)
    for prefix, (record, a) in sorted(selected.items()):
        for k in range(4):
            for dx, dy in [(0, 0), (3, -2), (20, 11), (45, 0)]:
                rotated = np.ascontiguousarray(np.rot90(a, k))
                b = rng.integers(0, 256, a.shape, dtype=np.uint8)
                h, w = a.shape
                x0, x1, y0, y1 = max(0, -dx), min(w, w-dx), max(0, -dy), min(h, h-dy)
                b[y0+dy:y1+dy, x0+dx:x1+dx] = rotated[y0:y1, x0:x1]
                coarse = pair_coarse(a, b, settings)
                middle = intermediate_check(a, b, settings, coarse)
                fine = verify_pair(a, b, settings, coarse)
                positives.append({"source": record, "generated_rotation_ccw": k*90,
                                  "generated_translation_after_rotation_xy": [dx, dy], **coarse, **middle,
                                  "coarse_pass": coarse["coarse_ncc"] >= settings["coarse_ncc_threshold"],
                                  "pixel_ncc": fine["pixel_ncc"], "gradient_ncc": fine["gradient_ncc"],
                                  "passes_review_threshold": fine["passes_review_threshold"]})
        shuffled = rng.permutation(a.ravel()).reshape(a.shape)
        coarse = pair_coarse(a, shuffled, settings)
        middle = intermediate_check(a, shuffled, settings, coarse)
        fine = verify_pair(a, shuffled, settings, coarse)
        negatives.append({"source": record, "construction": "pixel permutation", **coarse, **middle,
                          "coarse_pass": coarse["coarse_ncc"] >= settings["coarse_ncc_threshold"],
                          "pixel_ncc": fine["pixel_ncc"], "gradient_ncc": fine["gradient_ncc"],
                          "passes_review_threshold": fine["passes_review_threshold"]})
    return {"purpose": "Implementation sensitivity checks, not independent recall/specificity calibration",
            "reference_root": str(reference_root), "reference_split_sha256": split_hash,
            "known_pair_source": "https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected",
            "synthetic_seed": 20261031, "known_pairs": known, "synthetic_positive_controls": positives,
            "synthetic_negative_controls": negatives,
            "summary": {"known_pairs_all_stages_pass": sum(x["coarse_pass"] and x["intermediate_pass"] and x["passes_review_threshold"] for x in known),
                        "synthetic_positives": len(positives),
                        "synthetic_positive_coarse_pass": sum(x["coarse_pass"] for x in positives),
                        "synthetic_positive_all_stages_pass": sum(x["coarse_pass"] and x["intermediate_pass"] and x["passes_review_threshold"] for x in positives),
                        "synthetic_negatives": len(negatives),
                        "synthetic_negative_all_stages_pass": sum(x["coarse_pass"] and x["intermediate_pass"] and x["passes_review_threshold"] for x in negatives)}}


def write_montages(outdir, candidates, images, limit=24):
    for number, row in enumerate(candidates[:limit], 1):
        a, b = images[row["i"]], images[row["j"]]
        br = np.ascontiguousarray(np.rot90(b, row["rotation_b_ccw_degrees"] // 90))
        dx, dy = row["shift_b_xy_pixels"]
        warped = cv2.warpAffine(br, np.array([[1, 0, -dx], [0, 1, -dy]], np.float32), (200, 200))
        diff = cv2.absdiff(a, warped)
        canvas = np.full((248, 800), 255, np.uint8)
        canvas[48:, :] = np.concatenate([a, b, warped, diff], axis=1)
        for x, text in [(2, row["a"]["file"]), (202, row["b"]["file"]), (402, "B aligned to A"), (602, "absolute difference")]:
            cv2.putText(canvas, text, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, .37, 0, 1)
        cv2.putText(canvas, f"NCC {row['pixel_ncc']:.4f}; gradient {row['gradient_ncc']:.4f}; overlap {row['overlap_fraction']:.3f}; borders are invalid", (2, 37), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
        path = outdir / f"candidate_{number:03d}.png"
        cv2.imencode(".png", canvas)[1].tofile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--reference", type=Path, required=True, help="Original v1 pool containing the two known pairs")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-shift-fraction", type=float, default=.25)
    parser.add_argument("--min-overlap", type=float, default=.60)
    parser.add_argument("--coarse-ncc-threshold", type=float, default=.85)
    parser.add_argument("--refine-size", type=int, default=64)
    parser.add_argument("--refine-ncc-threshold", type=float, default=.85)
    parser.add_argument("--pixel-ncc-threshold", type=float, default=.95)
    parser.add_argument("--gradient-ncc-threshold", type=float, default=.90)
    args = parser.parse_args()
    root, reference, outdir = args.dataset.resolve(), args.reference.resolve(), args.outdir.resolve()
    if outdir.exists() or outdir == root or root in outdir.parents or reference in outdir.parents:
        raise SystemExit("Refusing existing output or an output within an input dataset")
    if not (4 <= args.grid_size <= 64 and 0 <= args.max_shift_fraction < .5 and 0 < args.min_overlap <= 1):
        raise SystemExit("Invalid grid or overlap settings")
    settings = {k: v for k, v in vars(args).items() if k not in {"dataset", "reference", "outdir"}}
    cv2.setNumThreads(1)
    started = time.perf_counter()
    records, images, split_hash = load_pool(root)
    outdir.mkdir(parents=True)
    manifest_text = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest = {"dataset": str(root), "split_sha256": split_hash, "image_manifest_sha256": manifest_hash,
                "images": records, "caution": CAUTION}
    (outdir / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    controls = sensitivity_controls(reference, records, images, settings)
    (outdir / "sensitivity_controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=2), encoding="utf-8")
    print("controls", controls["summary"], flush=True)
    coarse, score_matrix, shift_count = screen_pool(images, settings)
    np.savez_compressed(outdir / "coarse_all_pairs.npz", upper_triangle_max_ncc=score_matrix)
    verified = []
    intermediate_rejected = []
    verify_started = time.perf_counter()
    for number, pair in enumerate(coarse, 1):
        middle = intermediate_check(images[pair["i"]], images[pair["j"]], settings, pair)
        if not middle["intermediate_pass"]:
            intermediate_rejected.append({**pair, **middle})
            if number % 1000 == 0 or number == len(coarse):
                print(f"intermediate {number}/{len(coarse)}; full-resolution passes {len(verified)}; elapsed {time.perf_counter()-verify_started:.1f}s", flush=True)
            continue
        fine = verify_pair(images[pair["i"]], images[pair["j"]], settings, pair)
        verified.append({**pair, "a": records[pair["i"]], "b": records[pair["j"]],
                         "cross_split": records[pair["i"]]["split"] != records[pair["j"]]["split"], **middle, **fine})
        if number % 100 == 0 or number == len(coarse):
            print(f"full-resolution {number}/{len(coarse)}; elapsed {time.perf_counter()-verify_started:.1f}s", flush=True)
    verified.sort(key=lambda x: (-x["pixel_ncc"], -x["gradient_ncc"], x["i"], x["j"]))
    review = [row for row in verified if row["passes_review_threshold"]]
    cross = [row for row in review if row["cross_split"]]
    digest_after = []
    for row in records:
        path = root / "images" / row["split"] / row["file"]
        digest_after.append(sha256(path) == row["sha256"])
    unchanged = all(digest_after) and sha256(root / "split.json") == split_hash
    if not unchanged:
        raise RuntimeError("Input images/split changed during audit")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(root),
              "script_sha256": sha256(Path(__file__)), "settings": settings,
              "method": "All unordered pairs; overlap NCC on area-resized grid over every eligible grid translation and quarter turn; 64px phase-correlation proposed integer shifts; original-resolution phase refinement; overlap pixel and Sobel-gradient NCC",
              "shift_convention": "Rotate B counterclockwise first; compare A[y,x] with rotated_B[y+dy,x+dx]. Grid shifts multiply by 200/grid_size. Only valid overlap enters scores.",
              "threshold_basis": "Exploratory fixed review thresholds chosen before this run; not a learned/calibrated decision rule. Two known cases and synthetic controls are sensitivity checks only.",
              "scope": "All train/val/test image pixels and filenames; no annotation reads or detector/model metrics. No class restriction. Test pixels used exclusively for data-integrity screening.",
              "input_manifest_sha256": manifest_hash, "input_images_and_split_unchanged": unchanged,
              "image_count": len(records), "split_counts": dict(collections.Counter(r["split"] for r in records)),
              "unordered_pairs_screened": len(records)*(len(records)-1)//2,
              "quarter_turns": [0, 90, 180, 270], "eligible_translations_per_turn": shift_count,
              "coarse_passing_pairs": len(coarse), "review_candidate_count": len(review),
              "intermediate_rejected_count": len(intermediate_rejected), "original_resolution_verified_count": len(verified),
              "cross_split_review_candidate_count": len(cross), "review_candidates": review,
              "cross_split_review_candidates": cross, "all_intermediate_passing_pairs_with_verification": verified,
              "intermediate_rejected_pairs": intermediate_rejected,
              "controls_summary": controls["summary"], "wall_seconds": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__, "platform": platform.platform()},
              "caution": CAUTION,
              "sources": ["https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html", "https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected"]}
    (outdir / "candidate_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_montages(outdir, review, images)
    print(json.dumps({k: result[k] for k in ("image_count", "unordered_pairs_screened", "coarse_passing_pairs", "review_candidate_count", "cross_split_review_candidate_count", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
