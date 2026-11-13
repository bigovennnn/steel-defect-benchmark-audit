"""Read-only supplemental rigid screening; bypass the legacy 64-pixel gate.

This script changes no dataset, prior audit, annotations, or model results.
All outputs and scratch files must be on D:. Phase proposals do not give an
exhaustive search over translations; selected evidence is independently
checked with exhaustive full-resolution FFT overlap NCC.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import cv2
import numpy as np
import scipy
from scipy.fft import rfft2, irfft2
import audit_neu_rigid_candidates as old


CAUTION = (
    "Candidate evidence only. No deletion, group assignment, label changes, or test/model "
    "performance reads. Candidate pool remains limited by the prior 16px NCC >= .85 "
    "screen; only prior 64px rejects are supplemented. Phase proposals can miss "
    "translations. Quarter turns, <=25% per-axis integer translations and >=60% "
    "overlap only; no reflection, arbitrary rotation, scale, perspective or local "
    "deformation. Similar texture may pass. Controls are implementation sensitivity "
    "checks and cannot estimate real-world recall or specificity. A negative result "
    "does not establish absence of duplicates/capture-group leakage."
)


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=4)
def exact_geometry(size, fraction, overlap):
    bound = int(np.floor(size * fraction))
    shifts = [(x, y) for y in range(-bound, bound+1) for x in range(-bound, bound+1)
              if (size-abs(x))*(size-abs(y))/size**2 >= overlap]
    lookup = {s:i for i,s in enumerate(shifts)}
    counts = np.array([(size-abs(x))*(size-abs(y)) for x,y in shifts], dtype=np.float64)
    reverse = np.array([lookup[-x,-y] for x,y in shifts])
    return shifts, counts, reverse


def phase_verify(a, b, settings, coarse, window):
    h, w = a.shape
    bx, by = int(w * settings["max_shift_fraction"]), int(h * settings["max_shift_fraction"])
    af = a.astype(np.float32)
    best = None
    for k in range(4):
        br = np.ascontiguousarray(np.rot90(b, k))
        bf = br.astype(np.float32)
        seeds = {(0, 0)}
        for use_window in (False, True):
            shift, _response = cv2.phaseCorrelate(af.copy(), bf.copy(), window.copy() if use_window else None)
            if all(np.isfinite(shift)):
                x, y = (int(round(t)) for t in shift)
                seeds.update((x + sx, y + sy) for sx in (-1, 0, 1) for sy in (-1, 0, 1))
        if coarse["coarse_rotation_b_ccw_degrees"] == k * 90:
            x, y = coarse["coarse_shift_b_grid_xy"]
            seeds.add((int(round(x * w / settings["grid_size"])),
                       int(round(y * h / settings["grid_size"]))))
        for dx, dy in sorted(seeds):
            if abs(dx) > bx or abs(dy) > by:
                continue
            overlap = (w - abs(dx)) * (h - abs(dy)) / (w * h)
            if overlap < settings["min_overlap"]:
                continue
            aa, bb = old.overlap_regions(a, br, dx, dy)
            score = old.ncc(aa, bb)
            if best is None or score > best["pixel_ncc"]:
                best = {"rotation_b_ccw_degrees": k * 90, "shift_b_xy_pixels": [dx, dy],
                        "overlap_fraction": overlap, "pixel_ncc": score}
    if best is None:
        raise RuntimeError("Zero shift must be eligible")
    # All pixel scores and review thresholds use float64 overlap NCC.
    br = np.ascontiguousarray(np.rot90(b, best["rotation_b_ccw_degrees"] // 90))
    aa, bb = old.overlap_regions(a, br, *best["shift_b_xy_pixels"])
    best["pixel_ncc"] = old.ncc(aa, bb)
    best["gradient_ncc"] = None
    best["passes_review_threshold"] = False
    if best["pixel_ncc"] >= settings["pixel_ncc_threshold"] - 1e-5:
        ga = np.stack([cv2.Sobel(aa, cv2.CV_32F, 1, 0), cv2.Sobel(aa, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
        gb = np.stack([cv2.Sobel(bb, cv2.CV_32F, 1, 0), cv2.Sobel(bb, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
        best["gradient_ncc"] = old.ncc(ga, gb)
        best["mean_absolute_pixel_error_0_255"] = float(np.abs(aa.astype(np.float32) - bb).mean())
        best["passes_review_threshold"] = bool(best["pixel_ncc"] >= settings["pixel_ncc_threshold"]
                                              and best["gradient_ncc"] >= settings["gradient_ncc_threshold"])
    return best


def exact_verify(a, b, settings):
    """Exhaustive full-resolution NCC, no phase/resize or circular overlap."""
    size = a.shape[0]
    shifts, counts, reverse = exact_geometry(size, settings["max_shift_fraction"], settings["min_overlap"])
    dx = np.array([x for x, y in shifts])
    dy = np.array([y for x, y in shifts])
    yi, xi = dy % (2 * size), dx % (2 * size)

    def features(image):
        im = image.astype(np.float64)
        im -= im.mean()
        x0, x1 = np.maximum(0, -dx), np.minimum(size, size - dx)
        y0, y1 = np.maximum(0, -dy), np.minimum(size, size - dy)
        def sums(values):
            ii = cv2.integral(values)
            return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
        return rfft2(im, s=(2 * size, 2 * size), workers=1), sums(im), sums(im * im)

    fa, sa, ea = features(a)
    best = (-2.0, 0, 0, 0)
    for k in range(4):
        fb, sb, eb = features(np.ascontiguousarray(np.rot90(b, k)))
        cross = irfft2(np.conj(fa) * fb, s=(2 * size, 2 * size), workers=1)[yi, xi]
        va = np.maximum(ea - sa * sa / counts, 0)
        vb = np.maximum(eb[reverse] - sb[reverse] ** 2 / counts, 0)
        den = np.sqrt(va * vb)
        val = np.divide(cross - sa * sb[reverse] / counts, den,
                        out=np.full_like(cross, -1.0), where=den > 1e-8).clip(-1, 1)
        p = int(val.argmax())
        if float(val[p]) > best[0]:
            best = float(val[p]), k, int(dx[p]), int(dy[p])
    score, k, x, y = best
    aa, bb = old.overlap_regions(a, np.ascontiguousarray(np.rot90(b, k)), x, y)
    ga = np.stack([cv2.Sobel(aa, cv2.CV_32F, 1, 0), cv2.Sobel(aa, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
    gb = np.stack([cv2.Sobel(bb, cv2.CV_32F, 1, 0), cv2.Sobel(bb, cv2.CV_32F, 0, 1)])[:, 1:-1, 1:-1]
    pn, gn = old.ncc(aa, bb), old.ncc(ga, gb)
    return {"rotation_b_ccw_degrees": k * 90, "shift_b_xy_pixels": [x, y],
            "overlap_fraction": (size - abs(x)) * (size - abs(y)) / size ** 2,
            "fft_max_ncc": score, "pixel_ncc": pn, "gradient_ncc": gn,
            "passes_review_threshold": pn >= settings["pixel_ncc_threshold"] and gn >= settings["gradient_ncc_threshold"],
            "eligible_translations_per_turn": len(shifts)}


def regenerate_controls(records, images, settings, prior_controls, window):
    selected = {}
    for r, im in zip(records, images):
        if r["split"] == "train":
            selected.setdefault(r["file"].rsplit("_", 1)[0], (r, im))
    rng = np.random.default_rng(20261031)
    rows, negatives, mismatches = [], [], []
    prior = prior_controls["synthetic_positive_controls"]
    for prefix, (record, a) in sorted(selected.items()):
        for k in range(4):
            for dx, dy in [(0, 0), (3, -2), (20, 11), (45, 0)]:
                rotated = np.ascontiguousarray(np.rot90(a, k))
                b = rng.integers(0, 256, a.shape, dtype=np.uint8)
                x0, x1, y0, y1 = max(0, -dx), min(200, 200-dx), max(0, -dy), min(200, 200-dy)
                b[y0+dy:y1+dy, x0+dx:x1+dx] = rotated[y0:y1, x0:x1]
                previous = prior[len(rows)]
                coarse = old.pair_coarse(a, b, settings)
                if record != previous["source"] or abs(coarse["coarse_ncc"] - previous["coarse_ncc"]) > 1e-5:
                    mismatches.append(len(rows))
                phase = phase_verify(a, b, settings, coarse, window)
                exact = exact_verify(a, b, settings)
                rows.append({"source": record, "generated_rotation_ccw": k*90,
                             "generated_translation_after_rotation_xy": [dx, dy],
                             "prior_intermediate_pixel_ncc": previous["intermediate_pixel_ncc"],
                             "prior_intermediate_pass": previous["intermediate_pass"],
                             "supplement_phase": phase, "exhaustive_original": exact})
        # Keep the legacy RNG stream identical between classes.
        shuffled = rng.permutation(a.ravel()).reshape(a.shape)
        coarse = old.pair_coarse(a, shuffled, settings)
        negatives.append({"source": record, "construction": "pixel permutation",
                          "supplement_phase": phase_verify(a, shuffled, settings, coarse, window),
                          "exhaustive_original": exact_verify(a, shuffled, settings)})
    return {"purpose": "Same 96 deterministic implementation controls, not independent calibration",
            "regeneration_mismatch_indices": mismatches,
            "summary": {"count": len(rows),
                        "prior_all_stages_pass": sum(x["prior_intermediate_pass"] for x in rows),
                        "supplement_phase_pass": sum(x["supplement_phase"]["passes_review_threshold"] for x in rows),
                        "exhaustive_original_pass": sum(x["exhaustive_original"]["passes_review_threshold"] for x in rows),
                        "negative_count": len(negatives),
                        "negative_supplement_phase_pass": sum(x["supplement_phase"]["passes_review_threshold"] for x in negatives),
                        "negative_exhaustive_original_pass": sum(x["exhaustive_original"]["passes_review_threshold"] for x in negatives)},
            "controls": rows, "negative_controls": negatives}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    args = p.parse_args()
    dataset, prior, out = args.dataset.resolve(), args.prior.resolve(), args.outdir.resolve()
    if out.drive.lower() != "d:" or dataset == out or dataset in out.parents or prior == out or prior in out.parents:
        raise SystemExit("Output must be independent of inputs and on D:")
    if out.exists() and not args.resume:
        raise SystemExit("Existing output requires --resume")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    previous = json.loads((prior / "candidate_audit.json").read_text(encoding="utf-8"))
    settings = previous["settings"]
    records, images, split_hash = old.load_pool(dataset)
    old_manifest = json.loads((prior / "input_manifest.json").read_text(encoding="utf-8"))
    if records != old_manifest["images"] or split_hash != old_manifest["split_sha256"]:
        raise SystemExit("Dataset mismatch against prior input manifest")
    manifest_hash = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    window = cv2.createHanningWindow((200, 200), cv2.CV_32F)
    controls_path = out / "calibration.json"
    if not controls_path.exists():
        old_controls = json.loads((prior / "sensitivity_controls.json").read_text(encoding="utf-8"))
        controls = regenerate_controls(records, images, settings, old_controls, window)
        sample = previous["intermediate_rejected_pairs"][:200]
        t = time.perf_counter()
        differences = []
        for row in sample:
            new = phase_verify(images[row["i"]], images[row["j"]], settings, row, window)
            legacy = old.verify_pair(images[row["i"]], images[row["j"]], settings, row)
            differences.append(abs(new["pixel_ncc"] - legacy["pixel_ncc"]))
        controls["phase_implementation_agreement"] = {"pairs": len(sample), "max_abs_pixel_ncc_difference": max(differences),
                                                       "new_plus_legacy_wall_seconds": time.perf_counter()-t}
        t = time.perf_counter()
        for row in sample:
            phase_verify(images[row["i"]], images[row["j"]], settings, row, window)
        controls["phase_timing"] = {"pairs": len(sample), "wall_seconds": time.perf_counter()-t}
        dump(controls_path, controls)
        print("calibration", controls["summary"], controls["phase_implementation_agreement"], controls["phase_timing"], flush=True)
    else:
        controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if controls["regeneration_mismatch_indices"] or controls["summary"]["supplement_phase_pass"] != 96 or controls["summary"]["exhaustive_original_pass"] != 96 or controls["phase_implementation_agreement"]["max_abs_pixel_ncc_difference"] > 1e-5:
        raise SystemExit("Calibration failed; refuse full supplemental scan")
    if args.calibrate_only:
        return
    checkpoint = out / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"next_index": 0, "review_candidates": [], "near_candidates": [], "score_histogram": {}}
    pool = previous["intermediate_rejected_pairs"]
    if state.get("script_sha256", old.sha256(Path(__file__))) != old.sha256(Path(__file__)):
        raise SystemExit("Script changed since checkpoint")
    for number in range(state["next_index"], len(pool)):
        row = pool[number]
        fine = phase_verify(images[row["i"]], images[row["j"]], settings, row, window)
        key = str(min(100, max(-100, int(np.floor(fine["pixel_ncc"] * 100)))))
        state["score_histogram"][key] = state["score_histogram"].get(key, 0) + 1
        if fine["pixel_ncc"] >= .90:
            evidence = {**row, "a": records[row["i"]], "b": records[row["j"]],
                        "cross_split": records[row["i"]]["split"] != records[row["j"]]["split"], **fine}
            state["near_candidates"].append(evidence)
            if fine["passes_review_threshold"]:
                evidence["independent_exhaustive_original"] = exact_verify(images[row["i"]], images[row["j"]], settings)
                state["review_candidates"].append(evidence)
        state["next_index"] = number + 1
        if (number + 1) % args.checkpoint_every == 0 or number + 1 == len(pool):
            state["script_sha256"] = old.sha256(Path(__file__))
            state["dataset_manifest_sha256"] = manifest_hash
            state["latest_run_wall_seconds"] = time.perf_counter() - started
            dump(checkpoint, state)
            print(f"supplement {number+1}/{len(pool)}; review {len(state['review_candidates'])}; near {len(state['near_candidates'])}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    # A deterministic stratified check of phase-negative pairs with full FFT.
    validation_path = out / "phase_negative_exhaustive_checks.json"
    if not validation_path.exists():
        rng = np.random.default_rng(20261101)
        selected = rng.choice(len(pool), size=min(200, len(pool)), replace=False)
        rows = []
        for index in sorted(map(int, selected)):
            row = pool[index]
            phase = phase_verify(images[row["i"]], images[row["j"]], settings, row, window)
            exact = exact_verify(images[row["i"]], images[row["j"]], settings)
            rows.append({"pool_index": index, "i": row["i"], "j": row["j"], "phase": phase, "exhaustive_original": exact})
        dump(validation_path, {"purpose": "Deterministic implementation discrepancy probe, not recall estimate", "seed": 20261101,
                               "sample_size": len(rows), "exhaustive_review_pass": sum(x["exhaustive_original"]["passes_review_threshold"] for x in rows),
                               "phase_missed_review_pass": sum(x["exhaustive_original"]["passes_review_threshold"] and not x["phase"]["passes_review_threshold"] for x in rows),
                               "checks": rows})
    unchanged = all(old.sha256(dataset / "images" / r["split"] / r["file"]) == r["sha256"] for r in records) and old.sha256(dataset / "split.json") == split_hash
    if not unchanged:
        raise RuntimeError("Dataset changed during supplemental audit")
    reviews = sorted(state["review_candidates"], key=lambda x: (-x["pixel_ncc"], -x["gradient_ncc"], x["i"], x["j"]))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset), "prior_audit": str(prior),
              "script_sha256": old.sha256(Path(__file__)), "legacy_script_sha256": old.sha256(Path(old.__file__)),
              "input_manifest_sha256": manifest_hash, "input_images_and_split_unchanged": unchanged,
              "image_count": len(records), "all_unordered_pairs": len(records)*(len(records)-1)//2,
              "candidate_pool": "Exactly all 77791 prior coarse-passing (16px NCC>=.85) pairs rejected by 64px gate; no restriction by split/class",
              "settings": settings, "supplemental_pairs_verified": state["next_index"],
              "prior_original_verified": previous["original_resolution_verified_count"],
              "total_original_verified_in_union": state["next_index"] + previous["original_resolution_verified_count"],
              "new_review_candidate_count": len(reviews), "new_cross_split_review_candidate_count": sum(x["cross_split"] for x in reviews),
              "review_candidates": reviews, "near_candidate_count_pixel_ncc_ge_090": len(state["near_candidates"]),
              "score_histogram_floor_ncc_percent": state["score_histogram"], "controls_summary": controls["summary"],
              "phase_negative_exhaustive_probe": json.loads(validation_path.read_text(encoding="utf-8")),
              "wall_seconds_this_invocation": time.perf_counter()-started,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(), "blas_threads_requested": 1},
              "caution": CAUTION}
    dump(out / "supplemental_audit.json", result)
    old.write_montages(out, reviews, images, limit=100)
    print(json.dumps({k:result[k] for k in ("supplemental_pairs_verified", "new_review_candidate_count", "new_cross_split_review_candidate_count", "input_images_and_split_unchanged", "wall_seconds_this_invocation")}), flush=True)


if __name__ == "__main__":
    main()
