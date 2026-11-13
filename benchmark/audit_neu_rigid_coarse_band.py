"""Read-only full-resolution screening of a prior coarse-NCC score band.

No model metrics, annotation edits, deletions, or split changes. This widens
only the fixed 16px candidate band; bounded quarter-turn/translation scope
and non-exhaustive phase proposals remain limitations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import cv2
import numpy as np
import audit_neu_rigid_candidates as old
import audit_neu_rigid_supplement as supplement


def make_pool(images, scores, settings, lower, upper, output):
    indices_i, indices_j = np.where(np.isfinite(scores) & (scores >= lower) & (scores < upper))
    size = settings["grid_size"]
    shifts, counts, reverse = old.grid_geometry(size, settings["max_shift_fraction"], settings["min_overlap"])
    spectra, sums, energies = old.coarse_features(images, size, shifts)
    yi = np.array([dy % (2*size) for dx, dy in shifts])
    xi = np.array([dx % (2*size) for dx, dy in shifts])
    rotation = np.zeros(len(indices_i), np.uint8)
    shift_index = np.zeros(len(indices_i), np.int16)
    best = np.full(len(indices_i), -2, np.float32)
    for i in np.unique(indices_i):
        locations = np.flatnonzero(indices_i == i)
        js = indices_j[locations]
        for k in range(4):
            values = old.correlation_values(spectra[0, i], spectra[k, js], sums[0, i], sums[k, js],
                                           energies[0, i], energies[k, js], counts, reverse, size, yi, xi)
            p = values.argmax(axis=1)
            current = values[np.arange(len(js)), p]
            changed = current > best[locations]
            changed_locations = locations[changed]
            best[changed_locations] = current[changed]
            shift_index[changed_locations] = p[changed]
            rotation[changed_locations] = k
    discrepancy = float(np.max(np.abs(best - scores[indices_i, indices_j])))
    if discrepancy > 1e-5:
        raise RuntimeError(f"Coarse reconstruction mismatch {discrepancy}")
    shifts_array = np.array(shifts, np.int16)[shift_index]
    np.savez_compressed(output, i=indices_i.astype(np.int16), j=indices_j.astype(np.int16),
                        coarse_ncc=best, rotation=rotation, shifts=shifts_array)
    return {"count": len(best), "max_abs_reconstructed_coarse_score_error": discrepancy}


def pair_at(pool, index):
    return {"i": int(pool["i"][index]), "j": int(pool["j"][index]),
            "coarse_ncc": float(pool["coarse_ncc"][index]),
            "coarse_rotation_b_ccw_degrees": int(pool["rotation"][index])*90,
            "coarse_shift_b_grid_xy": list(map(int, pool["shifts"][index]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--validated-supplement", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--lower", type=float, default=.80)
    p.add_argument("--upper", type=float, default=.85)
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-estimated-seconds", type=float, default=1800)
    args = p.parse_args()
    root, prior, validated, out = (x.resolve() for x in (args.dataset, args.prior, args.validated_supplement, args.outdir))
    if out.drive.lower() != "d:" or any(out == x or x in out.parents for x in (root, prior, validated)):
        raise SystemExit("Independent D: output required")
    if out.exists() and not args.resume:
        raise SystemExit("Existing output requires --resume")
    if not 0 <= args.lower < args.upper <= 1:
        raise SystemExit("Invalid score band")
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    base = json.loads((validated / "supplemental_audit.json").read_text(encoding="utf-8"))
    calibration = json.loads((validated / "calibration.json").read_text(encoding="utf-8"))
    if base["supplemental_pairs_verified"] != 77791 or not base["input_images_and_split_unchanged"] or calibration["summary"]["supplement_phase_pass"] != 96:
        raise SystemExit("Validated prior supplemental controls/results are required")
    settings = base["settings"]
    if args.upper > settings["coarse_ncc_threshold"]:
        raise SystemExit("This supplemental score band must remain below the prior threshold")
    records, images, split_hash = old.load_pool(root)
    manifest = json.loads((prior / "input_manifest.json").read_text(encoding="utf-8"))
    if records != manifest["images"] or split_hash != manifest["split_sha256"]:
        raise SystemExit("Dataset differs from the original audit manifest")
    manifest_hash = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    sources = {str(p): old.sha256(p) for p in (Path(__file__), Path(old.__file__), Path(supplement.__file__))}
    pool_path = out / "candidate_pool.npz"
    if not pool_path.exists():
        scores = np.load(prior / "coarse_all_pairs.npz")["upper_triangle_max_ncc"]
        info = make_pool(images, scores, settings, args.lower, args.upper, pool_path)
        info.update({"lower_inclusive": args.lower, "upper_exclusive": args.upper,
                     "prior_score_matrix_sha256": old.sha256(prior / "coarse_all_pairs.npz"),
                     "input_manifest_sha256": manifest_hash, "sources_sha256": sources})
        supplement.dump(out / "pool_manifest.json", info)
        print("pool", info["count"], "coarse reconstruction max error", info["max_abs_reconstructed_coarse_score_error"], flush=True)
    info = json.loads((out / "pool_manifest.json").read_text(encoding="utf-8"))
    if info["input_manifest_sha256"] != manifest_hash or info["lower_inclusive"] != args.lower or info["upper_exclusive"] != args.upper:
        raise SystemExit("Pool identity mismatch")
    with np.load(pool_path) as saved_pool:
        pool = {key: saved_pool[key] for key in saved_pool.files}
    total = len(pool["i"])
    window = cv2.createHanningWindow((200, 200), cv2.CV_32F)
    pilot_path = out / "pilot.json"
    if not pilot_path.exists():
        rng = np.random.default_rng(20261102)
        selected = sorted(map(int, rng.choice(total, size=min(200,total), replace=False)))
        t = time.perf_counter()
        phase_rows = []
        for index in selected:
            pair = pair_at(pool, index)
            fine = supplement.phase_verify(images[pair["i"]], images[pair["j"]], settings, pair, window)
            phase_rows.append({"pool_index": index, **pair, "phase": fine})
        seconds = time.perf_counter()-t
        max_error = 0.0
        for row in phase_rows:
            legacy = old.verify_pair(images[row["i"]], images[row["j"]], settings, row)
            max_error = max(max_error, abs(legacy["pixel_ncc"] - row["phase"]["pixel_ncc"]))
        for row in phase_rows[:20]:
            row["exhaustive_original"] = supplement.exact_verify(images[row["i"]], images[row["j"]], settings)
        pilot = {"purpose": "Implementation and resource pilot, not independent recall/FPR calibration", "seed": 20261102,
                 "sample_size": len(selected), "phase_wall_seconds": seconds,
                 "estimated_full_phase_wall_seconds": seconds/len(selected)*total,
                 "max_abs_legacy_phase_ncc_difference": max_error,
                 "exact_original_checks": 20, "rows": phase_rows}
        supplement.dump(pilot_path, pilot)
        print("pilot", {k:pilot[k] for k in ("sample_size", "phase_wall_seconds", "estimated_full_phase_wall_seconds", "max_abs_legacy_phase_ncc_difference")}, flush=True)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot["max_abs_legacy_phase_ncc_difference"] > 1e-5:
        raise SystemExit("Phase implementation agreement failed")
    if args.pilot_only:
        return
    if pilot["estimated_full_phase_wall_seconds"] > args.max_estimated_seconds:
        raise SystemExit("Resource estimate exceeds the configured bounded run; pilot retained")
    checkpoint = out / "checkpoint.json"
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"next_index": 0, "review_candidates": [], "near_candidates": [], "score_histogram": {}}
    if state.get("sources_sha256", sources) != sources:
        raise SystemExit("Source changed after checkpoint")
    for number in range(state["next_index"], total):
        pair = pair_at(pool, number)
        fine = supplement.phase_verify(images[pair["i"]], images[pair["j"]], settings, pair, window)
        bin_key = str(min(100, max(-100, int(np.floor(fine["pixel_ncc"]*100)))))
        state["score_histogram"][bin_key] = state["score_histogram"].get(bin_key, 0)+1
        if fine["pixel_ncc"] >= .90:
            evidence = {"pool_index": number, **pair, **fine, "a": records[pair["i"]], "b": records[pair["j"]],
                        "cross_split": records[pair["i"]]["split"] != records[pair["j"]]["split"]}
            state["near_candidates"].append(evidence)
            if fine["passes_review_threshold"]:
                evidence["independent_exhaustive_original"] = supplement.exact_verify(images[pair["i"]], images[pair["j"]], settings)
                state["review_candidates"].append(evidence)
        state["next_index"] = number+1
        if (number+1)%1000 == 0 or number+1 == total:
            state["sources_sha256"] = sources
            state["input_manifest_sha256"] = manifest_hash
            state["wall_seconds_this_invocation"] = time.perf_counter()-started
            supplement.dump(checkpoint,state)
            print(f"band {number+1}/{total}; review {len(state['review_candidates'])}; near {len(state['near_candidates'])}; elapsed {time.perf_counter()-started:.1f}s",flush=True)
    unchanged = all(old.sha256(root/'images'/r['split']/r['file']) == r['sha256'] for r in records) and old.sha256(root/'split.json') == split_hash
    if not unchanged:
        raise RuntimeError("Dataset changed during band screening")
    reviews = sorted(state["review_candidates"],key=lambda r:(-r["pixel_ncc"],-r["gradient_ncc"],r["i"],r["j"]))
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"dataset":str(root),"prior":str(prior),"validated_supplement":str(validated),
            "pool_manifest":info,"sources_sha256":sources,"settings":settings,"input_manifest_sha256":manifest_hash,
            "input_images_and_split_unchanged":unchanged,"pairs_verified":state["next_index"],"new_review_candidate_count":len(reviews),
            "new_cross_split_review_candidate_count":sum(r["cross_split"] for r in reviews),"review_candidates":reviews,
            "near_candidate_count_pixel_ncc_ge_090":len(state["near_candidates"]),"score_histogram":state["score_histogram"],
            "wall_seconds_this_invocation":time.perf_counter()-started,"opencv_threads":cv2.getNumThreads(),"blas_threads_requested":1,
            "caution":"Candidate evidence only. The 16px score band [.80,.85) is fully visited, but original-resolution phase proposals are not exhaustive. <.80, non-quarter rotations, reflection, scale, deformation, >25% shifts and <60% overlap remain outside scope. Similar texture can pass. Pilot and controls do not estimate real recall/FPR. No automatic adjudication, split changes, deletion, annotation/model/test metric reads."}
    supplement.dump(out/'band_audit.json',result)
    old.write_montages(out,reviews,images,limit=100)
    print(json.dumps({k:result[k] for k in ("pairs_verified","new_review_candidate_count","new_cross_split_review_candidate_count","input_images_and_split_unchanged","wall_seconds_this_invocation")}),flush=True)


if __name__ == "__main__":
    main()
