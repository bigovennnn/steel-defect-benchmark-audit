"""Package candidate audit evidence for review; never adjudicate or change data."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    out = args.outdir.resolve()
    root = Path(audit["dataset"])
    if out.exists() or out == root or root in out.parents:
        raise SystemExit("Refusing existing output or writing within input data")
    module_path = Path(__file__).with_name("audit_neu_rigid_candidates.py")
    if hashlib.sha256(module_path.read_bytes()).hexdigest() != audit["script_sha256"]:
        raise SystemExit("Audit script changed since the completed run")
    spec = importlib.util.spec_from_file_location("rigid_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.cv2.setNumThreads(1)
    records, images, _ = module.load_pool(root)
    text = json.dumps(records, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if hashlib.sha256(text.encode()).hexdigest() != audit["input_manifest_sha256"]:
        raise SystemExit("Input image manifest changed")
    out.mkdir(parents=True)
    image_dir = out / "cross_split_evidence"
    image_dir.mkdir()
    cross = audit["cross_split_review_candidates"]
    module.write_montages(image_dir, cross, images, limit=len(cross))
    rank = {"train": 0, "val": 1, "test": 2}
    keys = [("train", "train"), ("train", "val"), ("train", "test"),
            ("val", "val"), ("val", "test"), ("test", "test")]
    counts = collections.Counter()
    cross_index = {(row["i"], row["j"]): number for number, row in enumerate(cross, 1)}
    rows = []
    for number, candidate in enumerate(audit["review_candidates"], 1):
        a, b = candidate["a"], candidate["b"]
        pair = tuple(sorted([a["split"], b["split"]], key=rank.get))
        counts[pair] += 1
        evidence_number = cross_index.get((candidate["i"], candidate["j"]))
        rows.append({"candidate_id": f"pair_{number:04d}", "a_file": a["file"], "a_split": a["split"],
                     "b_file": b["file"], "b_split": b["split"], "split_pair": "/".join(pair),
                     "cross_split": candidate["cross_split"],
                     "rotation_b_ccw_degrees": candidate["rotation_b_ccw_degrees"],
                     "shift_b_dx_pixels": candidate["shift_b_xy_pixels"][0],
                     "shift_b_dy_pixels": candidate["shift_b_xy_pixels"][1],
                     "overlap_fraction": candidate["overlap_fraction"],
                     "pixel_ncc": candidate["pixel_ncc"], "gradient_ncc": candidate["gradient_ncc"],
                     "human_review_status": "pending", "human_adjudication": "undetermined",
                     "assistant_visual_review_status": "not_recorded_in_this_generated_package",
                     "evidence_file": f"cross_split_evidence/candidate_{evidence_number:03d}.png" if evidence_number else "",
                     "review_notes": ""})
    fields = list(rows[0]) if rows else ["candidate_id", "human_review_status"]
    for name, selected in [("review_status.csv", rows), ("cross_split_candidates.csv", [r for r in rows if r["cross_split"]])]:
        with (out / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    summary = {"source_audit": str(args.audit.resolve()),
               "source_audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
               "review_candidates": len(rows), "cross_split_candidates": len(cross),
               "split_pair_counts": {"/".join(pair): counts[pair] for pair in keys},
               "human_review_completed": 0,
               "human_review_policy": "All candidates start pending. Automated thresholds and assistant image inspection are not human adjudication. No files or split assignments are changed.",
               "input_manifest_matches_completed_audit": True,
               "packaging_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
