"""Small deterministic exhaustive-NCC probe of a completed coarse-band audit."""
from pathlib import Path
import argparse
import hashlib
import json
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import cv2
import numpy as np
import audit_neu_rigid_candidates as old
import audit_neu_rigid_supplement as supplement
import audit_neu_rigid_coarse_band as band


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--banddir",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=200)
    args=parser.parse_args()
    out=args.banddir.resolve()
    if out.drive.lower() != "d:" or (out/"exhaustive_probe.json").exists():
        raise SystemExit("Require D: output without an existing probe result")
    result=json.loads((out/"band_audit.json").read_text(encoding="utf-8"))
    with np.load(out/"candidate_pool.npz") as saved:
        pool={key:saved[key] for key in saved.files}
    if result["pairs_verified"] != len(pool["i"]):
        raise SystemExit("Complete coarse-band audit required")
    records,images,split_hash=old.load_pool(args.dataset.resolve())
    manifest_hash=hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != result["input_manifest_sha256"]:
        raise SystemExit("Dataset manifest mismatch")
    cv2.setNumThreads(1)
    window=cv2.createHanningWindow((200,200),cv2.CV_32F)
    rng=np.random.default_rng(20261103)
    indices=sorted(map(int,rng.choice(len(pool["i"]),size=min(args.samples,len(pool["i"])),replace=False)))
    rows=[]
    for index in indices:
        pair=band.pair_at(pool,index)
        rows.append({"pool_index":index,**pair,"a":records[pair["i"]],"b":records[pair["j"]],
                     "phase":supplement.phase_verify(images[pair["i"]],images[pair["j"]],result["settings"],pair,window),
                     "exhaustive_original":supplement.exact_verify(images[pair["i"]],images[pair["j"]],result["settings"])})
    unchanged=all(old.sha256(args.dataset/"images"/r["split"]/r["file"])==r["sha256"] for r in records) and old.sha256(args.dataset/"split.json")==split_hash
    if not unchanged:
        raise RuntimeError("Input changed during probe")
    report={"purpose":"Uniform deterministic implementation discrepancy probe; not recall/FPR estimation", "seed":20261103,
            "samples":len(rows),"exhaustive_review_pass":sum(r["exhaustive_original"]["passes_review_threshold"] for r in rows),
            "phase_missed_review_pass":sum(r["exhaustive_original"]["passes_review_threshold"] and not r["phase"]["passes_review_threshold"] for r in rows),
            "max_additional_pixel_ncc_from_exhaustive":max(r["exhaustive_original"]["pixel_ncc"]-r["phase"]["pixel_ncc"] for r in rows),
            "input_manifest_sha256":manifest_hash,"input_images_and_split_unchanged":unchanged,"script_sha256":old.sha256(Path(__file__)),"checks":rows}
    supplement.dump(out/"exhaustive_probe.json",report)
    print({k:v for k,v in report.items() if k!="checks"})


if __name__ == "__main__":
    main()
