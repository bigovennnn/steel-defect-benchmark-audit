"""Small independent brute-force check of the supplemental FFT alignment."""
from pathlib import Path
import argparse
import numpy as np
import audit_neu_rigid_candidates as old
import audit_neu_rigid_supplement as supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().drive.lower() != "d:":
        raise SystemExit("D: output required")
    settings = {"max_shift_fraction": .25, "min_overlap": .6,
                "pixel_ncc_threshold": .95, "gradient_ncc_threshold": .9}
    rng = np.random.default_rng(20261101)
    shifts, _, _ = supplement.exact_geometry(17, .25, .6)
    tests = []
    for case in range(5):
        a = rng.integers(0, 256, (17, 17), dtype=np.uint8)
        b = rng.integers(0, 256, (17, 17), dtype=np.uint8)
        exact = supplement.exact_verify(a, b, settings)
        brute = max((old.ncc(*old.overlap_regions(a, np.ascontiguousarray(np.rot90(b, k)), dx, dy)), k, dx, dy)
                    for k in range(4) for dx, dy in shifts)
        tests.append({"case": case, "absolute_ncc_error": abs(exact["pixel_ncc"] - brute[0]),
                      "exhaustive_fft": exact, "brute_best": brute})
    result = {"purpose": "17x17 brute-force NCC at every eligible integer shift and quarter turn",
              "seed": 20261101, "tests": tests,
              "max_ncc_error": max(x["absolute_ncc_error"] for x in tests)}
    supplement.dump(args.out.resolve(), result)
    if result["max_ncc_error"] > 1e-10:
        raise SystemExit("Exhaustive FFT disagrees with brute force")
    print("FFT/brute-force NCC max error", result["max_ncc_error"])


if __name__ == "__main__":
    main()
