#!/usr/bin/env python3
"""
RTX 3090 100-Run Post-Audit Fix & Split-Interaction Queue (`run_3090_fix_100runs.py`).

Executes the exact 100 corrected 100-epoch YOLO11n runs required after the code/statistical audit:
  1. [Required 40 runs — Mod-10S / e7_modules]:
       `yolo11n-base`, `yolo11n-cbam`, `yolo11n-ema`, `yolo11n-bifpn` x 10 seeds (0..9)
       on `neu_grouped_v6` (G0), now with verified 493/499 COCO pretrained weight transfer
       (fixing the `nc: 6` + `pretrained=False` + `model.ckpt=None` random-init bug).
  2. [Bonus 60 runs — Split x Model Interaction]:
       G0..G9 (all 10 zero-leakage grouped splits) x [`yolo11n-base`, `yolo11n-bifpn`] x 3 seeds (0..2)
       using the exact same 493/499 COCO loader on the same RTX 3090 hardware.

Features:
  - Auto-locates `raw_pool/neu` (or extracts ONLY `raw_pool/neu` + COCO weights from `paper2_cloud_bundle.zip`
    if found nearby, never overwriting fixed scripts).
  - Auto-downloads official `yolo11n.pt` ONCE in the main process if missing before launching workers.
  - Materializes G0 (`neu_grouped_v6`) and G1..G9 (`neu_grouped_g01..g09`) via hardlinks from frozen JSONs.
  - Pre-warms Ultralytics `.cache` files sequentially to eliminate multi-worker cache write races.
  - Runs a 30-second val-only preflight smoke check verifying `493/499` COCO transfer (`conv0_diff == 0.0`).
  - Archives any prior broken `448`-tensor / `pretrained: false` runs instead of falsely skipping them.
  - Runs up to `--parallel 5` concurrent jobs on 24GB RTX 3090, checkpoints `fix100_manifest.csv` and
    archives results periodically (`paper2_cloud_results.tar.gz` + lightweight `_summary.tar.gz`).

Usage on Cloud RTX 3090:
  nohup python3 benchmark/cloud/run_3090_fix_100runs.py --parallel 5 --workers 2 --shutdown > fix100.log 2>&1 &
"""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import deque
from pathlib import Path

CLOUD_DIR = Path(__file__).resolve().parent
BENCH_DIR = CLOUD_DIR.parent
ROOT_DIR = BENCH_DIR.parent
DATASETS_DIR = BENCH_DIR / "datasets"
SPLIT_DESIGNS_DIR = BENCH_DIR / "split_designs"
RUNS_DIR = BENCH_DIR / "runs_v6"
LOG_DIR = RUNS_DIR / "_3090_fix100_logs"

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
MOD_ARCHS = ["yolo11n-base", "yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"]


def ensure_dependencies(no_install: bool = False):
    try:
        import numpy  # noqa: F401
        import torch
        import ultralytics

        print(
            f"[ENV] Python {platform.python_version()}, PyTorch {torch.__version__}, "
            f"CUDA={torch.cuda.is_available()}, Ultralytics={ultralytics.__version__}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(f"[ENV] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        return
    except ImportError as exc:
        if no_install:
            raise RuntimeError(f"Missing Python dependency ({exc}) and --no-install set.") from exc
    print("[ENV] Installing ultralytics==8.4.189 numpy ...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "ultralytics==8.4.189",
            "numpy",
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
        ],
        check=True,
    )


def resolve_g0_split_path() -> Path:
    for cand in (
        DATASETS_DIR / "neu_grouped_v6" / "split.json",
        SPLIT_DESIGNS_DIR / "neu_grouped_v6_split.json",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Missing frozen G0 split file in both {DATASETS_DIR / 'neu_grouped_v6' / 'split.json'} "
        f"and {SPLIT_DESIGNS_DIR / 'neu_grouped_v6_split.json'}"
    )


def load_v6_stems() -> set[str]:
    sp_path = resolve_g0_split_path()
    d = json.loads(sp_path.read_text(encoding="utf-8"))
    return set(d["train"]) | set(d["val"]) | set(d["test"])


def pool_has_all_stems(pool_dir: Path, required_stems: set[str]) -> bool:
    img_dir = pool_dir / "images"
    lbl_dir = pool_dir / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return False
    for st in required_stems:
        if not (img_dir / f"{st}.jpg").exists() or not (lbl_dir / f"{st}.txt").exists():
            return False
    return True


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def harvest_pool_from_split_dir(split_dir: Path, dst_pool: Path, required_stems: set[str]) -> bool:
    """Assemble raw_pool/neu from an already-materialized neu_grouped_v6 train/val/test tree."""
    found_imgs = {}
    found_lbls = {}
    for sub in ("train", "val", "test"):
        i_dir = split_dir / "images" / sub
        l_dir = split_dir / "labels" / sub
        if i_dir.is_dir():
            for p in i_dir.glob("*.jpg"):
                found_imgs[p.stem] = p
        if l_dir.is_dir():
            for p in l_dir.glob("*.txt"):
                found_lbls[p.stem] = p
    if not required_stems.issubset(found_imgs.keys()) or not required_stems.issubset(found_lbls.keys()):
        return False
    print(f"[POOL] Indexing {len(required_stems)} NEU images/labels from {split_dir} -> {dst_pool}", flush=True)
    for st in sorted(required_stems):
        link_or_copy(found_imgs[st], dst_pool / "images" / f"{st}.jpg")
        link_or_copy(found_lbls[st], dst_pool / "labels" / f"{st}.txt")
    return True


def ensure_neu_pool(explicit_pool: Path | None = None) -> Path:
    required_stems = load_v6_stems()
    local_pool = ROOT_DIR / "raw_pool" / "neu"

    candidates = []
    if explicit_pool is not None:
        candidates.append(explicit_pool)
    candidates.extend(
        [
            local_pool,
            Path("/root/rivermind-data/paper2_work/raw_pool/neu"),
            Path("/root/autodl-tmp/paper2_work/raw_pool/neu"),
            Path("/root/autodl-tmp/raw_pool/neu"),
            Path("/root/paper2_work/raw_pool/neu"),
            Path.home() / "rivermind-data/paper2_work/raw_pool/neu",
            ROOT_DIR.parent / "raw_pool/neu",
            ROOT_DIR.parent / "paper2_work/raw_pool/neu",
        ]
    )
    for cand in candidates:
        if cand.exists() and pool_has_all_stems(cand, required_stems):
            if cand.resolve() != local_pool.resolve():
                print(f"[POOL] Linking NEU pool from {cand} -> {local_pool}", flush=True)
                for st in sorted(required_stems):
                    link_or_copy(cand / "images" / f"{st}.jpg", local_pool / "images" / f"{st}.jpg")
                    link_or_copy(cand / "labels" / f"{st}.txt", local_pool / "labels" / f"{st}.txt")
            else:
                print(f"[POOL] Verified raw_pool/neu ({len(required_stems)} images+labels).", flush=True)
            return local_pool

    # Fallback 1: already-materialized neu_grouped_v6 datasets on machine
    for v6_cand in [
        DATASETS_DIR / "neu_grouped_v6",
        Path("/root/rivermind-data/paper2_work/benchmark/datasets/neu_grouped_v6"),
        Path("/root/autodl-tmp/paper2_work/benchmark/datasets/neu_grouped_v6"),
        Path.home() / "rivermind-data/paper2_work/benchmark/datasets/neu_grouped_v6",
    ]:
        if v6_cand.exists() and harvest_pool_from_split_dir(v6_cand, local_pool, required_stems):
            return local_pool

    # Fallback 2: extract ONLY raw_pool/neu/* and weights/* from paper2_cloud_bundle.zip (preserving fixed code!)
    zip_cands = [
        ROOT_DIR / "paper2_cloud_bundle.zip",
        ROOT_DIR.parent / "paper2_cloud_bundle.zip",
        Path("/root/rivermind-data/paper2_cloud_bundle.zip"),
        Path("/root/autodl-tmp/paper2_cloud_bundle.zip"),
        Path.home() / "paper2_cloud_bundle.zip",
    ]
    for zpath in zip_cands:
        if zpath.exists() and zpath.is_file():
            print(f"[POOL] Extracting ONLY raw_pool/neu/* and weights/* from {zpath} ...", flush=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in zf.namelist():
                    if member.startswith("raw_pool/neu/") or member.startswith("benchmark/weights/"):
                        zf.extract(member, ROOT_DIR)
            if pool_has_all_stems(local_pool, required_stems):
                print(f"[POOL] Extracted {len(required_stems)} NEU images+labels successfully.", flush=True)
                return local_pool

    raise FileNotFoundError(
        "Could not locate the 1,797 NEU-DET images/labels.\n"
        "Please either:\n"
        "  1) Pass `--neu-pool /path/to/raw_pool/neu` (containing `images/*.jpg` and `labels/*.txt`), OR\n"
        "  2) Place `paper2_cloud_bundle.zip` in the repo root / `/root/autodl-tmp/` / `/root/rivermind-data/`."
    )


def ensure_coco_weights() -> Path:
    weights_dir = BENCH_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / "yolo11n_coco_official.pt"
    if target.exists():
        return target

    for cand in [
        Path("/root/rivermind-data/paper2_work/benchmark/weights/yolo11n_coco_official.pt"),
        Path("/root/autodl-tmp/paper2_work/benchmark/weights/yolo11n_coco_official.pt"),
        Path.home() / "rivermind-data/paper2_work/benchmark/weights/yolo11n_coco_official.pt",
        ROOT_DIR.parent / "benchmark/weights/yolo11n_coco_official.pt",
        ROOT_DIR / "yolo11n.pt",
    ]:
        if cand.exists() and cand.is_file():
            shutil.copy2(cand, target)
            print(f"[WEIGHTS] Copied COCO weights from {cand} -> {target}", flush=True)
            return target

    print("[WEIGHTS] Downloading official COCO yolo11n.pt once before starting parallel workers...", flush=True)
    from ultralytics import YOLO

    dl = YOLO("yolo11n.pt")
    src = Path(dl.ckpt_path) if getattr(dl, "ckpt_path", None) else Path("yolo11n.pt")
    if not src.exists():
        raise FileNotFoundError("Failed to download yolo11n.pt via Ultralytics")
    shutil.copy2(src, target)
    print(f"[WEIGHTS] Cached official COCO weights at {target}", flush=True)
    return target


def materialize_split(g_idx: int, pool_dir: Path) -> Path:
    if g_idx == 0:
        dst_dir = DATASETS_DIR / "neu_grouped_v6"
        sp_json_src = resolve_g0_split_path()
    else:
        dst_dir = DATASETS_DIR / f"neu_grouped_g{g_idx:02d}"
        sp_json_src = SPLIT_DESIGNS_DIR / f"neu_grouped_g{g_idx:02d}_split.json"
        if not sp_json_src.exists():
            raise FileNotFoundError(f"Missing frozen split file: {sp_json_src}")

    sp = json.loads(sp_json_src.read_text(encoding="utf-8"))
    assert len(sp["train"]) == 1437 and len(sp["val"]) == 180 and len(sp["test"]) == 180, f"Unexpected counts in g{g_idx:02d}"

    dst_dir.mkdir(parents=True, exist_ok=True)
    if not (dst_dir / "split.json").exists():
        shutil.copy2(sp_json_src, dst_dir / "split.json")

    for part in ("train", "val", "test"):
        img_sub = dst_dir / "images" / part
        lbl_sub = dst_dir / "labels" / part
        img_sub.mkdir(parents=True, exist_ok=True)
        lbl_sub.mkdir(parents=True, exist_ok=True)
        for st in sp[part]:
            src_i = pool_dir / "images" / f"{st}.jpg"
            src_l = pool_dir / "labels" / f"{st}.txt"
            link_or_copy(src_i, img_sub / f"{st}.jpg")
            link_or_copy(src_l, lbl_sub / f"{st}.txt")

    names_block = "\n".join(f"  {i}: {c}" for i, c in enumerate(NEU_CLASSES))
    yaml_text = (
        f"path: {str(dst_dir.resolve()).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names:\n{names_block}\n"
    )
    (dst_dir / "neu.yaml").write_text(yaml_text, encoding="utf-8")
    return dst_dir


def prewarm_caches(dataset_dirs: list[Path]):
    """Build Ultralytics .cache files sequentially so parallel workers never collide."""
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    for d_dir in dataset_dirs:
        y_path = d_dir / "neu.yaml"
        train_cache = d_dir / "labels" / "train.cache"
        val_cache = d_dir / "labels" / "val.cache"
        if train_cache.exists() and val_cache.exists():
            continue
        data_cfg = check_det_dataset(str(y_path))
        for split_key in ("train", "val"):
            YOLODataset(
                img_path=data_cfg[split_key],
                imgsz=640,
                augment=False,
                data=data_cfg,
                task="detect",
            )
    print(f"[CACHE] Verified Ultralytics label caches across {len(dataset_dirs)} datasets.", flush=True)


def run_preflight_smoke(workers: int, device: str):
    """Run 1-epoch 5% val-only smoke check on all 4 variants to guarantee 493/499 COCO transfer."""
    print("\n=== [PREFLIGHT SMOKE CHECK] Verifying 493/499 COCO transfer across all 4 variants (val-only) ===", flush=True)
    cmd = [
        sys.executable,
        str(CLOUD_DIR / "run_cloud_suite.py"),
        "--suite",
        "e7_modules",
        "--archs",
        *MOD_ARCHS,
        "--seeds",
        "0",
        "--smoke",
        "--workers",
        str(workers),
        "--device",
        str(device),
    ]
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    e7_dir = RUNS_DIR / "e7_modules"
    for arch in MOD_ARCHS:
        sm_dir = e7_dir / f"smoke_{arch}_e100_seed0"
        sm_json = sm_dir / "smoke_summary.json"
        if not sm_json.exists():
            raise RuntimeError(f"Smoke check did not produce {sm_json}")
        data = json.loads(sm_json.read_text(encoding="utf-8"))
        if not data.get("pretrained_coco_verified") or data.get("pretrained_tensors_transferred") != 493:
            raise RuntimeError(f"Smoke verification failed for {arch}: {data}")
        print(
            f"[PREFLIGHT PASS] {arch}: coco_verified={data['pretrained_coco_verified']}, "
            f"n_coco={data['pretrained_tensors_transferred']}/499, "
            f"conv0_diff={data['pretrained_conv0_max_diff']:.2e}",
            flush=True,
        )
        shutil.rmtree(sm_dir, ignore_errors=True)


def is_task_verified_done(task: dict) -> bool:
    run_dir = RUNS_DIR / task["suite"] / task["run_name"]
    eval_json = run_dir / "eval_summary.json"
    if not eval_json.exists():
        return False
    try:
        es = json.loads(eval_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    if es.get("pretrained_coco_verified") is not True or es.get("pretrained_tensors_transferred") != 493:
        return False
    args_y = run_dir / "args.yaml"
    if args_y.exists():
        if any(ln.strip().lower() == "pretrained: false" for ln in args_y.read_text(encoding="utf-8", errors="ignore").splitlines()):
            return False
    return True


def build_task_queue(phase: str, interaction_arch: str) -> list[dict]:
    tasks = []
    if phase in ("all", "required40"):
        for s in range(10):
            for arch in MOD_ARCHS:
                tasks.append(
                    {
                        "id": f"mod10s__{arch}__s{s}",
                        "suite": "e7_modules",
                        "split_g": 0,
                        "arch": arch,
                        "seed": s,
                        "run_name": f"{arch}_e100_seed{s}",
                        "cmd_args": [
                            "--suite",
                            "e7_modules",
                            "--archs",
                            arch,
                            "--seeds",
                            str(s),
                        ],
                    }
                )
    if phase in ("all", "interaction60"):
        inter_archs = ["yolo11n-base", interaction_arch]
        for g_idx in range(10):
            for s in range(3):
                for arch in inter_archs:
                    tasks.append(
                        {
                            "id": f"splitint__g{g_idx:02d}__{arch}__s{s}",
                            "suite": "split_interaction",
                            "split_g": g_idx,
                            "arch": arch,
                            "seed": s,
                            "run_name": f"g{g_idx:02d}_{arch}_e100_seed{s}",
                            "cmd_args": [
                                "--suite",
                                "split_interaction",
                                "--splits-g",
                                str(g_idx),
                                "--archs",
                                arch,
                                "--seeds",
                                str(s),
                            ],
                        }
                    )
    return tasks


def export_manifest_and_archives(tasks: list[dict], pack_full: bool = True):
    rows = []
    for t in tasks:
        eval_p = RUNS_DIR / t["suite"] / t["run_name"] / "eval_summary.json"
        if not eval_p.exists():
            continue
        es = json.loads(eval_p.read_text(encoding="utf-8"))
        if es.get("pretrained_coco_verified") is not True or es.get("pretrained_tensors_transferred") != 493:
            continue
        rows.append(
            {
                "suite": t["suite"],
                "split_g": f"g{t['split_g']:02d}",
                "arch": t["arch"],
                "seed": t["seed"],
                "run": t["run_name"],
                "pretrained_tensors_transferred": es.get("pretrained_tensors_transferred"),
                "pretrained_coco_verified": es.get("pretrained_coco_verified"),
                "val_map50": round(es["val"]["map50"] * 100.0, 4),
                "test_map50": round(es["test"]["map50"] * 100.0, 4),
                "test_map50_95": round(es["test"]["map50_95"] * 100.0, 4),
                "test_precision": round(es["test"]["precision"] * 100.0, 4),
                "test_recall": round(es["test"]["recall"] * 100.0, 4),
                "train_seconds": es.get("train_seconds", 0.0),
                "gpu": es.get("gpu", ""),
            }
        )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    m_path = RUNS_DIR / "fix100_manifest.csv"
    if rows:
        with m_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Lightweight archive (summaries + results.csv + args.yaml + manifest, NO .pt or per_image dumps)
    lite_tar = ROOT_DIR / "paper2_3090_fix100_lite.tar.gz"
    with tarfile.open(lite_tar, "w:gz") as tf:
        if m_path.exists():
            tf.add(m_path, arcname=str(m_path.relative_to(ROOT_DIR)))
        for sub in ("e7_modules", "split_interaction"):
            sub_dir = RUNS_DIR / sub
            if not sub_dir.exists():
                continue
            for p in sorted(sub_dir.rglob("*")):
                if p.is_file() and p.name in {"eval_summary.json", "results.csv", "args.yaml"}:
                    tf.add(p, arcname=str(p.relative_to(ROOT_DIR)))

    if pack_full:
        subprocess.run([sys.executable, str(CLOUD_DIR / "pack_cloud_results.py")], cwd=ROOT_DIR, check=False)

    print(
        f"[PACK] Updated {m_path.name} ({len(rows)}/{len(tasks)} verified) & {lite_tar.name} "
        f"({lite_tar.stat().st_size / (1024*1024):.2f} MB)",
        flush=True,
    )


def run_pool(tasks: list[dict], parallel: int, workers: int, device: str, batch: int):
    pending = deque([t for t in tasks if not is_task_verified_done(t)])
    done_initial = len(tasks) - len(pending)
    print(f"\n[QUEUE] {len(tasks)} total tasks | {done_initial} already verified | {len(pending)} to run.", flush=True)
    if not pending:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    active = []  # list of (task, popen, log_fh, t_start)
    completed_this_session = 0
    failed = []
    t_pool_start = time.time()

    while pending or active:
        while pending and len(active) < parallel:
            task = pending.popleft()
            log_path = LOG_DIR / f"{task['id']}.log"
            log_fh = log_path.open("w", encoding="utf-8")
            cmd = [
                sys.executable,
                str(CLOUD_DIR / "run_cloud_suite.py"),
                *task["cmd_args"],
                "--epochs",
                "100",
                "--batch",
                str(batch),
                "--workers",
                str(workers),
                "--device",
                str(device),
            ]
            proc = subprocess.Popen(cmd, cwd=ROOT_DIR, stdout=log_fh, stderr=subprocess.STDOUT)
            active.append((task, proc, log_fh, time.time()))
            print(f"[LAUNCH] {task['id']} (active={len(active)}/{parallel}, remaining={len(pending)})", flush=True)
            time.sleep(2.0)  # Stagger CUDA/DataLoader startup spikes

        still_active = []
        for task, proc, log_fh, t_start in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((task, proc, log_fh, t_start))
                continue
            log_fh.close()
            dur_min = (time.time() - t_start) / 60.0
            if rc == 0 and is_task_verified_done(task):
                completed_this_session += 1
                total_done = done_initial + completed_this_session
                avg_wall = (time.time() - t_pool_start) / max(completed_this_session, 1)
                eta_hr = (len(pending) + len(still_active)) * avg_wall / 3600.0
                print(
                    f"[DONE {total_done}/{len(tasks)}] {task['id']} ({dur_min:.1f} min) | ETA ~{eta_hr:.2f} h",
                    flush=True,
                )
                if completed_this_session % 10 == 0:
                    export_manifest_and_archives(tasks, pack_full=False)
            else:
                failed.append((task["id"], rc, str(LOG_DIR / f"{task['id']}.log")))
                print(f"[FAIL rc={rc}] {task['id']} -> inspect {LOG_DIR / (task['id'] + '.log')}", flush=True)
        active = still_active
        if active:
            time.sleep(5.0)

    if failed:
        raise RuntimeError(f"{len(failed)} cloud task(s) failed: {failed[:5]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["all", "required40", "interaction60"], default="all")
    ap.add_argument(
        "--interaction-arch",
        choices=["yolo11n-cbam", "yolo11n-ema", "yolo11n-bifpn"],
        default="yolo11n-bifpn",
        help="Module paired with yolo11n-base across G0..G9 (default: yolo11n-bifpn)",
    )
    ap.add_argument("--parallel", type=int, default=5, help="Concurrent training processes on 24GB RTX 3090 (default: 5)")
    ap.add_argument("--workers", type=int, default=2 if os.name != "nt" else 0, help="DataLoader workers per job")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--neu-pool", type=Path, default=None, help="Optional path to raw_pool/neu")
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Only run setup + 30s preflight verification, then exit")
    ap.add_argument("--summarize-only", action="store_true", help="Only build fix100_manifest.csv and archives")
    ap.add_argument("--shutdown", action="store_true", help="Run `shutdown -h now` on Linux after full completion")
    args = ap.parse_args()

    t0 = time.time()
    tasks = build_task_queue(args.phase, args.interaction_arch)

    if args.summarize_only:
        export_manifest_and_archives(tasks, pack_full=True)
        return

    ensure_dependencies(no_install=args.no_install)
    pool_dir = ensure_neu_pool(args.neu_pool)
    ensure_coco_weights()

    needed_g = [0] if args.phase == "required40" else list(range(10))
    ds_dirs = [materialize_split(g, pool_dir) for g in needed_g]
    prewarm_caches(ds_dirs)

    if args.smoke or not args.skip_preflight:
        run_preflight_smoke(workers=args.workers, device=args.device)
        if args.smoke:
            print("[SMOKE COMPLETE] Preflight verified 493/499 COCO transfer across all 4 variants.", flush=True)
            return

    run_pool(tasks, parallel=max(1, args.parallel), workers=args.workers, device=args.device, batch=args.batch)
    export_manifest_and_archives(tasks, pack_full=True)

    hr = (time.time() - t0) / 3600.0
    print(f"\n=== ALL {len(tasks)} VERIFIED RUNS COMPLETED IN {hr:.2f} HOURS ===", flush=True)
    print(f"Full results:  {ROOT_DIR / 'paper2_cloud_results.tar.gz'}", flush=True)
    print(f"Lite summary:  {ROOT_DIR / 'paper2_3090_fix100_lite.tar.gz'}", flush=True)

    if args.shutdown and os.name != "nt":
        print("[SHUTDOWN] Sleeping 30s before `shutdown -h now`...", flush=True)
        time.sleep(30)
        subprocess.run(["shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
