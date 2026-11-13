#!/usr/bin/env python3
"""
Aggregates all completed runs in `论文2/benchmark/runs_v6/*/*/eval_summary.json` into:
  - `论文2/results/local_v6_3seed_summary_20261111.json`
  - `论文2/results/local_v6_3seed_summary_20261111.csv`
commits them to the local Git repository (`<workspace>`), and pushes the updated
summary files to both GitHub repositories (`bigovennnn/steel-defect-benchmark-audit`
and `bigovennnn/paper-try`) via the GitHub REST API.

If called with `--wait-for-24`, waits until all 24 local priority queue runs complete
before executing the final aggregation, local git commit, and GitHub push.
"""

import argparse
import base64
import csv
import json
import subprocess
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(r"<workspace>")
P2 = ROOT / "论文2"
RUNS_V6 = P2 / "benchmark" / "runs_v6"
OUT_JSON = P2 / "results" / "local_v6_3seed_summary_20261111.json"
OUT_CSV = P2 / "results" / "local_v6_3seed_summary_20261111.csv"


def get_github_token() -> str:
    p = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    creds = dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)
    return creds["password"].strip()


def put_github_file(token: str, repo: str, remote_path: str, local_path: Path, msg: str):
    url = f"https://api.github.com/repos/bigovennnn/{repo}/contents/{urllib.parse.quote(remote_path)}"
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "paper-try-auto-sync",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }
    sha = None
    try:
        req = urllib.request.Request(url, headers=headers)
        sha = json.loads(urllib.request.urlopen(req, timeout=20).read().decode()).get("sha")
    except Exception:
        pass
    data = {
        "message": msg,
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    if sha:
        data["sha"] = sha
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method="PUT",
            )
            urllib.request.urlopen(req, timeout=30)
            print(f"[GITHUB API SYNCED] {repo}:{remote_path}")
            return
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def collect_and_aggregate() -> tuple[int, dict]:
    files = sorted(RUNS_V6.rglob("eval_summary.json")) if RUNS_V6.exists() else []
    records = []
    grouped = defaultdict(list)

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rec = {
            "suite": d["suite"],
            "arch": d["arch"],
            "seed": int(d["seed"]),
            "split_name": d["split_name"],
            "split_design_sha256": d["split_design_sha256"],
            "pretrained_tensors_transferred": d.get("pretrained_tensors_transferred"),
            "val_precision": round(float(d["val"]["precision"]) * 100, 4),
            "val_recall": round(float(d["val"]["recall"]) * 100, 4),
            "val_map50": round(float(d["val"]["map50"]) * 100, 4),
            "val_map50_95": round(float(d["val"]["map50_95"]) * 100, 4),
            "test_precision": round(float(d["test"]["precision"]) * 100, 4),
            "test_recall": round(float(d["test"]["recall"]) * 100, 4),
            "test_map50": round(float(d["test"]["map50"]) * 100, 4),
            "test_map50_95": round(float(d["test"]["map50_95"]) * 100, 4),
            "train_seconds": round(float(d.get("train_seconds", 0.0)), 1),
            "rel_path": str(f.relative_to(RUNS_V6)).replace("\\", "/"),
            "full_path": f,
        }
        records.append(rec)
        grouped[(d["suite"], d["arch"], d["split_name"])].append(rec)

    records.sort(key=lambda r: (r["suite"], r["arch"], r["seed"]))

    agg_summary = []
    for (suite, arch, split_name), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["seed"])
        val50 = np.array([r["val_map50"] for r in rows], dtype=float)
        test50 = np.array([r["test_map50"] for r in rows], dtype=float)
        test50_95 = np.array([r["test_map50_95"] for r in rows], dtype=float)
        agg_summary.append(
            {
                "suite": suite,
                "arch": arch,
                "split_name": split_name,
                "n_seeds": len(rows),
                "seeds": [r["seed"] for r in rows],
                "val_map50_mean": round(float(np.mean(val50)), 2),
                "val_map50_sd": round(float(np.std(val50, ddof=1)), 2) if len(val50) > 1 else 0.0,
                "val_map50_range": round(float(np.max(val50) - np.min(val50)), 2),
                "test_map50_mean": round(float(np.mean(test50)), 2),
                "test_map50_sd": round(float(np.std(test50, ddof=1)), 2) if len(test50) > 1 else 0.0,
                "test_map50_range": round(float(np.max(test50) - np.min(test50)), 2),
                "test_map50_95_mean": round(float(np.mean(test50_95)), 2),
                "test_map50_95_sd": round(float(np.std(test50_95, ddof=1)), 2) if len(test50_95) > 1 else 0.0,
                "per_seed_test_map50": {str(r["seed"]): round(r["test_map50"], 2) for r in rows},
                "per_seed_val_map50": {str(r["seed"]): round(r["val_map50"], 2) for r in rows},
            }
        )

    csv_rows = [{k: v for k, v in r.items() if k not in {"rel_path", "full_path"}} for r in records]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if csv_rows:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    payload = {
        "updated_at": datetime.now().isoformat(),
        "total_completed_runs": len(records),
        "target_local_queue_runs": 24,
        "aggregated_by_model": agg_summary,
        "individual_runs": csv_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(records), {"records": records, "payload": payload}


def sync_git_and_github(n_runs: int, records: list[dict]):
    # 1. Local git commit
    subprocess.run(
        [
            "git",
            "add",
            str(OUT_JSON.relative_to(ROOT)),
            str(OUT_CSV.relative_to(ROOT)),
            "论文2/scripts/finalize_local_queue_and_sync.py",
        ],
        cwd=ROOT,
        check=True,
    )
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode("utf-8").strip()
    if status:
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"Update local GPU v6 benchmark results ({n_runs}/24 runs completed)",
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"[LOCAL GIT COMMITTED] ({n_runs}/24 runs)")

    # 2. Push to both GitHub repos via REST API
    token = get_github_token()
    msg = f"Update local GPU v6 benchmark results ({n_runs}/24 runs completed)"
    put_github_file(
        token,
        "steel-defect-benchmark-audit",
        "results/local_v6_3seed_summary_20261111.json",
        OUT_JSON,
        msg,
    )
    put_github_file(
        token,
        "steel-defect-benchmark-audit",
        "results/local_v6_3seed_summary_20261111.csv",
        OUT_CSV,
        msg,
    )
    put_github_file(
        token,
        "paper-try",
        "论文2/results/local_v6_3seed_summary_20261111.json",
        OUT_JSON,
        msg,
    )
    put_github_file(
        token,
        "paper-try",
        "论文2/results/local_v6_3seed_summary_20261111.csv",
        OUT_CSV,
        msg,
    )

    # Also sync individual eval_summary.json files that aren't pushed yet
    for r in records:
        rel = r["rel_path"]
        fp = r["full_path"]
        put_github_file(
            token,
            "steel-defect-benchmark-audit",
            f"results/runs_v6/{rel}",
            fp,
            f"Sync {rel}",
        )
        put_github_file(
            token,
            "paper-try",
            f"论文2/benchmark/runs_v6/{rel}",
            fp,
            f"Sync {rel}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-24", action="store_true", help="Wait until all 24 runs exist before final sync")
    args = parser.parse_args()

    if args.wait_for_24:
        last_count = -1
        while True:
            n_runs, data = collect_and_aggregate()
            if n_runs != last_count:
                print(f"[{datetime.now().isoformat()}] Progress: {n_runs}/24 runs completed. Syncing snapshot...")
                try:
                    sync_git_and_github(n_runs, data["records"])
                except Exception as e:
                    print(f"[WARN] Snapshot sync error: {e}")
                last_count = n_runs
            if n_runs >= 24:
                print(f"[{datetime.now().isoformat()}] All 24/24 runs completed and synced to local Git + GitHub!")
                break
            time.sleep(60)
    else:
        n_runs, data = collect_and_aggregate()
        sync_git_and_github(n_runs, data["records"])
        print(f"Synced {n_runs}/24 runs.")


if __name__ == "__main__":
    main()
