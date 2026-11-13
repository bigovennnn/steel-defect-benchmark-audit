"""Create an auditable exploratory NEU split without the known duplicate.

This is an exploratory split, not a preregistered final test set. Never modify the source.
"""
import hashlib, json, shutil
from pathlib import Path

base = Path(__file__).resolve().parent
source = base / "paper-try" / "datasets" / "neu"
target = base / "datasets" / "neu_dedup_v1"
if target.exists():
    raise SystemExit("Target exists; refuse to overwrite: " + str(target))
manifest = json.loads((source / "split.json").read_text(encoding="utf-8"))
pair = [("train", "patches_105"), ("val", "patches_101")]
images = [source / "images" / split / (stem + ".jpg") for split, stem in pair]
digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in images]
if digests[0] != digests[1] or any(stem not in manifest[split] for split, stem in pair):
    raise SystemExit("Unexpected source data or split; stopped before writing.")
for split in ("train", "val", "test"):
    for kind in ("images", "labels"):
        (target / kind / split).mkdir(parents=True, exist_ok=True)
    for stem in manifest[split]:
        if split == "train" and stem == "patches_105":
            continue
        image = source / "images" / split / (stem + ".jpg")
        label = source / "labels" / split / (stem + ".txt")
        shutil.copy2(image, target / "images" / split / image.name)
        shutil.copy2(label, target / "labels" / split / label.name)
    manifest[split] = [s for s in manifest[split] if not (split == "train" and s == "patches_105")]
manifest["provenance"] = {"source": "paper-try/datasets/neu/split.json", "removed": "train/patches_105", "retained": "val/patches_101", "image_sha256": digests[0], "note": "Exploratory baseline. Annotation choice is not a ground-truth adjudication."}
(target / "split.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
yaml = "path: " + target.as_posix() + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
(target / "neu.yaml").write_text(yaml, encoding="utf-8")
print("created", target, {p: len(manifest[p]) for p in ("train", "val", "test")})
