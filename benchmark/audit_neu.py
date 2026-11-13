"""Audit converted NEU-DET split for exact image duplicates and invalid labels."""
from pathlib import Path
import collections, hashlib, json

root = Path(__file__).resolve().parent / "paper-try" / "datasets" / "neu"
manifest = json.loads((root / "split.json").read_text(encoding="utf-8"))
parts = ("train", "val", "test")
sets = {part: set(manifest[part]) for part in parts}
files = collections.defaultdict(list)
errors, boxes = [], collections.Counter()
for part in parts:
    for image in sorted((root / "images" / part).iterdir()):
        if not image.is_file():
            continue
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        label = root / "labels" / part / (image.stem + ".txt")
        files[digest].append({"split": part, "name": image.name, "label": label.read_text(encoding="utf-8").strip() if label.exists() else None})
        if not label.exists():
            errors.append(f"missing label: {label}")
            continue
        for n, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            try:
                fields = line.split()
                cls = int(fields[0])
                vals = [float(v) for v in fields[1:]]
                if len(fields) != 5 or not 0 <= cls < 6 or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    raise ValueError("invalid label")
                boxes[cls] += 1
            except (ValueError, IndexError):
                errors.append(f"invalid label: {label}:{n}")
result = {"counts": {p: len(sets[p]) for p in parts}, "name_overlaps": {f"{a}-{b}": len(sets[a] & sets[b]) for i, a in enumerate(parts) for b in parts[i+1:]}, "boxes_per_class": dict(sorted(boxes.items())), "label_errors": errors, "duplicate_sha256": [{"sha256": digest, "files": entries} for digest, entries in files.items() if len(entries) > 1]}
print(json.dumps(result, ensure_ascii=False, indent=2))
