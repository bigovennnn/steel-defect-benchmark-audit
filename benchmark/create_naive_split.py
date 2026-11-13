"""Generate the Naive Stratified Random Split for NEU-DET (Reviewer D2).

This split uses the exact same 1,797 images and defect class distribution as
neu_v5_frozen (1437 train / 180 val / 180 test), but assigns images by standard
stratified random sampling without grouping near-duplicate candidate pairs.

Quantifies the number of cross-split candidate near-duplicate edges that leak
between train, val, and test, and creates datasets/neu_naive_random on disk.
"""
from pathlib import Path
import json
import csv
import random
import shutil
from collections import defaultdict, Counter

base = Path(__file__).resolve().parent
v5_dir = base / 'datasets' / 'neu_v5_frozen'
v5_split = json.loads((v5_dir / 'split.json').read_text(encoding='utf-8'))
edges_csv = base / 'audits' / 'neu_candidate_edges_223_20261101.csv'
out_split_file = base / 'split_designs' / 'neu_naive_random_split_20261103.json'
out_dataset_dir = base / 'datasets' / 'neu_naive_random'

# Collect all 1,797 stems and their class
all_stems = v5_split['train'] + v5_split['val'] + v5_split['test']
assert len(all_stems) == 1797

# Map each class to its stems
by_class = defaultdict(list)
for s in sorted(all_stems):
    cls = s.rsplit('_', 1)[0]
    by_class[cls].append(s)

# Target counts per class matching v5 exactly
target_val = Counter([s.rsplit('_', 1)[0] for s in v5_split['val']])
target_test = Counter([s.rsplit('_', 1)[0] for s in v5_split['test']])
target_train = Counter([s.rsplit('_', 1)[0] for s in v5_split['train']])

# Perform stratified random sampling with fixed seed 20261103
rng = random.Random(20261103)
naive_val = []
naive_test = []
naive_train = []

for cls, stems in sorted(by_class.items()):
    shuffled = list(sorted(stems))
    rng.shuffle(shuffled)
    n_v = target_val[cls]
    n_t = target_test[cls]
    val_set = shuffled[:n_v]
    test_set = shuffled[n_v:n_v + n_t]
    train_set = shuffled[n_v + n_t:]
    assert len(train_set) == target_train[cls]
    naive_val.extend(val_set)
    naive_test.extend(test_set)
    naive_train.extend(train_set)

naive_val = sorted(naive_val)
naive_test = sorted(naive_test)
naive_train = sorted(naive_train)

assert len(naive_train) == 1437
assert len(naive_val) == 180
assert len(naive_test) == 180
assert set(naive_train).isdisjoint(set(naive_val))
assert set(naive_train).isdisjoint(set(naive_test))
assert set(naive_val).isdisjoint(set(naive_test))

# Map stem to naive partition
part_map = {}
for s in naive_train: part_map[s] = 'train'
for s in naive_val: part_map[s] = 'val'
for s in naive_test: part_map[s] = 'test'

# Audit cross-split edges across the 223 candidates
with edges_csv.open(encoding='utf-8-sig', newline='') as f:
    edges = list(csv.DictReader(f))

edge_counts = defaultdict(int)
cross_split_edges = []
leaked_test_stems = set()
leaked_val_stems = set()

for e in edges:
    a_stem = Path(e['a_file']).stem
    b_stem = Path(e['b_file']).stem
    if a_stem not in part_map or b_stem not in part_map:
        continue
    p_a = part_map[a_stem]
    p_b = part_map[b_stem]
    pair_type = f"{min(p_a, p_b)}/{max(p_a, p_b)}"
    edge_counts[pair_type] += 1
    if p_a != p_b:
        cross_split_edges.append((a_stem, p_a, b_stem, p_b, e['candidate_id'], float(e['pixel_ncc'])))
        if 'train' in (p_a, p_b) and 'test' in (p_a, p_b):
            test_stem = a_stem if p_a == 'test' else b_stem
            leaked_test_stems.add(test_stem)
        if 'train' in (p_a, p_b) and 'val' in (p_a, p_b):
            val_stem = a_stem if p_a == 'val' else b_stem
            leaked_val_stems.add(val_stem)

split_data = {
    "purpose": "Reviewer D2: ablation benchmark measuring the empirical effect of near-duplicate leakage on NEU-DET",
    "seed": 20261103,
    "strategy": "naive stratified random sampling (class-balanced, no group isolation)",
    "counts": {"train": len(naive_train), "val": len(naive_val), "test": len(naive_test)},
    "candidate_edges_cross_split_audit": {
        "total_candidate_edges_audited": len(edges),
        "edge_distribution": dict(sorted(edge_counts.items())),
        "total_cross_split_edges": len(cross_split_edges),
        "train_test_leaking_edges": edge_counts.get("test/train", 0),
        "train_val_leaking_edges": edge_counts.get("train/val", 0),
        "val_test_leaking_edges": edge_counts.get("test/val", 0),
        "leaked_test_image_count": len(leaked_test_stems),
        "leaked_test_images": sorted(leaked_test_stems),
        "clean_test_image_count": len(naive_test) - len(leaked_test_stems),
        "clean_test_images": sorted(set(naive_test) - leaked_test_stems)
    },
    "train": naive_train,
    "val": naive_val,
    "test": naive_test
}

out_split_file.write_text(json.dumps(split_data, indent=2), encoding='utf-8')
print(f"Saved naive split to {out_split_file}")
print(f"Candidate edge breakdown in naive split:")
for k, v in sorted(edge_counts.items()):
    print(f"  {k}: {v}")
print(f"Total cross-split edges: {len(cross_split_edges)} (vs. 0 in neu_v5_frozen)")
print(f"Test images leaked from train: {len(leaked_test_stems)} / 180 ({round(len(leaked_test_stems)/180*100, 1)}%)")
print(f"Clean test images: {len(naive_test) - len(leaked_test_stems)} / 180")

# Now construct datasets/neu_naive_random on disk
if out_dataset_dir.exists():
    shutil.rmtree(out_dataset_dir)

for part in ['train', 'val', 'test']:
    (out_dataset_dir / 'images' / part).mkdir(parents=True, exist_ok=True)
    (out_dataset_dir / 'labels' / part).mkdir(parents=True, exist_ok=True)

# Build a lookup for all images and labels in neu_v5_frozen
source_images = {}
source_labels = {}
for part in ['train', 'val', 'test']:
    for f in (v5_dir / 'images' / part).iterdir():
        source_images[f.stem] = f
    for f in (v5_dir / 'labels' / part).iterdir():
        source_labels[f.stem] = f

for part in ['train', 'val', 'test']:
    for stem in split_data[part]:
        shutil.copy2(source_images[stem], out_dataset_dir / 'images' / part / source_images[stem].name)
        shutil.copy2(source_labels[stem], out_dataset_dir / 'labels' / part / source_labels[stem].name)

yaml_content = f"""path: {out_dataset_dir.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
"""
(out_dataset_dir / 'neu.yaml').write_text(yaml_content, encoding='utf-8')
(out_dataset_dir / 'split.json').write_text(json.dumps(split_data, indent=2), encoding='utf-8')
print(f"Built on-disk dataset at {out_dataset_dir}")
