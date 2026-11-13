"""Create the Purged Causal Training Split for NEU-DET (Reviewer P0 Issue 1).

This dataset is identical to neu_naive_random in:
- val set (180 images)
- test set (180 images, including 25 leaked and 155 clean)
- test_leaked (25 images)
- test_clean (155 images)

The ONLY change is that the 31 training images that are near-duplicate partners
to the 25 test images are removed from the training set, yielding:
- train_purged: 1,406 images (1,437 - 31).

Evaluating the exact same 25 test images under models trained on neu_naive_random
vs. neu_naive_purged provides a pure within-subject causal measure of leakage.
"""
from pathlib import Path
import json
import csv
import shutil

base = Path(__file__).resolve().parent
naive_dir = base / 'datasets/neu_naive_random'
naive_split = json.loads((naive_dir / 'split.json').read_text(encoding='utf-8'))
edges_csv = base / 'audits/neu_candidate_edges_223_20261101.csv'
out_dataset_dir = base / 'datasets/neu_naive_purged'

leaked_test = set(naive_split['candidate_edges_cross_split_audit']['leaked_test_images'])
assert len(leaked_test) == 25

with edges_csv.open(encoding='utf-8-sig', newline='') as f:
    edges = list(csv.DictReader(f))

train_partners = set()
for e in edges:
    a = Path(e['a_file']).stem
    b = Path(e['b_file']).stem
    if a in leaked_test and b in naive_split['train']:
        train_partners.add(b)
    elif b in leaked_test and a in naive_split['train']:
        train_partners.add(a)

print(f"Identified {len(train_partners)} training images linked to the 25 leaked test images.")
assert len(train_partners) == 31

purged_train = [s for s in naive_split['train'] if s not in train_partners]
assert len(purged_train) == 1437 - 31

if out_dataset_dir.exists():
    shutil.rmtree(out_dataset_dir)

# Create directory structure
for sub in ['train', 'val', 'test', 'test_leaked', 'test_clean']:
    (out_dataset_dir / 'images' / sub).mkdir(parents=True, exist_ok=True)
    (out_dataset_dir / 'labels' / sub).mkdir(parents=True, exist_ok=True)

# Copy val and test sets directly from naive_dir
for sub in ['val', 'test', 'test_leaked', 'test_clean']:
    for f in (naive_dir / 'images' / sub).iterdir():
        if f.is_file() and not f.name.endswith('.cache'):
            shutil.copy2(f, out_dataset_dir / 'images' / sub / f.name)
    for f in (naive_dir / 'labels' / sub).iterdir():
        if f.is_file() and not f.name.endswith('.cache'):
            shutil.copy2(f, out_dataset_dir / 'labels' / sub / f.name)

# Copy only the 1,406 purged train images and labels
for stem in purged_train:
    img_f = naive_dir / 'images/train' / f"{stem}.jpg"
    lbl_f = naive_dir / 'labels/train' / f"{stem}.txt"
    shutil.copy2(img_f, out_dataset_dir / 'images/train' / img_f.name)
    shutil.copy2(lbl_f, out_dataset_dir / 'labels/train' / lbl_f.name)

# Verify counts
assert len(list((out_dataset_dir / 'images/train').glob('*.jpg'))) == 1406
assert len(list((out_dataset_dir / 'labels/train').glob('*.txt'))) == 1406
assert len(list((out_dataset_dir / 'images/test_leaked').glob('*.jpg'))) == 25
assert len(list((out_dataset_dir / 'images/test_clean').glob('*.jpg'))) == 155

# Write YAML files
base_yaml = f"""path: {out_dataset_dir.as_posix()}
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
(out_dataset_dir / 'neu.yaml').write_text(base_yaml, encoding='utf-8')

leaked_yaml = base_yaml.replace('test: images/test', 'test: images/test_leaked')
(out_dataset_dir / 'neu_leaked.yaml').write_text(leaked_yaml, encoding='utf-8')

clean_yaml = base_yaml.replace('test: images/test', 'test: images/test_clean')
(out_dataset_dir / 'neu_clean.yaml').write_text(clean_yaml, encoding='utf-8')

# Write split.json
purged_split_data = {
    "purpose": "Reviewer P0 Issue 1: Causal training ablation (purging 31 partner images to isolate within-subject effect on 25 leaked test images)",
    "baseline_split": "neu_naive_random",
    "counts": {
        "train": len(purged_train),
        "val": len(naive_split['val']),
        "test": len(naive_split['test']),
        "purged_partner_count": len(train_partners)
    },
    "purged_partners": sorted(train_partners),
    "leaked_test_images": sorted(leaked_test),
    "clean_test_images": naive_split['candidate_edges_cross_split_audit']['clean_test_images'],
    "train": purged_train,
    "val": naive_split['val'],
    "test": naive_split['test']
}
(out_dataset_dir / 'split.json').write_text(json.dumps(purged_split_data, indent=2), encoding='utf-8')
print("Successfully built datasets/neu_naive_purged with 1,406 train images.")
