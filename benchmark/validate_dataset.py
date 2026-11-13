"""Validate dataset manifest, normalized boxes and exact decoded-pixel duplication."""
import argparse, collections, hashlib, json
from pathlib import Path
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument("data_dir",type=Path);ap.add_argument("--classes",type=int,required=True);ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
root=args.data_dir.resolve();manifest=json.loads((root/"split.json").read_text(encoding="utf-8"));parts=("train","val","test")
seen_names={}; pixels=collections.defaultdict(list);counts=collections.Counter();errors=[];counts_img={}; counts_empty={}
for part in parts:
 images=sorted((root/"images"/part).glob("*.jpg")); names={x.stem for x in images}; expected=set(manifest[part]);counts_img[part]=len(images)
 if names!=expected:errors.append({"type":"manifest_mismatch","split":part,"missing_images":sorted(expected-names),"extra_images":sorted(names-expected)})
 labels={x.stem for x in (root/"labels"/part).glob("*.txt")}
 if labels!=names:errors.append({"type":"label_mismatch","split":part,"missing_labels":sorted(names-labels),"orphan_labels":sorted(labels-names)})
 counts_empty[part]=0
 for image in images:
  if image.stem in seen_names: errors.append({"type":"duplicate_name","name":image.stem,"splits":[seen_names[image.stem],part]})
  seen_names[image.stem]=part
  with Image.open(image) as im:
   gray=im.convert("L");h=hashlib.sha256((str(gray.size)+":").encode()+gray.tobytes()).hexdigest()
  pixels[h].append({"split":part,"file":image.name})
  label=root/"labels"/part/(image.stem+".txt")
  if not label.exists():continue
  lines=label.read_text(encoding="utf-8").splitlines()
  if not lines:counts_empty[part]+=1
  for lineno,line in enumerate(lines,1):
   try:
    fields=line.split();assert len(fields)==5
    cls=int(fields[0]);x,y,w,h=map(float,fields[1:]);assert 0<=cls<args.classes and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
    assert x-w/2>=-1e-5 and x+w/2<=1+1e-5 and y-h/2>=-1e-5 and y+h/2<=1+1e-5
    counts[cls]+=1
   except (ValueError,AssertionError):errors.append({"type":"invalid_box","label":str(label),"line":lineno,"value":line})
dups=[{"pixel_sha256":k,"files":v} for k,v in pixels.items() if len(v)>1]
result={"dataset":str(root),"image_counts":counts_img,"empty_labels":counts_empty,"box_counts":dict(sorted(counts.items())),"errors":errors,"pixel_duplicate_groups":dups,"cross_split_duplicate_groups":[x for x in dups if len({e["split"] for e in x["files"]})>1],"limitations":"Exact decoded grayscale pixels only; no near-duplicate or semantic annotation validation."}
args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({k:result[k] for k in ("image_counts","empty_labels","box_counts")},ensure_ascii=False));print("errors",len(errors),"duplicate groups",len(dups),"cross-split",len(result["cross_split_duplicate_groups"]))
