"""Find perceptually similar pairs; low hash distance is not proof of leakage."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
ap=argparse.ArgumentParser();ap.add_argument("dataset",type=Path);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--max-distance",type=int,default=4);args=ap.parse_args()
items=[]
for part in ("train","val","test"):
 for path in sorted((args.dataset/"images"/part).glob("*.jpg")):
  with Image.open(path) as image:
   a=np.asarray(image.convert("L").resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
  d=cv2.dct(a)[:8,:8].flatten();v=d[1:];median=float(np.median(v));h=0
  for bit in v>median:h=(h<<1)|int(bit)
  items.append((part,path.name,h))
pairs=[]
for i,(part,name,h) in enumerate(items):
 for part2,name2,h2 in items[i+1:]:
  if part==part2:continue
  dist=(h^h2).bit_count()
  if dist<=args.max_distance:pairs.append({"a":{"split":part,"file":name},"b":{"split":part2,"file":name2},"phash_hamming":dist})
pairs.sort(key=lambda x:(x["phash_hamming"],x["a"]["file"],x["b"]["file"]))
result={"method":"63-bit DCT perceptual hash of 32x32 grayscale image; pairs across splits only", "max_distance":args.max_distance,"images":len(items),"candidates":pairs,"caution":"Candidates require visual verification; perceptual similarity alone does not establish duplication, shared capture, or leakage."}
args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(str(args.dataset),"images",len(items),"candidate pairs",len(pairs))
