"""Audit same-name and decoded-pixel duplicates in the GC10 Kaggle mirror."""
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib, json
base=Path(__file__).resolve().parent
root=base/"datasets/raw_gc10_kaggle"
paths=sorted(root.rglob("*.jpg")); stems=defaultdict(list); pixel=defaultdict(list)
for i,p in enumerate(paths,1):
 with Image.open(p) as im:
  im=im.convert("L"); h=hashlib.sha256((str(im.size)+":").encode()+im.tobytes()).hexdigest()
 stems[p.stem].append(str(p.relative_to(root)));pixel[h].append(str(p.relative_to(root)))
 if i%500==0: print(i,"/",len(paths),flush=True)
xml={p.stem for p in root.rglob("*.xml")}; img=set(stems)
result={"n_image_files":len(paths),"n_unique_stems":len(stems),"n_xml_stems":len(xml),"image_stems_without_xml":sorted(img-xml),"xml_stems_without_image":sorted(xml-img),"repeated_stems":{k:v for k,v in stems.items() if len(v)>1},"pixel_duplicate_groups":{k:v for k,v in pixel.items() if len(v)>1}}
(base/"gc10_pixel_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print("done: repeated stems",len(result["repeated_stems"]),"pixel groups",len(result["pixel_duplicate_groups"]),"without xml",len(img-xml),flush=True)
