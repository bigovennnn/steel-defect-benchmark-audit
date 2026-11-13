"""Build conservative exploratory GC10 split excluding ambiguous annotations."""
from pathlib import Path
import json, shutil
base = Path(__file__).resolve().parent
src = base / "datasets/gc10_exploratory"
dst = base / "datasets/gc10_review_v1"
if dst.exists(): raise SystemExit("Target exists; refusing to overwrite")
m = json.loads((src / "split.json").read_text(encoding="utf-8"))
excluded = {"img_02_4406772100_00175":"no object in XML", "img_06_3402617700_00988":"no object in XML", "img_02_425616500_00770":"unknown class d in XML"}
all_names = {name for part in ("train","val","test") for name in m[part]}
if not set(excluded) <= all_names: raise SystemExit("Unexpected source split, stopping")
for part in ("train", "val", "test"):
    for kind in ("images","labels"): (dst/kind/part).mkdir(parents=True, exist_ok=True)
    for stem in m[part]:
        if stem in excluded: continue
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            f = src/kind/part/(stem+suffix)
            if not f.exists(): raise SystemExit("Missing source: "+str(f))
            shutil.copy2(f, dst/kind/part/f.name)
    m[part] = [name for name in m[part] if name not in excluded]
m["provenance"]={"source":"gc10_exploratory", "excluded":excluded, "scope":"exploratory; exclusions require annotation review; no test set claims"}
(dst/"split.json").write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")
names=["punching_hole","welding_line","crescent_gap","water_spot","oil_spot","silk_spot","inclusion","rolled_pit","crease","waist_folding"]
yaml="path: "+dst.as_posix()+"\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"+"".join(f"  {i}: {n}\n" for i,n in enumerate(names))
(dst/"gc10.yaml").write_text(yaml,encoding="utf-8")
print("created",dst,{p:len(m[p]) for p in ("train","val","test")})
