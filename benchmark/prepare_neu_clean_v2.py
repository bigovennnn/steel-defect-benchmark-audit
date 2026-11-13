"""Create a separate exploratory NEU copy excluding two known transformed duplicates.

Source provenance: KeenForgeAI NEU-DET-corrected dataset card. No original file is changed.
"""
from pathlib import Path
import json,hashlib,shutil
base=Path(__file__).resolve().parent;src=base/'datasets/neu_dedup_v1';dst=base/'datasets/neu_clean_v2'
if dst.exists():raise SystemExit('Target exists; refusing to overwrite')
m=json.loads((src/'split.json').read_text(encoding='utf-8'));drop={'scratches_247':'rotated near duplicate of scratches_232','pitted_surface_292':'shifted near duplicate of pitted_surface_171'}
if not set(drop)<=set(m['train']):raise SystemExit('Source split differs from verified split')
for part in ('train','val','test'):
 for kind in ('images','labels'):(dst/kind/part).mkdir(parents=True,exist_ok=True)
 for stem in m[part]:
  if stem in drop:continue
  image=src/'images'/part/(stem+'.jpg');label=src/'labels'/part/(stem+'.txt')
  shutil.copy2(image,dst/'images'/part/image.name)
  original=label.read_text(encoding='utf-8').splitlines();seen=set();clean=[]
  for line in original:
   key=tuple(line.split())
   if key and key not in seen:clean.append(line);seen.add(key)
  (dst/'labels'/part/label.name).write_text('\n'.join(clean)+'\n',encoding='utf-8')
 m[part]=[stem for stem in m[part] if stem not in drop]
m['provenance']={'source':'neu_dedup_v1/split.json','source_split_sha256':hashlib.sha256((src/'split.json').read_bytes()).hexdigest(),'excluded':drop,'duplicate_annotation_rows':'remove exact duplicate rows per image','external_report':'https://huggingface.co/datasets/KeenForgeAI/NEU-DET-corrected','status':'exploratory; not adjudicated for all annotation quality'}
(dst/'split.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
names=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
yaml='path: '+dst.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(names))
(dst/'neu.yaml').write_text(yaml,encoding='utf-8')
print('created',dst,{part:len(m[part]) for part in ('train','val','test')})
