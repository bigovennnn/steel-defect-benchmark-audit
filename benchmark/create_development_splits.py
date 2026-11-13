"""Generate exploratory development splits with a fixed, untouched test partition.

No files are copied and no model is evaluated. Seed changes training/validation membership;
the test list stays exactly as in the source manifest. This is not a preregistration.
"""
from pathlib import Path
import argparse,collections,hashlib,json,random
ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
root=a.dataset.resolve();source=json.loads((root/'split.json').read_text(encoding='utf-8'));parts=('train','val','test')
sets={part:set(source[part]) for part in parts}
if any(len(sets[p])!=len(source[p]) for p in parts) or any(sets[x]&sets[y] for x,y in [('train','val'),('train','test'),('val','test')]):raise SystemExit('Duplicate names or overlapping partitions')
records={};frequency=collections.Counter()
for part in ('train','val'):
 for stem in sorted(sets[part]):
  label=root/'labels'/part/(stem+'.txt');image=root/'images'/part/(stem+'.jpg')
  if not image.is_file() or not label.is_file():raise SystemExit('Missing file for '+stem)
  labels={int(line.split()[0]) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()}
  if not labels:raise SystemExit('Empty labels not supported: '+str(label))
  frequency.update(labels);records[stem]={'source_split':part,'classes':sorted(labels)}
groups=collections.defaultdict(list)
for stem,rec in records.items():groups[min(rec['classes'],key=lambda c:(frequency[c],c))].append(stem)
# Exactly preserve source validation size; allocate by largest remainder per class group.
total=len(records);nval=len(sets['val']);allocation={c:len(g)*nval//total for c,g in groups.items()}
remainder=nval-sum(allocation.values())
for c in sorted(groups,key=lambda c:(-(len(groups[c])*nval%total),c))[:remainder]:allocation[c]+=1
if a.outdir.exists():raise SystemExit('Output exists; refusing overwrite')
a.outdir.mkdir(parents=True)
sourcehash=hashlib.sha256((root/'split.json').read_bytes()).hexdigest()
for seed in a.seeds:
 rng=random.Random(seed);val=[];train=[]
 for cls in sorted(groups):
  names=sorted(groups[cls]);rng.shuffle(names);n=allocation[cls];val.extend(names[:n]);train.extend(names[n:])
 assert len(val)==nval and len(train)+len(val)==total and not (set(train)&set(val))
 result={'purpose':'exploratory development-resampling only; test fixed','source_dataset':str(root),'source_split_sha256':sourcehash,'seed':seed,'method':'group each image by rarest image-level class; largest-remainder validation allocation; Python random.Random','train':sorted(train),'val':sorted(val),'test':sorted(sets['test']),'source_parts':{k:v['source_split'] for k,v in sorted(records.items())},'validation_group_counts':dict(sorted(allocation.items()))}
 out=a.outdir/('split_seed'+str(seed)+'.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(out, 'counts',len(train),len(val),len(sets['test']))
