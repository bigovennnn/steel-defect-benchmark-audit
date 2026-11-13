"""Explore matching candidate purge groups to non-test groups, without choosing controls."""
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parts(stem):
    m=re.fullmatch(r'(.+)_(\d+)',stem)
    if not m or not 1<=int(m.group(2))<=300:raise ValueError('invalid ID')
    return m.group(1),int(m.group(2))>=241
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',required=True,type=Path);ap.add_argument('--candidates',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    split=json.loads(a.split.read_text(encoding='utf-8'));names=[s for p in ('train','val','test') for s in split[p]]
    if len(names)!=len(set(names)):raise ValueError('duplicate ID')
    par={s:s for s in names}
    def find(x):
        while par[x]!=x:par[x]=par[par[x]];x=par[x]
        return x
    with a.candidates.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        x,y=find(Path(r['a_file']).stem),find(Path(r['b_file']).stem)
        if x!=y:par[y]=x
    groups={}
    for s in names:groups.setdefault(find(s),[]).append(s)
    purge=[];available=[]
    for group in groups.values():
        classes={parts(s)[0] for s in group}
        if len(classes)!=1:raise ValueError('mixed filename classes; requires multivariate handling')
        dev=[s for s in group if not parts(s)[1]];test=[s for s in group if parts(s)[1]]
        if dev and test:purge.append((next(iter(classes)),len(dev)))
        elif dev:available.append((next(iter(classes)),len(dev)))
    a_count=Counter(purge);b_count=Counter(available)
    signatures=[{'filename_class':c,'group_size':size,'required':v,'available':b_count[(c,size)],
                 'shortfall':max(0,v-b_count[(c,size)])} for (c,size),v in sorted(a_count.items())]
    out={'status':'exploratory; filename class is not multi-label annotation matching',
         'scope':'availability before allocating a validation subset; all candidate edges unadjudicated',
         'input_hashes':{'split':sha(a.split),'candidates':sha(a.candidates)},'purge_group_count':len(purge),
         'purge_image_count':sum(size for _,size in purge),'exact_group_size_and_filename_class_possible':all(r['shortfall']==0 for r in signatures),
         'signatures':signatures,'limitations':'does not match image labels, boxes, defect size, validation assignment, or unknown acquisition groups'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
