"""Assess conditional candidate-component purging under an index-based test rule.

No split is finalized, no image is changed, and no model is run or evaluated.
"""
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def is_test(stem):
    m=re.fullmatch(r'(.+)_(\d+)',stem)
    if not m or not 1<=int(m.group(2))<=300:raise ValueError('bad stem '+stem)
    return int(m.group(2))>=241

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',required=True,type=Path)
    ap.add_argument('--candidates',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    m=json.loads(a.split.read_text(encoding='utf-8'));names=[s for p in ('train','val','test') for s in m[p]]
    if len(names)!=len(set(names)):raise ValueError('duplicate image ID')
    parent={s:s for s in names}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    with a.candidates.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        x,y=find(Path(row['a_file']).stem),find(Path(row['b_file']).stem)
        if x!=y:parent[y]=x
    groups={}
    for name in names:groups.setdefault(find(name),[]).append(name)
    cross=[g for g in groups.values() if any(is_test(s) for s in g) and any(not is_test(s) for s in g)]
    dev_removed=sorted(s for g in cross for s in g if not is_test(s))
    affected_test=sorted(s for g in cross for s in g if is_test(s))
    out={'status':'exploratory feasibility only; all unadjudicated candidates treated as links',
         'rule_source':'https://doi.org/10.1038/s41598-026-64588-4',
         'scope':'local dataset and index mapping only; authors actual files and splits not verified',
         'dataset_manifest_sha256':sha(a.split),'candidate_csv_sha256':sha(a.candidates),
         'candidate_pairs':len(rows),'source_images':len(names),
         'development_images':sum(not is_test(s) for s in names),'test_images':sum(is_test(s) for s in names),
         'cross_index_candidate_components':len(cross),'purged_development_images':dev_removed,
         'affected_test_images':affected_test,
         'purged_by_filename_class':dict(sorted(Counter(s.rsplit('_',1)[0] for s in dev_removed).items())),
         'limitations':'candidate links can be false positives or miss related images; this is not causal performance evidence; no evaluation was run'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:(len(v) if k in ('purged_development_images','affected_test_images') else v) for k,v in out.items()},ensure_ascii=False))

if __name__=='__main__':main()
