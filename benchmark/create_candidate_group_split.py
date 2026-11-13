"""Create an exploratory candidate-group split, without copying images or evaluating models.

All candidate edges are conservatively treated as indivisible constraints. This is not a
confirmed leakage-free or confirmatory dataset. Test membership is allowed to change.
"""
from pathlib import Path
import argparse, csv, hashlib, json
from collections import Counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

PARTS = ('train', 'val', 'test')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,default=60)
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Refusing overwrite')
    manifest=args.dataset/'split.json'
    original=json.loads(manifest.read_text(encoding='utf-8'))
    old={s:p for p in PARTS for s in original[p]}
    if len(old)!=sum(len(original[p]) for p in PARTS):raise ValueError('duplicate manifest image')
    parent={s:s for s in old}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    with args.candidates.open(encoding='utf-8-sig',newline='') as f:edges=list(csv.DictReader(f))
    for row in edges:
        a,b=Path(row['a_file']).stem,Path(row['b_file']).stem
        if a not in old or b not in old or old[a]!=row['a_split'] or old[b]!=row['b_split']:raise ValueError('candidate/manifest mismatch')
        x,y=find(a),find(b)
        if x!=y:parent[y]=x
    groups={}
    for stem in sorted(old):groups.setdefault(find(stem),[]).append(stem)
    groups=sorted((sorted(v) for v in groups.values()),key=lambda v:v[0])
    labels={}
    for stem,part in old.items():
        classes=set()
        for row in (args.dataset/'labels'/part/(stem+'.txt')).read_text(encoding='utf-8').splitlines():
            if row.strip():
                c=int(row.split()[0])
                if not 0<=c<6:raise ValueError('unexpected class')
                classes.add(c)
        labels[stem]=classes
    size_targets=[len(original[p]) for p in PARTS]
    positive_targets=np.array([[sum(c in labels[s] for s in original[p]) for c in range(6)] for p in PARTS])
    n=len(groups);xvars=n*3;slack=36;columns=xvars+slack
    costs=np.zeros(columns);integrality=np.zeros(columns,dtype=int);integrality[:xvars]=1
    for i,g in enumerate(groups):
        for p,part in enumerate(PARTS):costs[3*i+p]=sum(old[s]!=part for s in g)
    costs[xvars:]=10.0
    # Each component has exactly one part; part sizes match original exactly; class deviations have L1 slack.
    A=lil_matrix((n+3+18,columns),dtype=float);lb=np.zeros(n+21);ub=np.zeros(n+21)
    for i,g in enumerate(groups):
        for p in range(3):
            col=3*i+p;A[i,col]=1;A[n+p,col]=len(g)
            for c in range(6):A[n+3+6*p+c,col]=sum(c in labels[s] for s in g)
        lb[i]=ub[i]=1
    for p in range(3):lb[n+p]=ub[n+p]=size_targets[p]
    for p in range(3):
        for c in range(6):
            idx=6*p+c;row=n+3+idx;A[row,xvars+2*idx]=1;A[row,xvars+2*idx+1]=-1
            lb[row]=ub[row]=positive_targets[p,c]
    upper=np.full(columns,np.inf);upper[:xvars]=1
    result=milp(costs,integrality=integrality,bounds=Bounds(np.zeros(columns),upper),
                constraints=LinearConstraint(A.tocsr(),lb,ub),
                options={'time_limit':args.seconds,'mip_rel_gap':0.01})
    if result.x is None:raise RuntimeError(f'no feasible solution: {result.message}')
    assignment=result.x[:xvars].reshape(n,3)
    if np.max(np.abs(assignment-np.rint(assignment)))>1e-5:raise RuntimeError('non-integer solution')
    new={s:PARTS[int(np.argmax(assignment[i]))] for i,g in enumerate(groups) for s in g}
    if any(sum(new[s]==p for s in new)!=size_targets[i] for i,p in enumerate(PARTS)):raise RuntimeError('split sizes changed')
    if any(new[Path(e['a_file']).stem]!=new[Path(e['b_file']).stem] for e in edges):raise RuntimeError('candidate edge split')
    counts={p:[sum(c in labels[s] for s in new if new[s]==p) for c in range(6)] for p in PARTS}
    out={'status':'exploratory only; candidate edges may include false positives and miss other associations',
         'dataset_split_sha256':sha(manifest),'candidate_csv_sha256':sha(args.candidates),'script_sha256':sha(Path(__file__)),
         'design':'all candidate-edge components grouped; exact previous split sizes; minimize changed memberships plus 10 times class-positive count absolute deviations',
         'test_membership_changed':set(original['test'])!={s for s in new if new[s]=='test'},'model_metrics_used':False,'candidate_count':len(edges),'component_count':len(groups),
         'optimizer':{'status':int(result.status),'message':str(result.message),'objective':float(result.fun),
                      'mip_gap':float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None},
         'sizes':dict(zip(PARTS,size_targets)),'original_class_positive_images':{p:positive_targets[i].tolist() for i,p in enumerate(PARTS)},
         'new_class_positive_images':counts,'changed_image_count':sum(old[s]!=new[s] for s in old),
         'moves':dict(sorted(Counter(f'{old[s]}->{new[s]}' for s in old if old[s]!=new[s]).items())),
         'train':sorted(s for s in new if new[s]=='train'),'val':sorted(s for s in new if new[s]=='val'),
         'test':sorted(s for s in new if new[s]=='test')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k not in PARTS},ensure_ascii=False))

if __name__=='__main__':main()
