"""Test class-wise image/group-count matching after exact size matching failed.

Uses aggregate availability only; does not select images, split data, or evaluate models.
"""
import argparse,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feasibility',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    source=json.loads(a.feasibility.read_text(encoding='utf-8'));grouped=defaultdict(list)
    for row in source['signatures']:grouped[row['filename_class']].append(row)
    reports=[]
    for cls,rows in sorted(grouped.items()):
        rows.sort(key=lambda r:r['group_size']);n=len(rows);sizes=np.array([r['group_size'] for r in rows]);needed=np.array([r['required'] for r in rows]);available=np.array([r['available'] for r in rows])
        # x is the number of groups chosen at each observed size; signed slack captures size-count deviation.
        cost=np.r_[np.zeros(n),np.ones(2*n)];A=np.zeros((n+2,3*n))
        A[0,:n]=1;A[1,:n]=sizes
        for i in range(n):A[i+2,i]=1;A[i+2,n+i]=1;A[i+2,2*n+i]=-1
        targets=np.r_[needed.sum(),sizes@needed,needed]
        bounds=Bounds(np.zeros(3*n),np.r_[available,np.full(2*n,np.inf)])
        res=milp(cost,integrality=np.ones(3*n),bounds=bounds,constraints=LinearConstraint(A,targets,targets),options={'time_limit':10})
        if res.x is None:raise RuntimeError('No feasible aggregate solution for '+cls+': '+res.message)
        x=np.rint(res.x[:n]).astype(int)
        if x.sum()!=needed.sum() or sizes@x!=sizes@needed or np.any(x>available):raise AssertionError('Invalid solution')
        reports.append({'filename_class':cls,'required_groups':int(needed.sum()),'required_images':int(sizes@needed),
                        'selected_groups':int(x.sum()),'selected_images':int(sizes@x),
                        'group_size_composition':[{'size':int(s),'purge':int(nv),'control':int(xv),'available':int(av)}
                                                  for s,nv,xv,av in zip(sizes,needed,x,available)],
                        'status':int(res.status),'solver_message':str(res.message)})
    out={'status':'aggregate feasibility only; no actual groups selected',
         'source_sha256':hashlib.sha256(a.feasibility.read_bytes()).hexdigest(),
         'result':reports,'limitations':'Only filename class, image count, and group count matched; group sizes can differ. Does not account for validation allocation, multilabel boxes, or missing candidate edges.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
