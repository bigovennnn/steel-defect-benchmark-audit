"""Summarize candidate-connected components; no duplicate adjudication or new split."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite: '+str(args.output))
    data=json.loads(args.split.read_text(encoding='utf-8'))
    parts={name:part for part in ('train','val','test') for name in data[part]}
    if len(parts)!=sum(len(data[p]) for p in ('train','val','test')):
        raise ValueError('Image occurs in more than one split')
    parent={name:name for name in parts}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    with args.candidates.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        a,b=Path(row['a_file']).stem,Path(row['b_file']).stem
        if a not in parts or b not in parts or parts[a]!=row['a_split'] or parts[b]!=row['b_split']:
            raise ValueError('Candidate and split manifest disagree')
        ra,rb=find(a),find(b)
        if ra!=rb:parent[rb]=ra
    groups={}
    for name in parts:groups.setdefault(find(name),[]).append(name)
    multi=[sorted(g) for g in groups.values() if len(g)>1]
    cross=[g for g in multi if len({parts[n] for n in g})>1]
    report={'status':'exploratory; all algorithmic candidates treated as edges without adjudication',
            'caveat':'Components may join unrelated images through false-positive chains and omit undetected links; do not use as a final split without review.',
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.candidates,args.split)},
            'images':len(parts),'candidate_edges':len(rows),'components':len(groups),
            'multi_image_components':len(multi),'images_in_multi_image_components':sum(map(len,multi)),
            'cross_split_components':len(cross),'images_in_cross_split_components':sum(map(len,cross)),
            'min_images_reassigned_if_all_candidates_enforced_no_size_constraints':sum(
                len(g)-max(Counter(parts[n] for n in g).values()) for g in cross),
            'multi_component_size_distribution':dict(sorted(Counter(map(len,multi)).items())),
            'cross_split_groups':[{'members':[{'image':n,'split':parts[n]} for n in g]} for g in sorted(cross,key=lambda x:(-len(x),x))]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cross_split_groups','input_sha256')},ensure_ascii=False))

if __name__=='__main__':main()
