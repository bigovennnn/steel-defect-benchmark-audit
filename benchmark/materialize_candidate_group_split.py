"""Materialize an exploratory split snapshot as a new copy; no train/test evaluation."""
from pathlib import Path
import argparse, hashlib, json, shutil

PARTS=('train','val','test')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--design',type=Path,required=True)
    ap.add_argument('--destination',type=Path,required=True)
    args=ap.parse_args()
    src=args.source.resolve();dst=args.destination.resolve();design=args.design.resolve()
    if dst.exists():raise SystemExit('Destination exists; refusing overwrite')
    if dst.drive.upper()!='D:':raise SystemExit('Destination must be on D:')
    manifest=src/'split.json';old=json.loads(manifest.read_text(encoding='utf-8'));new=json.loads(design.read_text(encoding='utf-8'))
    if new['dataset_split_sha256']!=sha(manifest):raise SystemExit('Source manifest hash mismatch')
    source_parts={s:p for p in PARTS for s in old[p]};dest_parts={s:p for p in PARTS for s in new[p]}
    if len(source_parts)!=sum(len(old[p]) for p in PARTS) or len(dest_parts)!=sum(len(new[p]) for p in PARTS):raise SystemExit('Duplicate image ID')
    if set(source_parts)!=set(dest_parts):raise SystemExit('Image identity mismatch')
    if any(Path(s).name!=s or '/' in s or '\\' in s for s in source_parts):raise SystemExit('Invalid image ID')
    for stem,part in source_parts.items():
        for kind,ext in (('images','.jpg'),('labels','.txt')):
            if not (src/kind/part/(stem+ext)).is_file():raise SystemExit('Missing source '+stem)
    for part in PARTS:
        for kind in ('images','labels'):(dst/kind/part).mkdir(parents=True,exist_ok=True)
        for stem in new[part]:
            for kind,ext in (('images','.jpg'),('labels','.txt')):
                shutil.copy2(src/kind/source_parts[stem]/(stem+ext),dst/kind/part/(stem+ext))
    info={part:new[part] for part in PARTS}
    info['provenance']={'status':'exploratory candidate-group snapshot; unverified residual leakage',
                        'source_split_sha256':sha(manifest),'design_sha256':sha(design),
                        'test_membership_changed':new['test_membership_changed']}
    (dst/'split.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    names=['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
    yaml='path: '+dst.as_posix()+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(names))
    (dst/'neu.yaml').write_text(yaml,encoding='utf-8')
    print(json.dumps({'destination':str(dst),'counts':{p:len(info[p]) for p in PARTS},'design_sha256':sha(design)},ensure_ascii=False))

if __name__=='__main__':main()
