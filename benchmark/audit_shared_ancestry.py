"""Identify matching-key parameter paths whose module ancestry differs between local variants."""
from pathlib import Path
import argparse, hashlib, json, sys
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'paper-try'))
from yolo_paper import register_modules
from ultralytics.nn.tasks import DetectionModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Refusing overwrite: '+str(out))
    audit_path = BASE/'transfer_coverage_local_v1.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    register_modules()
    models = {name: DetectionModel(str(BASE/'paper-try/configs/models'/f'yolo11n-{name}.yaml'), nc=6, verbose=False)
              for name in ('base','hwd-strip')}
    modules = {name: dict(model.named_modules()) for name,model in models.items()}
    rows = {name:{r['key']:r for r in audit['variants'][name]['tensors']
                  if r['kind']=='parameter' and r['partial_transfer_eligible']} for name in models}
    shared=[]
    for key in sorted(rows['base'].keys() & rows['hwd-strip'].keys()):
        a,b=rows['base'][key],rows['hwd-strip'][key]
        assert a['target_shape']==b['target_shape'] and a['numel']==b['numel']
        parts=key.split('.')[:-1]
        differences=[]
        for i in range(1,len(parts)+1):
            prefix='.'.join(parts[:i])
            ma,mb=modules['base'].get(prefix),modules['hwd-strip'].get(prefix)
            if ma is None or mb is None:
                differences.append({'path':prefix,'base':str(type(ma)),'hwd_strip':str(type(mb))})
            elif type(ma) is not type(mb):
                differences.append({'path':prefix,'base':type(ma).__module__+'.'+type(ma).__qualname__,
                                    'hwd_strip':type(mb).__module__+'.'+type(mb).__qualname__})
        shared.append({'key':key,'numel':a['numel'],'differing_ancestors':differences})
    different=[r for r in shared if r['differing_ancestors']]
    result={'scope':'same-key/same-shape parameters eligible under local partial transfer policy',
            'caveat':'Matching ancestry types do not prove identical computation or semantic roles.',
            'coverage_input_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'shared_tensors':len(shared),'shared_elements':sum(r['numel'] for r in shared),
            'different_ancestor_tensors':len(different),'different_ancestor_elements':sum(r['numel'] for r in different),
            'rows':shared}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    main()
