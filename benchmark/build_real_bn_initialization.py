"""Build and audit real-weight BN policy states on the fixed 6-class MSRDet.

Only saves a CPU tensor-only target initialization and an audit; no training or data access.
"""
from pathlib import Path
import argparse,hashlib,json,os
from copy import deepcopy
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
import audit_msrdet_structure as structure
import audit_msrdet_bn_policy as policy

CHECKPOINT_SHA='31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261101);a=ap.parse_args()
    dest=a.output_dir.resolve()
    if dest.exists() or dest.drive.upper()!='D:':raise SystemExit('Output must be a new D: directory')
    if any(not os.environ.get(k,'').upper().startswith('D:') for k in ('TEMP','TMP','TORCH_HOME','YOLO_CONFIG_DIR')):raise SystemExit('Load project-env.ps1 first')
    if sha(a.weights)!=CHECKPOINT_SHA:raise SystemExit('Unexpected source weight')
    for p,d in structure.PINNED.items():
        if sha(structure.REPO/p)!=d:raise RuntimeError('Pinned source changed: '+p)
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    source=YOLO(str(a.weights.resolve())).model.cpu().float()
    if source.nc!=80:raise RuntimeError('Unexpected source class count')
    record={};ns=structure.namespace(torch,nn,record)
    cfg=yaml.safe_load((structure.REPO/'ultralytics/cfg/models/v8/MSRDet.yaml').read_text(encoding='utf-8'))
    cfg['scale']='n';cfg['nc']=6
    with torch.no_grad(),torch.device('cpu'):target=ns['DetectionModel'](deepcopy(cfg),verbose=False)
    if record['inactive_class_constructor_calls']:raise RuntimeError('Unexpected guard constructor')
    affected=[r for r in policy.inspect_pairs(source,target,nn) if not r['conv_key_shape_matched']]
    if len(affected)!=12:raise RuntimeError('Unexpected treatment site count')
    paths=[r['bn_path'] for r in affected];keys={k for r in affected for k in r['bn_keys']}
    initial={k:v.detach().cpu().clone() for k,v in target.state_dict().items()};src=source.state_dict()
    states={};summary={}
    for label,(affine,stats) in policy.POLICIES.items():
        state,copied=policy.policy_state(src,initial,paths,affine,stats)
        for path in paths:
            for suffix in policy.BN_AFFINE+policy.BN_STATS:
                key=path+'.'+suffix;use_source=affine if suffix in policy.BN_AFFINE else stats
                if not torch.equal(state[key],src[key] if use_source else initial[key]):raise AssertionError('Treatment mismatch '+key)
        states[label]=state
        summary[label]={'state_sha256':policy.state_hash(state),'affine_from_source':affine,'running_stats_from_source':stats,
                        'treatment_tensors_changed_from_initial':sum(not torch.equal(state[k],initial[k]) for k in keys)}
    for k in initial:
        if k not in keys and any(not torch.equal(states['A0R0'][k],s[k]) for s in states.values()):raise AssertionError('Nontreatment changed '+k)
    for k,v in initial.items():
        expected=src[k] if k in src and src[k].shape==v.shape else v
        if not torch.equal(states['A1R1'][k],expected):raise AssertionError('Full policy differs from intersection '+k)
    dest.mkdir(parents=True,exist_ok=False);state_path=dest/'target_initial_state.pt'
    torch.save(initial,state_path)
    loaded=torch.load(state_path,map_location='cpu',weights_only=True)
    if policy.state_hash(loaded)!=policy.state_hash(initial):raise AssertionError('Saved initial state changed')
    result={'status':'exploratory tensor initialization audit; no training, data access, or evaluation',
            'seed':a.seed,'source_weight_sha256':CHECKPOINT_SHA,'target_nc':6,
            'target_initial_file_sha256':sha(state_path),'target_initial_state_sha256':policy.state_hash(initial),
            'source_state_sha256':policy.state_hash(src),'source_matching_state_tensors':len(copied),
            'affected_bn_paths':paths,'affected_bn_state_keys':sorted(keys),'affected_bn_sites':len(affected),
            'policies':summary,'nontreatment_equal':True,'A1R1_matches_key_shape_intersection':True,
            'source_code_pins':{p:sha(structure.REPO/p) for p in structure.PINNED},
            'script_sha256':sha(Path(__file__)),'torch':torch.__version__,
            'limitations':'Source and target software differ; matching does not establish semantic compatibility, performance, or published author practice.'}
    (dest/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('affected_bn_state_keys','source_code_pins','affected_bn_paths')},ensure_ascii=False))

if __name__=='__main__':main()
