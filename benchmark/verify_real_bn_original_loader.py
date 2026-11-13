"""Cross-check saved 6-class BN policy hashes against the pinned original loader."""
from pathlib import Path
import argparse,hashlib,json,os
from copy import deepcopy
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
import audit_msrdet_structure as structure
import audit_msrdet_bn_policy as policy

EXPECTED='31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--init-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve()
    if out.exists() or out.drive.upper()!='D:':raise SystemExit('New D: output required')
    if any(not os.environ.get(k,'').upper().startswith('D:') for k in ('TEMP','TMP','TORCH_HOME','YOLO_CONFIG_DIR')):raise SystemExit('Load project-env.ps1')
    if sha(a.weights)!=EXPECTED:raise RuntimeError('Source checkpoint changed')
    for k,v in structure.PINNED.items():
        if sha(structure.REPO/k)!=v:raise RuntimeError('Source code changed: '+k)
    audit=json.loads((a.init_dir/'audit.json').read_text(encoding='utf-8'));p=a.init_dir/'target_initial_state.pt'
    if sha(p)!=audit['target_initial_file_sha256'] or audit['source_weight_sha256']!=EXPECTED:raise RuntimeError('Audit inputs changed')
    saved=torch.load(p,map_location='cpu',weights_only=True)
    if policy.state_hash(saved)!=audit['target_initial_state_sha256']:raise RuntimeError('Saved tensor state changed')
    torch.set_num_threads(2);torch.manual_seed(audit['seed'])
    source=YOLO(str(a.weights.resolve())).model.cpu().float();meta={};ns=structure.namespace(torch,nn,meta)
    cfg=yaml.safe_load((structure.REPO/'ultralytics/cfg/models/v8/MSRDet.yaml').read_text(encoding='utf-8'));cfg['scale']='n';cfg['nc']=6
    with torch.no_grad(),torch.device('cpu'):target=ns['DetectionModel'](deepcopy(cfg),verbose=False)
    target.load_state_dict(saved,strict=True)
    source_state=source.state_dict();states={}
    for label,(affine,stats) in policy.POLICIES.items():
        states[label],_=policy.policy_state(source_state,saved,audit['affected_bn_paths'],affine,stats)
        if policy.state_hash(states[label])!=audit['policies'][label]['state_sha256']:raise AssertionError('Policy hash mismatch: '+label)
    target.load(source,verbose=False)
    loaded=target.state_dict()
    if any(not torch.equal(v,states['A1R1'][k]) for k,v in loaded.items()):raise AssertionError('Original loader differs')
    if meta['inactive_class_constructor_calls']:raise RuntimeError('Unexpected guard use')
    result={'status':'passed','scope':'CPU real official checkpoint and fixed source loader; no dataset or optimization',
            'target_nc':6,'source_weight_sha256':EXPECTED,'initial_file_sha256':sha(p),
            'state_tensors_compared':len(loaded),'all_four_policy_hashes_match':True,
            'original_loader_equals_A1R1':True,'policy_A1R1_sha256':policy.state_hash(loaded),
            'limitations':'Does not show authors used this checkpoint or loader; no training or performance effect.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
