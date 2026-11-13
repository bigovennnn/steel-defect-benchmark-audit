"""Check standard train-mode running-stat null with real source tensors and synthetic inputs."""
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
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',required=True,type=Path);ap.add_argument('--init-dir',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();out=a.output.resolve()
    if out.exists() or out.drive.upper()!='D:':raise SystemExit('New D: output required')
    if any(not os.environ.get(k,'').upper().startswith('D:') for k in ('TEMP','TMP','TORCH_HOME','YOLO_CONFIG_DIR')):raise SystemExit('Load project-env.ps1')
    if sha(a.weights)!=EXPECTED:raise RuntimeError('Unexpected weights')
    for name,digest in structure.PINNED.items():
        if sha(structure.REPO/name)!=digest:raise RuntimeError('Source changed: '+name)
    audit=json.loads((a.init_dir/'audit.json').read_text(encoding='utf-8'));initial_path=a.init_dir/'target_initial_state.pt'
    if sha(initial_path)!=audit['target_initial_file_sha256'] or audit['source_weight_sha256']!=EXPECTED:raise RuntimeError('Initialization changed')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    if policy.state_hash(initial)!=audit['target_initial_state_sha256']:raise RuntimeError('Initial tensor hash changed')
    torch.set_num_threads(2);torch.manual_seed(audit['seed'])
    source=YOLO(str(a.weights.resolve())).model.cpu().float();meta={};ns=structure.namespace(torch,nn,meta)
    cfg=yaml.safe_load((structure.REPO/'ultralytics/cfg/models/v8/MSRDet.yaml').read_text(encoding='utf-8'));cfg['scale']='n';cfg['nc']=6
    with torch.no_grad(),torch.device('cpu'):target=ns['DetectionModel'](deepcopy(cfg),verbose=False)
    states={}
    for label,(affine,stats) in policy.POLICIES.items():
        states[label],_=policy.policy_state(source.state_dict(),initial,audit['affected_bn_paths'],affine,stats)
        if policy.state_hash(states[label])!=audit['policies'][label]['state_sha256']:raise AssertionError('Policy hash mismatch')
    sample=torch.rand((2,3,64,64),generator=torch.Generator().manual_seed(20261102))
    checks={}
    for left,right in (('A0R0','A0R1'),('A1R0','A1R1')):
        models=[deepcopy(target),deepcopy(target)]
        for model,label in zip(models,(left,right)):model.load_state_dict(states[label],strict=True)
        values=[policy.forward_and_gradient(model,sample,torch) for model in models]
        out_a,grad_a=values[0];out_b,grad_b=values[1]
        if grad_a.keys()!=grad_b.keys():raise AssertionError('Different gradient keys')
        eq_out=all(torch.equal(x,y) for x,y in zip(out_a,out_b)) and len(out_a)==len(out_b)
        eq_grad=all(torch.equal(grad_a[k],grad_b[k]) for k in grad_a)
        if not eq_out or not eq_grad:raise AssertionError('Train-mode null failed for '+left)
        checks[left+'_vs_'+right]={'outputs_bitwise_equal':eq_out,'gradients_bitwise_equal':eq_grad,
                                    'outputs_checked':len(out_a),'gradient_tensors_checked':len(grad_a)}
    if meta['inactive_class_constructor_calls']:raise RuntimeError('Unexpected guard use')
    result={'status':'passed','scope':'one synthetic minibatch and diagnostic loss; no dataset, optimizer, model evaluation or training run',
            'source_weight_sha256':EXPECTED,'initial_state_sha256':audit['target_initial_state_sha256'],
            'input_seed':20261102,'input_shape':[2,3,64,64],'checks':checks,
            'limits':'Only this model, input and train-mode BN; running state may affect eval, EMA, early stopping, or other policies.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
