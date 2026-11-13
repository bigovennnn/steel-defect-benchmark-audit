"""Audit official YOLOv8 state compatibility with a 6-class MSRDet reconstruction.

This is a CPU state inspection, not training, benchmark inference, or a claim about
the published authors' weights. The official checkpoint is loaded by local Ultralytics.
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

BASE=Path(__file__).resolve().parent
EXPECTED='31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() or out.drive.upper()!='D:':raise SystemExit('Output must be a new file on D:')
    if any(not os.environ.get(k,'').upper().startswith('D:') for k in ('TEMP','TMP','TORCH_HOME','YOLO_CONFIG_DIR')):raise SystemExit('Load project-env.ps1 first')
    if sha(args.weights)!=EXPECTED:raise SystemExit('Unexpected checkpoint hash')
    for name,digest in structure.PINNED.items():
        if structure.digest(structure.REPO/name)!=digest:raise SystemExit('Source file hash changed: '+name)
    torch.set_num_threads(2);torch.manual_seed(20261101)
    source=YOLO(str(args.weights.resolve())).model.cpu().float()
    if source.nc!=80:raise RuntimeError('Expected 80 source classes')
    record={};ns=structure.namespace(torch,nn,record)
    cfg=yaml.safe_load((structure.REPO/'ultralytics/cfg/models/v8/MSRDet.yaml').read_text(encoding='utf-8'))
    cfg['scale']='n';cfg['nc']=6
    with torch.no_grad(),torch.device('cpu'):target=ns['DetectionModel'](deepcopy(cfg),verbose=False)
    pairs=policy.inspect_pairs(source,target,nn)
    affected=[r for r in pairs if not r['conv_key_shape_matched']]
    src,dst=source.state_dict(),target.state_dict()
    matching=[k for k,v in src.items() if k in dst and v.shape==dst[k].shape]
    items=[]
    for r in affected:
        path=r['bn_path'];stats={}
        for suffix in policy.BN_AFFINE+policy.BN_STATS:
            key=path+'.'+suffix;v=src[key].detach().float()
            stats[suffix]={'shape':list(v.shape),'min':float(v.min()),'max':float(v.max()),
                           'mean':float(v.mean()),'tensor_sha256':policy.tensor_hash(src[key])}
        items.append({'bn_path':path,'category':r['category'],'channels':r['channels'],
                      'source_conv_shape':list(src[r['conv_key']].shape),'target_conv_shape':list(dst[r['conv_key']].shape),
                      'bn_statistics':stats})
    if record['inactive_class_constructor_calls']:raise RuntimeError('Unexpected guard constructor called')
    result={'status':'exploratory state compatibility; no training or detector evaluation',
            'source_checkpoint_sha256':EXPECTED,'source_repo':'https://huggingface.co/Ultralytics/YOLOv8',
            'source_revision':'27f858f2dc238503a66ce10248cf7ec71a3ded5d','source_classes':source.nc,
            'target_classes':6,'source_tensors':len(src),'target_tensors':len(dst),'matching_tensors':len(matching),
            'target_parameters':sum(p.numel() for p in target.parameters()),'bn_pairs':len(pairs),
            'affected_bn_sites':len(affected),'affected_bn_channels':sum(r['channels'] for r in affected),
            'affected_sites':items,'source_provenance':{k:structure.digest(structure.REPO/k) for k in structure.PINNED},
            'limits':'Different source and repository software versions; no assertion that paper authors used these weights or loader. Matched shapes do not prove semantic equivalence.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('affected_sites','source_provenance')},ensure_ascii=False))

if __name__=='__main__':main()
