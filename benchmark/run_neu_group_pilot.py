"""Explore matched initialization on candidate-group data; validation only, no confirmatory results."""
from pathlib import Path
import argparse,csv,hashlib,json,time,torch,random,numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
base=Path(__file__).resolve().parent
dataset_split=base/'datasets/neu_candidate_groups_v3/split.json'
dataset_manifest=base/'manifest_neu_candidate_groups_v3.json'
expected_design='13f45ae5fe66cf7872cfc79f1e5a753aad13459aa266bf386054a28657b2fed7'
if json.loads(dataset_split.read_text(encoding='utf-8'))['provenance']['design_sha256']!=expected_design:raise SystemExit('Unexpected split design')
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--init',choices=['scratch','pretrained'],required=True)
ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--fraction',type=float,default=1.0);ap.add_argument('--imgsz',type=int,default=640);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--project',default='runs/grouped_pilot_v3');args=ap.parse_args()
name=f'neu_{args.epochs}_{args.init}_seed{args.seed}';project=base/args.project;run=project/name
if run.exists():raise SystemExit('Run directory exists; refusing overwrite: '+str(run))
weight=base/'weights/yolo11n_coco_official.pt';expected='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
if args.init=='pretrained' and hashlib.sha256(weight.read_bytes()).hexdigest()!=expected:raise SystemExit('Pretrained weight hash mismatch')
class FixedArchitectureTrainer(DetectionTrainer):
 def get_model(self,cfg=None,weights=None,verbose=True):
  # Never call BaseModel.load: its class-branch widening changes the architecture.
  model=super().get_model(cfg=cfg,weights=None,verbose=verbose)
  target=model.state_dict();head=f'model.{len(model.model)-1}.cv3.'
  before={k:v.detach().clone() for k,v in target.items() if k.startswith(head)}
  copied=[]
  if args.init=='pretrained':
   py_rng=random.getstate();np_rng=np.random.get_state();torch_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
   try:src=YOLO(str(weight)).model.float().state_dict()
   finally:
    random.setstate(py_rng);np.random.set_state(np_rng);torch.set_rng_state(torch_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
   selected={k:v for k,v in src.items() if k in target and v.shape==target[k].shape and not k.startswith(head)}
   if not selected:raise RuntimeError('No pretrained tensors matched')
   model.load_state_dict(selected,strict=False);copied=sorted(selected)
  after=model.state_dict()
  if not all(torch.equal(v,after[k]) for k,v in before.items()):raise RuntimeError('Classification branch unexpectedly changed')
  if copied and not all(torch.equal(after[k].cpu(),selected[k].cpu()) for k in copied):raise RuntimeError('Pretrained tensor verification failed')
  digest=hashlib.sha256()
  for k in sorted(before):digest.update(k.encode());digest.update(before[k].cpu().contiguous().numpy().tobytes())
  shape_digest=hashlib.sha256(json.dumps([(k,list(v.shape)) for k,v in target.items()],separators=(',',':')).encode()).hexdigest()
  audit={'init':args.init,'seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),'tensor_shapes_sha256':shape_digest,'classification_initial_sha256':digest.hexdigest(),'torch_rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'transferred_tensors':len(copied),'excluded':'entire classification branch cv3; nonmatching shapes','weight_sha256':expected if copied else None,'split_sha256':hashlib.sha256(dataset_split.read_bytes()).hexdigest(),'manifest_file_sha256':hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),'design_sha256':expected_design}
  run.mkdir(parents=True,exist_ok=True);(run/'initialization_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('INITIALIZATION_AUDIT',json.dumps(audit),flush=True)
  return model
model=YOLO('yolo11n.yaml');start=time.time()
# pretrained=False prevents framework's automatic class-head rebuilding; custom trainer transfers selected tensors.
model.train(trainer=FixedArchitectureTrainer,data=str(base/'datasets/neu_candidate_groups_v3/neu.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device=0,workers=0,pretrained=False,seed=args.seed,deterministic=True,fraction=args.fraction,project=str(project),name=name,exist_ok=False,plots=False,verbose=False,amp=False,save=True,optimizer='SGD',lr0=0.01,momentum=0.937,weight_decay=0.0005,patience=0)
save=Path(model.trainer.save_dir)
if save.resolve()!=run.resolve():raise SystemExit('Unexpected run directory')
with (save/'results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
if [int(x['epoch']) for x in rows]!=list(range(1,args.epochs+1)):raise SystemExit('Unexpected epoch history')
summary={'purpose':'exploratory matched initialization; no test evaluation','seed':args.seed,'init':args.init,'epochs_completed':args.epochs,'seconds':round(time.time()-start,2),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'run_dir':str(save)}
(save/'case_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
