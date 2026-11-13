"""Audit complete matched runs and summarize best-checkpoint validation metrics only."""
from pathlib import Path
import argparse,csv,json,math,statistics
import yaml
from ultralytics import YOLO
base=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--project',default='runs/paired_exploratory_v2');ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
project=base/a.project;rows=[];pairs=[];ignored={'name','save_dir'}
if a.output.exists():raise SystemExit('Refusing to overwrite '+str(a.output))
for seed in a.seeds:
 cases={}
 for init in ('scratch','pretrained'):
  run=project/f'neu_{a.epochs}_{init}_seed{seed}'
  summary=json.loads((run/'case_summary.json').read_text(encoding='utf-8'));audit=json.loads((run/'initialization_audit.json').read_text(encoding='utf-8'));args=yaml.safe_load((run/'args.yaml').read_text(encoding='utf-8'))
  if summary['seed']!=seed or summary['init']!=init or summary['epochs_completed']!=a.epochs or audit['seed']!=seed or audit['init']!=init:raise SystemExit('Summary mismatch: '+str(run))
  with (run/'results.csv').open(encoding='utf-8-sig',newline='') as f:epochs=list(csv.DictReader(f))
  if [int(x['epoch']) for x in epochs]!=list(range(1,a.epochs+1)):raise SystemExit('Epochs incomplete: '+str(run))
  checkpoint=YOLO(str(run/'weights/best.pt')).ckpt; metrics=checkpoint.get('train_metrics') or {}
  names=['metrics/mAP50(B)','metrics/mAP50-95(B)']; vals=[float(metrics[n]) for n in names]
  if not all(math.isfinite(v) for v in vals):raise SystemExit('Nonfinite metric: '+str(run))
  row={'seed':seed,'initialization':init,'val_map50':vals[0],'val_map50_95':vals[1],'selection_metric':'validation mAP50-95','epochs':a.epochs,'run_dir':str(run)}
  rows.append(row);cases[init]={'audit':audit,'args':args,'row':row}
 x,y=cases['scratch'],cases['pretrained']
 for key in ('parameters','tensor_shapes_sha256','classification_initial_sha256','torch_rng_sha256'):
  if x['audit'][key]!=y['audit'][key]:raise SystemExit('Initialization mismatch '+str(seed)+' '+key)
 if x['audit']['transferred_tensors']!=0 or y['audit']['transferred_tensors']<=0:raise SystemExit('Transfer count invalid')
 diffs={k:(x['args'].get(k),y['args'].get(k)) for k in (x['args'].keys()|y['args'].keys())-ignored if x['args'].get(k)!=y['args'].get(k)}
 if diffs:raise SystemExit('Training argument mismatch '+repr(diffs))
 pairs.append({'seed':seed,'delta_val_map50':y['row']['val_map50']-x['row']['val_map50'],'delta_val_map50_95':y['row']['val_map50_95']-x['row']['val_map50_95']})
result={'purpose':'exploratory validation only; shared validation data and seeds are not independent test samples','rows':rows,'paired_differences':pairs,'mean_delta_val_map50':statistics.mean(x['delta_val_map50'] for x in pairs),'mean_delta_val_map50_95':statistics.mean(x['delta_val_map50_95'] for x in pairs),'notes':'Checkpoint metrics are the same selected best.pt evaluation, not independent per-metric maxima. No statistical significance or test set estimate is claimed.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print('Audited',len(rows),'runs;',len(pairs),'pairs. Validation only.')
