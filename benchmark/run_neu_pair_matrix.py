"""Sequential, interruption-visible exploratory runs; never evaluates test set."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
base=Path(__file__).resolve().parent
manifest=json.loads((base/'manifest_neu_v1.json').read_text(encoding='utf-8'))
root=base/'datasets/neu_dedup_v1'
for entry in manifest['images']:
 for kind,key,file in [('images','image_sha256',entry['image']),('labels','label_sha256',Path(entry['image']).stem+'.txt')]:
  f=root/kind/entry['split']/file
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=entry[key]:raise SystemExit('Data integrity mismatch: '+str(f))
print('Dataset hashes verified; each case is isolated and only validation is run.',flush=True)
logs=base/'runs/paired_exploratory_logs';logs.mkdir(parents=True,exist_ok=True)
for seed in (0,1,2):
 for init in ('scratch','pretrained'):
  log=logs/f'{init}_seed{seed}.log'
  if log.exists():raise SystemExit('Case log already exists; inspect before restarting: '+str(log))
  cmd=[sys.executable,str(base/'run_neu_pair_case.py'),'--seed',str(seed),'--init',init]
  print('start',init,seed,flush=True)
  with log.open('w',encoding='utf-8') as f:r=subprocess.run(cmd,cwd=base,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode!=0:raise SystemExit(f'Case {init} seed {seed} failed with {r.returncode}; inspect {log}')
  print('complete',init,seed,flush=True)
  if init=='pretrained':
   parent=base/'runs/paired_exploratory_v2'
   x=json.loads((parent/f'neu_100_scratch_seed{seed}'/'initialization_audit.json').read_text(encoding='utf-8'))
   y=json.loads((parent/f'neu_100_pretrained_seed{seed}'/'initialization_audit.json').read_text(encoding='utf-8'))
   for key in ('parameters','tensor_shapes_sha256','classification_initial_sha256','torch_rng_sha256'):
    if x[key]!=y[key]:raise SystemExit(f'Paired initialization mismatch: seed {seed} field {key}')
   if x['transferred_tensors']!=0 or y['transferred_tensors']<=0:raise SystemExit('Unexpected transfer counts')
   print('paired initialization checks pass',seed,flush=True)
print('All requested exploratory cases completed.',flush=True)
