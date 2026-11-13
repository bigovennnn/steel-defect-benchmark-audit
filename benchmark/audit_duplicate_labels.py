"""Report exact duplicate YOLO annotation rows; never alter source files."""
from pathlib import Path
from collections import Counter
import json
base=Path(__file__).resolve().parent
result={}
for name in ('neu_dedup_v1','gc10_review_v1'):
 rows=[]
 for part in ('train','val','test'):
  for label in sorted((base/'datasets'/name/'labels'/part).glob('*.txt')):
   lines=[tuple(line.split()) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
   duplicates=[{'label':' '.join(k),'count':n} for k,n in Counter(lines).items() if n>1]
   if duplicates:rows.append({'split':part,'file':label.name,'duplicates':duplicates})
 result[name]=rows
 print(name,'duplicate-label files',len(rows))
out=base/'duplicate_label_audit.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
