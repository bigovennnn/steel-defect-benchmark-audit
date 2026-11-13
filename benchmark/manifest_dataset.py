"""Record content hashes for a YOLO detection dataset without modifying it."""
from pathlib import Path
import argparse,hashlib,json
ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
root=a.dataset.resolve();parts=('train','val','test');items=[]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
for part in parts:
 for image in sorted((root/'images'/part).glob('*.jpg')):
  label=root/'labels'/part/(image.stem+'.txt')
  if not label.is_file():raise SystemExit('Missing label: '+str(label))
  items.append({'split':part,'image':image.name,'image_sha256':sha(image),'label_sha256':sha(label)})
canonical=json.dumps(items,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
result={'description':'Exploratory dataset; manifest does not establish image or annotation quality', 'root_at_creation':str(root),'split_manifest_sha256':sha(root/'split.json'),'content_manifest_sha256':hashlib.sha256(canonical).hexdigest(),'images':items}
if a.output.exists():raise SystemExit('Output already exists; refusing overwrite')
a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('images',len(items),'manifest SHA256',result['content_manifest_sha256'])
