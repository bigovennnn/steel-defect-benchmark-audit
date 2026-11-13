"""Download the public Kaggle GC10-DET archive, recording provenance and SHA-256."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, requests
base=Path(__file__).resolve().parent / "downloads"
base.mkdir(exist_ok=True)
url="https://www.kaggle.com/api/v1/datasets/download/alex000kim/gc10det"
target=base / "gc10det_kaggle.zip"
part=target.with_suffix(".zip.part")
if target.exists(): raise SystemExit("Archive already present; not overwriting.")
s=requests.Session();s.trust_env=False
r=s.get(url,stream=True,timeout=(20,120));r.raise_for_status()
if "zip" not in r.headers.get("Content-Type", "").lower(): raise SystemExit("Not a ZIP response: "+r.headers.get("Content-Type", ""))
h=hashlib.sha256(); size=0; report=100*1024*1024
with part.open("wb") as f:
 for chunk in r.iter_content(1024*1024):
  if chunk:
   f.write(chunk);h.update(chunk);size+=len(chunk)
   if size>=report: print(f"downloaded {size//1024//1024} MiB",flush=True);report+=100*1024*1024
expected=r.headers.get("Content-Length")
if expected and size!=int(expected): raise SystemExit("Incomplete download; retained .part")
part.rename(target)
meta={"source":url,"retrieved_utc":datetime.now(timezone.utc).isoformat(),"bytes":size,"sha256":h.hexdigest(),"notes":"Kaggle mirror; confirm origin and redistribution rights before publication."}
(base/"gc10det_provenance.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
print(json.dumps(meta,indent=2),flush=True)
