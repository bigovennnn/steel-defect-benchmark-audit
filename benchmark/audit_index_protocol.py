"""Assess precomputed image-pair candidates against a reported filename-index partition.

Purely image-level retrospective audit: no training or inference, no author data assumed.
"""
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def parse(name):
    m=re.fullmatch(r'(.+)_(\d+)\.jpg',name)
    if not m:raise ValueError('unexpected filename '+name)
    return m.group(1),int(m.group(2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Refusing overwrite')
    with args.candidates.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    cross=[]
    for row in rows:
        ca,ia=parse(row['a_file']);cb,ib=parse(row['b_file'])
        if not 1<=ia<=300 or not 1<=ib<=300:raise ValueError('index out of range')
        if (ia<=240)!=(ib<=240):
            dev=row['a_file'] if ia<=240 else row['b_file'];test=row['b_file'] if ia<=240 else row['a_file']
            cross.append({'candidate_id':row['candidate_id'],'development_image':dev,'test_image':test,
                          'class_a':ca,'class_b':cb,'pixel_ncc':float(row['pixel_ncc']),
                          'gradient_ncc':float(row['gradient_ncc']),'overlap_fraction':float(row['overlap_fraction']),
                          'visual_review':'not individually adjudicated by this script'})
    out={'status':'exploratory; candidate pairs are not automatically confirmed near duplicates',
         'protocol_source':'https://doi.org/10.1038/s41598-026-64588-4',
         'protocol':'development indices 1-240, test indices 241-300 per class',
         'scope':'local neu_clean_v2 candidate pool only; authors image identities and actual splits unverified',
         'input_csv_sha256':sha(args.candidates),'input_candidate_pairs':len(rows),
         'cross_index_candidate_pairs':len(cross),'cross_index_unique_development_images':len({r['development_image'] for r in cross}),
         'cross_index_unique_test_images':len({r['test_image'] for r in cross}),
         'class_pairs':dict(sorted(Counter(r['class_a']+' / '+r['class_b'] for r in cross).items())),
         'pairs':cross}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in out.items() if k!='pairs'},ensure_ascii=False))

if __name__=='__main__':main()
