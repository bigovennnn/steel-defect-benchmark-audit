"""Merge a completed coarse-band candidate audit with a prior candidate CSV."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True,type=Path);p.add_argument('--band',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--metadata',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists() or a.metadata.exists():raise SystemExit('Refusing overwrite')
    with a.original.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
    band=json.loads(a.band.read_text(encoding='utf-8'))
    if band.get('pairs_verified')!=96572 or not band.get('input_images_and_split_unchanged'):raise ValueError('Band is incomplete or input changed')
    pairs={tuple(sorted((r['a_file'],r['b_file']))) for r in rows};ids={r['candidate_id'] for r in rows}
    if len(pairs)!=len(rows) or len(ids)!=len(rows):raise ValueError('Duplicate original candidates')
    for i,r in enumerate(band['review_candidates'],1):
        x,y=r['a'],r['b'];key=tuple(sorted((x['file'],y['file'])))
        if key in pairs:raise ValueError('Band pair already in original candidates')
        pairs.add(key);shift=r['shift_b_xy_pixels'];item={k:'' for k in fields}
        item.update({'candidate_id':f'band_{i:04d}','a_file':x['file'],'a_split':x['split'],
                     'b_file':y['file'],'b_split':y['split'],'split_pair':x['split']+'/'+y['split'],
                     'cross_split':str(x['split']!=y['split']),'rotation_b_ccw_degrees':r['rotation_b_ccw_degrees'],
                     'shift_b_dx_pixels':shift[0],'shift_b_dy_pixels':shift[1],'overlap_fraction':r['overlap_fraction'],
                     'pixel_ncc':r['pixel_ncc'],'gradient_ncc':r['gradient_ncc'],'human_review_status':'pending',
                     'human_adjudication':'undetermined','assistant_visual_review_status':'not_recorded_in_merged_file',
                     'review_notes':'Coarse band candidate; see band audit for exact evidence and limitations.'})
        if item['candidate_id'] in ids:raise ValueError('Duplicate candidate ID')
        ids.add(item['candidate_id']);rows.append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={'status':'unadjudicated candidates; other transforms and coarse scores remain unscreened',
            'original_sha256':sha(a.original),'band_audit_sha256':sha(a.band),'merged_sha256':sha(a.output),
            'original_count':len(rows)-len(band['review_candidates']),
            'band_count':len(band['review_candidates']),'merged_count':len(rows)}
    with a.metadata.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
