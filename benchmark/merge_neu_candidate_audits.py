"""Merge existing and supplemental candidate edges, without adjudication or data edits."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--original',type=Path,required=True);ap.add_argument('--supplement',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() or a.metadata.exists():raise SystemExit('Refusing overwrite')
    with a.original.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    supplement=json.loads(a.supplement.read_text(encoding='utf-8'))
    if supplement['supplemental_pairs_verified']!=77791:raise ValueError('Supplement is incomplete')
    seen={tuple(sorted((r['a_file'],r['b_file']))) for r in rows}
    if len(seen)!=len(rows):raise ValueError('Duplicate in original candidates')
    for i,r in enumerate(supplement['review_candidates'],1):
        x,y=r['a'],r['b'];pair=tuple(sorted((x['file'],y['file'])))
        if pair in seen:raise ValueError('Duplicate supplemental pair: '+str(pair))
        seen.add(pair);shift=r['shift_b_xy_pixels'];cross=x['split']!=y['split']
        item={k:'' for k in fields}
        item.update({'candidate_id':f'supp_{i:04d}','a_file':x['file'],'a_split':x['split'],
                     'b_file':y['file'],'b_split':y['split'],'split_pair':x['split']+'/'+y['split'],
                     'cross_split':str(cross),'rotation_b_ccw_degrees':r['rotation_b_ccw_degrees'],
                     'shift_b_dx_pixels':shift[0],'shift_b_dy_pixels':shift[1],
                     'overlap_fraction':r['overlap_fraction'],'pixel_ncc':r['pixel_ncc'],
                     'gradient_ncc':r['gradient_ncc'],'human_review_status':'pending',
                     'human_adjudication':'undetermined','assistant_visual_review_status':'not_recorded_in_merged_file',
                     'review_notes':'Supplemental candidate; retain original evidence and review limitations.'})
        rows.append(item)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    result={'status':'candidate list only; unadjudicated and incomplete for other transforms/coarse misses',
            'original_csv_sha256':sha(a.original),'supplement_json_sha256':sha(a.supplement),
            'merged_csv_sha256':sha(a.output),'original_count':len(rows)-len(supplement['review_candidates']),
            'supplement_count':len(supplement['review_candidates']),'merged_count':len(rows)}
    with a.metadata.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
