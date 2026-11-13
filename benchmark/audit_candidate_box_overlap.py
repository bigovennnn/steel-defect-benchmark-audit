"""Compare bounding boxes over aligned candidate image overlap, not annotation truth.

Quarter-turn geometry and offsets follow audit_neu_rigid_candidates.overlap_regions.
No model is run and no annotation is modified. Counts are sensitive to clipping rules.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

SIZE=200.0
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rotate_point(x,y,k):
    if k==0:return x,y
    if k==1:return y,SIZE-x
    if k==2:return SIZE-x,SIZE-y
    if k==3:return SIZE-y,x
    raise ValueError('rotation must be a quarter turn')
def box_transform(box,k,dx,dy):
    x0,y0,x1,y1=box
    p=[rotate_point(x,y,k) for x in (x0,x1) for y in (y0,y1)]
    return (min(x for x,y in p)-dx,min(y for x,y in p)-dy,max(x for x,y in p)-dx,max(y for x,y in p)-dy)
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def clip(b,r):return max(b[0],r[0]),max(b[1],r[1]),min(b[2],r[2]),min(b[3],r[3])
def load_boxes(dataset,part,name,k,dx,dy,region,visible):
    if Path(name).name!=name or not name.endswith('.jpg'):raise ValueError('bad filename')
    p=dataset/'labels'/part/(Path(name).stem+'.txt');out=[];hidden=0
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        cls,cx,cy,w,h=map(float,line.split())
        if int(cls)!=cls or not 0<=int(cls)<6:raise ValueError('bad class')
        orig=((cx-w/2)*SIZE,(cy-h/2)*SIZE,(cx+w/2)*SIZE,(cy+h/2)*SIZE)
        transformed=box_transform(orig,k,dx,dy);clipped=clip(transformed,region);full=area(transformed)
        if full<=0:raise ValueError('empty box')
        if area(clipped)/full>=visible:out.append((int(cls),clipped))
        else:hidden+=1
    return out,hidden
def iou(a,b):
    intersection=area(clip(a,b));union=area(a)+area(b)-intersection
    return intersection/union if union>0 else 0.0
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--visible',type=float,default=.5);p.add_argument('--iou',type=float,default=.5)
    a=p.parse_args()
    if a.output.exists() or not 0<a.visible<=1 or not 0<a.iou<=1:raise SystemExit('Invalid output or thresholds')
    # Pixel-coordinate geometry checks, including the shift sign used by overlap_regions.
    assert box_transform((10,20,30,40),1,0,0)==(20,170,40,190)
    assert box_transform((10,20,30,40),0,5,-3)==(5,23,25,43)
    assert abs(iou((0,0,10,10),(0,0,10,10))-1)<1e-12
    with a.candidates.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    result=[]
    for r in rows:
        k=int(r['rotation_b_ccw_degrees'])//90;degrees=int(r['rotation_b_ccw_degrees'])
        if degrees not in (0,90,180,270):raise ValueError('invalid rotation')
        dx,dy=int(r['shift_b_dx_pixels']),int(r['shift_b_dy_pixels'])
        region=(max(0,-dx),max(0,-dy),min(SIZE,SIZE-dx),min(SIZE,SIZE-dy))
        left,lh=load_boxes(a.dataset,r['a_split'],r['a_file'],0,0,0,region,a.visible)
        right,rh=load_boxes(a.dataset,r['b_split'],r['b_file'],k,dx,dy,region,a.visible)
        scores=np.zeros((len(left),len(right)),dtype=float);raw=np.zeros_like(scores)
        for i,(c1,b1) in enumerate(left):
            for j,(c2,b2) in enumerate(right):
                v=iou(b1,b2);raw[i,j]=v
                if c1==c2 and v>=a.iou:scores[i,j]=1+v*.001
        matches=[]
        if scores.size:
            ii,jj=linear_sum_assignment(-scores)
            matches=[{'a_index':int(i),'b_index':int(j),'class':left[i][0],'iou':float(raw[i,j])} for i,j in zip(ii,jj) if scores[i,j]>0]
        result.append({'candidate_id':r['candidate_id'],'a_file':r['a_file'],'b_file':r['b_file'],
                       'a_split':r['a_split'],'b_split':r['b_split'],'a_visible_boxes':len(left),'b_visible_boxes':len(right),
                       'a_below_visibility':lh,'b_below_visibility':rh,'matched_boxes':len(matches),'matches':matches,
                       'unmatched_a':len(left)-len(matches),'unmatched_b':len(right)-len(matches)})
    out={'status':'exploratory annotation correspondence; no adjudication of annotation correctness',
         'candidate_csv_sha256':sha(a.candidates),'dataset_split_sha256':sha(a.dataset/'split.json'),
         'visible_fraction_threshold':a.visible,'same_class_iou_threshold':a.iou,'candidate_pairs':len(result),
         'pairs_with_unmatched_visible_boxes':sum(x['unmatched_a']>0 or x['unmatched_b']>0 for x in result),
         'pairs_with_no_visible_boxes_in_either_image':sum(x['a_visible_boxes']==0 and x['b_visible_boxes']==0 for x in result),
         'limits':'Box clipping, visibility and IoU thresholds affect counts; candidate pairs and annotations are not independently adjudicated.',
         'pairs':result}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in out.items() if k!='pairs'},ensure_ascii=False))
if __name__=='__main__':main()
