"""Render aligned candidate boxes for visual audit, not an annotation verdict."""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from audit_candidate_box_overlap import load_boxes,SIZE

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--id',required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise SystemExit('Refusing overwrite')
    with a.candidates.open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['candidate_id']==a.id]
    if len(rows)!=1:raise ValueError('Candidate ID must be unique')
    r=rows[0];rot=int(r['rotation_b_ccw_degrees']);dx=int(r['shift_b_dx_pixels']);dy=int(r['shift_b_dy_pixels'])
    if rot not in (0,90,180,270):raise ValueError('Unsupported rotation')
    def image(part,name):
        if Path(name).name!=name:raise ValueError('Invalid filename')
        with Image.open(a.dataset/'images'/part/name) as im:
            if im.size!=(200,200):raise ValueError('Unexpected image dimensions')
            return np.asarray(im.convert('L'))
    left=image(r['a_split'],r['a_file']);right=np.rot90(image(r['b_split'],r['b_file']),rot//90)
    x0,x1=max(0,-dx),min(200,200-dx);y0,y1=max(0,-dy),min(200,200-dy)
    aligned=np.zeros_like(right);aligned[y0:y1,x0:x1]=right[y0+dy:y1+dy,x0+dx:x1+dx]
    region=(x0,y0,x1,y1);boxes_a,_=load_boxes(a.dataset,r['a_split'],r['a_file'],0,0,0,region,.5)
    boxes_b,_=load_boxes(a.dataset,r['b_split'],r['b_file'],rot//90,dx,dy,region,.5)
    panels=[Image.fromarray(left).convert('RGB'),Image.fromarray(aligned).convert('RGB'),Image.fromarray(left).convert('RGB')]
    def draw_boxes(im,boxes,color):
        d=ImageDraw.Draw(im)
        for cls,box in boxes:
            d.rectangle(tuple(round(v) for v in box),outline=color,width=2);d.text((round(box[0]),round(box[1])),str(cls),fill=color)
        d.rectangle(region,outline='yellow',width=1)
    draw_boxes(panels[0],boxes_a,'red');draw_boxes(panels[1],boxes_b,'cyan')
    draw_boxes(panels[2],boxes_a,'red');draw_boxes(panels[2],boxes_b,'cyan')
    scale=2;canvas=Image.new('RGB',(1200,440),'white');d=ImageDraw.Draw(canvas)
    for i,p in enumerate(panels):canvas.paste(p.resize((400,400),Image.Resampling.NEAREST),(400*i,40))
    d.text((5,4),f"{a.id}: A red | aligned B cyan | overlay; yellow = overlap; >=50% visible",fill='black')
    d.text((5,21),f"A {r['a_file']} | B {r['b_file']} | rotation {rot}, offset ({dx},{dy})",fill='black')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
    print('saved',a.output,'visible boxes',len(boxes_a),len(boxes_b))
if __name__=='__main__':main()
