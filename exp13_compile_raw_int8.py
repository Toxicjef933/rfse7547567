import zipfile,hashlib
from pathlib import Path
import cv2, numpy as np, onnx
from rknn.api import RKNN

root=Path('/work')
out=Path('/out')
with zipfile.ZipFile(root/'coco128.zip') as z:
    files=sorted(x for x in z.namelist() if x.lower().endswith(('.jpg','.jpeg','.png')) and '/images/' in x.replace('\\','/'))
    assert len(files)>=100,len(files)
    cal=root/'calibration_raw'
    cal.mkdir(exist_ok=True)
    names=[]
    for i in range(200):
        v=np.frombuffer(z.read(files[i % len(files)]),np.uint8)
        im=cv2.imdecode(v,cv2.IMREAD_COLOR)
        if im is None: raise ValueError(files[i % len(files)])
        h,w=im.shape[:2]
        ratio=min(320/h,320/w)
        h1,w1=max(1,round(h*ratio)),max(1,round(w*ratio))
        im=cv2.resize(im,(w1,h1),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
        canvas=np.full((320,320,3),114,dtype=np.uint8)
        x,y=(320-w1)//2,(320-h1)//2
        canvas[y:y+h1,x:x+w1]=im
        if i>=len(files):canvas=cv2.flip(canvas,1)
        path=cal/f'{i:03d}.jpg'
        cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
        names.append(str(path))
dataset=root/'dataset_raw.txt'
dataset.write_text('\n'.join(names)+'\n')
print('CALIBRATION_DATASET',len(names),flush=True)

for mode in (9,6):
    path=root/f'exp13_raw{mode}.onnx'
    m=onnx.load(str(path))
    onnx.checker.check_model(m)
    print('INPUT_OUTPUT',mode,[(v.name,[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.output],flush=True)
    r=RKNN(verbose=False)
    try:
        rc=r.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[255,255,255]],quantized_dtype='asymmetric_quantized-8')
        if rc!=0: raise RuntimeError(f'config({mode}) {rc}')
        rc=r.load_onnx(model=str(path))
        if rc!=0: raise RuntimeError(f'load_onnx({mode}) {rc}')
        rc=r.build(do_quantization=True,dataset=str(dataset))
        print('BUILD_RET',mode,rc,flush=True)
        if rc!=0: raise RuntimeError(f'build({mode}) {rc}')
        target=out/f'exp-13_RK3588_INT8_RAW{mode}.rknn'
        rc=r.export_rknn(str(target))
        print('EXPORT_RET',mode,rc,flush=True)
        if rc!=0:raise RuntimeError(f'export({mode}) {rc}')
        print('RKNN_RAW_CONCLUIDO',mode,str(target),target.stat().st_size,flush=True)
        print('SHA256',mode,hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
    finally:
        r.release()
