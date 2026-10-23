import os
import hashlib
import zipfile
from pathlib import Path
import cv2
import numpy as np
import onnx
from rknn.api import RKNN

root = Path('/work')
out = Path('/out')
root.mkdir(exist_ok=True)
out.mkdir(exist_ok=True)
archive = root / 'coco128.zip'
assert archive.exists(), 'COCO128 missing'
with zipfile.ZipFile(archive) as z:
    files = [x for x in z.namelist() if x.lower().endswith(('.jpg','.jpeg','.png')) and '/images/' in x.replace('\\','/')]
    print('COCO_IMAGES',len(files),flush=True)
    if len(files)<100: raise RuntimeError('Too few calibration images')
    imgs = sorted(files)
    dest = root / 'calibration'
    dest.mkdir(exist_ok=True)
    samples = []
    for i in range(200):
        f = imgs[i % len(imgs)]
        raw = np.frombuffer(z.read(f), np.uint8)
        image = cv2.imdecode(raw,cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f'Decode failed: {f}')
        h,w = image.shape[:2]
        ratio = min(320 / h, 320 / w)
        nh,nw = max(1,round(h*ratio)),max(1,round(w*ratio))
        resized=cv2.resize(image, (nw,nh), interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
        canvas=np.full((320,320,3),114,dtype=np.uint8)
        y=(320-nh)//2; x=(320-nw)//2
        canvas[y:y+nh,x:x+nw]=resized
        if i>=len(imgs): canvas=cv2.flip(canvas,1)
        target=dest/f'cal_{i:03d}.jpg'
        if not cv2.imwrite(str(target),canvas,[cv2.IMWRITE_JPEG_QUALITY,92]):
            raise RuntimeError(f'Write failed {target}')
        samples.append(str(target))
dataset = root/'dataset.txt'
dataset.write_text('\n'.join(samples)+'\n')
print('CALIBRATION_DATASET',len(samples), 'IMAGES', 'source=COCO128 + flips', flush=True)
print('CALIBRATION_LIST',str(dataset),flush=True)

source=root/'exp13_sorted.onnx'
model=onnx.load(str(source))
onnx.checker.check_model(model)
def shp(v):
    return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
print('ONNX_SHAPES',[(x.name,shp(x)) for x in model.graph.input],[(x.name,shp(x)) for x in model.graph.output],flush=True)
assert len(model.graph.input)==1 and shp(model.graph.input[0]) == [1,3,320,320]
assert len(model.graph.output)==1 and shp(model.graph.output[0]) == [1,5,2100]
print('ONNX_GRAPH',len(model.graph.node),len(model.graph.initializer),flush=True)

r=RKNN(verbose=False)
try:
    x=r.config(target_platform='rk3588', mean_values=[[0,0,0]], std_values=[[255,255,255]], quantized_dtype='asymmetric_quantized-8')
    print('CONFIG_RET',x,flush=True)
    if x!=0: raise RuntimeError('config failed')
    x=r.load_onnx(model=str(source))
    print('LOAD_RET',x,flush=True)
    if x!=0: raise RuntimeError('load failed')
    x=r.build(do_quantization=True,dataset=str(dataset))
    print('BUILD_QUANT_RET',x,flush=True)
    if x!=0: raise RuntimeError('build int8 failed')
    target=out/'exp-13_RK3588_INT8.rknn'
    x=r.export_rknn(str(target))
    print('EXPORT_RET',x,flush=True)
    if x!=0: raise RuntimeError('export failed')
    print('RKNN_INT8_CONCLUIDO',target,target.stat().st_size,flush=True)
    print('SHA256_RKNN',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
finally:
    r.release()
