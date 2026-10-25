import os, re, hashlib, urllib.request, heapq, zipfile, shutil
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import cv2, onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

work=Path('/work'); out=Path('/out')
work.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
src=work/'v8_320_valo_ihack_v3.onnx'
url=os.environ.get('MODEL_URL','')
print('MODEL_URL_SOURCE',url.split('/')[2] if '://' in url else 'none',flush=True)
if url:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=240) as r, src.open('wb') as f: shutil.copyfileobj(r,f,1024*1024)
else:
    import gdown
    gdown.download(id='1ZLJtue94Z4cOf0SfjBdA7bI1viZtYhU3',output=str(src),quiet=False)
size=src.stat().st_size
digest=hashlib.sha256(src.read_bytes()).hexdigest()
print('SOURCE_READY',size,'SHA256',digest,flush=True)
assert size==44607241 and digest=='a11f32d0ecff9c1df51db49852ce0dff71704bd94b1fcd5e8a53ce63d25a34ba', 'Source mismatched'

m=onnx.load(str(src))
g=m.graph
producers={}
for idx,node in enumerate(g.node):
    for k in node.output:
        if k:producers[k]=idx
known={v.name for v in g.input}|{v.name for v in g.initializer}
depends=[set() for _ in g.node];succ=defaultdict(list)
for idx,n in enumerate(g.node):
    for k in n.input:
        if k and k in producers:depends[idx].add(producers[k])
        elif k and k not in known:raise RuntimeError(f'Unknown graph value {k}')
for i,deps in enumerate(depends):
    for dep in deps:succ[dep].append(i)
q=[i for i,deps in enumerate(depends) if not deps];heapq.heapify(q)
order=[]
while q:
    i=heapq.heappop(q);order.append(i)
    for j in succ[i]:
        depends[j].remove(i)
        if not depends[j]:heapq.heappush(q,j)
if len(order)!=len(g.node): raise RuntimeError('Graph order contains cycle')
nodes=list(g.node)
del g.node[:];g.node.extend(nodes[i] for i in order)
onnx.checker.check_model(m)
print('SORTED_GRAPH',len(g.node),len(g.initializer),flush=True)
inf=onnx.shape_inference.infer_shapes(m,strict_mode=False)
meta={v.name:v for v in list(inf.graph.value_info)+list(inf.graph.input)+list(inf.graph.output)}
consumers=defaultdict(list)
for i,n in enumerate(g.node):
    for input_name in n.input:
        consumers[input_name].append(n)
def shp(name):
    v=meta.get(name)
    if not v:return None
    return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
def tp(name):
    v=meta.get(name)
    return v.type.tensor_type.elem_type if v else None
# Select the last convolution feeding the YOLO Detect head's regression/class reshape,
# not intermediate Conv layers of the backbone.
selected=[]
for side in (40,20,10):
    for channels in (64,2):
        matches=[]
        for idx,n in enumerate(g.node):
            if n.op_type!='Conv' or len(n.output)!=1:continue
            output=n.output[0]
            if shp(output)!=[1,channels,side,side]:continue
            # The final head Conv may feed Concat instead of Reshape.
            # Match terminal cv2/cv3 head indices exactly and preserve original shape/type.
            branch='cv2' if channels==64 else 'cv3'
            is_head=int(bool(re.search(r'/'+branch+r'\\.'+str((40,20,10).index(side))+r'\\.[^/]*\\.2/Conv
        if not matches:
            print('MISSING',side,channels,'possible',[(n.name,shp(n.output[0])) for n in g.node if n.op_type=='Conv' and n.output and shp(n.output[0])==[1,channels,side,side]][-12:],flush=True)
            raise RuntimeError(f'No Conv head found shape {[1,channels,side,side]}')
        matches.sort(reverse=True)
        print('HEAD_CANDIDATES',side,channels,matches[-8:],flush=True)
        chosen=matches[0]
        selected.append((chosen[3],[1,channels,side,side],chosen[4]))
print('CHOSEN_OUTPUTS',selected,flush=True)
assert len({k[0] for k in selected})==6
assert all(k[2] in (TensorProto.FLOAT,TensorProto.FLOAT16) for k in selected)
del g.output[:]
for name,shape,typ in selected:
    g.output.append(helper.make_tensor_value_info(name,typ,shape))
required={x[0] for x in selected}
kept=[]
for node in reversed(g.node):
    if any(x in required for x in node.output if x):
        kept.append(node);required.update(x for x in node.input if x)
kept.reverse()
del g.node[:];g.node.extend(kept)
needed={x for node in g.node for x in node.input}
weights=[i for i in g.initializer if i.name in needed]
del g.initializer[:];g.initializer.extend(weights)
del g.value_info[:]
onnx.checker.check_model(m)
raw=work/'v8_320_valo_ihack_v3_RAW6.onnx'
onnx.save(m,str(raw))
print('RAW_ONNX_READY',raw.stat().st_size,'NODES',len(g.node),'WEIGHTS',len(g.initializer),flush=True)

# Validation on synthetic input: correctness of model structure and shape, not detection quality.
import onnxruntime as ort
session=ort.InferenceSession(str(raw),providers=['CPUExecutionProvider'])
inp=session.get_inputs()[0]
print('INPUT',inp.name,inp.shape,inp.type,flush=True)
assert inp.shape==[1,3,320,320]
image=np.zeros((1,3,320,320),dtype=np.float16 if inp.type=='tensor(float16)' else np.float32)
results=session.run(None,{inp.name:image})
for i,(output,expected) in enumerate(zip(results,selected)):
    if list(output.shape)!=expected[1] or not np.all(np.isfinite(output)): raise RuntimeError(f'Bad output {i}')
print('ONNX_RUNTIME_OK',[list(z.shape) for z in results],flush=True)

archive=work/'coco128.zip'
with urllib.request.urlopen('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',timeout=180) as r,archive.open('wb') as f:shutil.copyfileobj(r,f)
imgs_dir=work/'calibration';imgs_dir.mkdir(exist_ok=True)
dataset=[]
with zipfile.ZipFile(archive) as z:
    entries=sorted(s for s in z.namelist() if '/images/' in s and s.lower().endswith(('.jpg','.jpeg','.png')))
    assert len(entries)>=100
    for i in range(200):
        raw_img=np.frombuffer(z.read(entries[i%len(entries)]),np.uint8)
        img=cv2.imdecode(raw_img,cv2.IMREAD_COLOR)
        h,w=img.shape[:2];scale=min(320/w,320/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
        res=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        canvas=np.full((320,320,3),114,dtype=np.uint8);x=(320-nw)//2;y=(320-nh)//2
        canvas[y:y+nh,x:x+nw]=res
        if i>=len(entries):canvas=cv2.flip(canvas,1)
        p=imgs_dir/f'c_{i:03d}.jpg'
        assert cv2.imwrite(str(p),canvas)
        dataset.append(str(p))
data_file=work/'dataset.txt';data_file.write_text('\n'.join(dataset)+'\n')
print('CALIBRATION_COUNT',len(dataset),'GENERIC_COCO128',flush=True)

rknn=RKNN(verbose=False)
try:
    rc=rknn.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[255,255,255]],quantized_dtype='asymmetric_quantized-8')
    print('CONFIG_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('config failed')
    rc=rknn.load_onnx(model=str(raw));print('LOAD_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('load failed')
    rc=rknn.build(do_quantization=True,dataset=str(data_file))
    print('BUILD_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('build failed')
    dest=out/'v8_320_valo_ihack_v3_RK3588_INT8_RAW6.rknn'
    rc=rknn.export_rknn(str(dest))
    print('EXPORT_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('export failed')
    print('RKNN_INT8_CONCLUIDO',str(dest),dest.stat().st_size,flush=True)
    print('RKNN_SHA256',hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
finally:rknn.release()
,n.name)))
            matches.append((is_head,idx,n.name,output,tp(output)))
        if not matches:
            print('MISSING',side,channels,'possible',[(n.name,shp(n.output[0])) for n in g.node if n.op_type=='Conv' and n.output and shp(n.output[0])==[1,channels,side,side]][-12:],flush=True)
            raise RuntimeError(f'No Conv head found shape {[1,channels,side,side]}')
        matches.sort(reverse=True)
        print('HEAD_CANDIDATES',side,channels,matches[-8:],flush=True)
        chosen=matches[0]
        selected.append((chosen[3],[1,channels,side,side],chosen[4]))
print('CHOSEN_OUTPUTS',selected,flush=True)
assert len({k[0] for k in selected})==6
assert all(k[2] in (TensorProto.FLOAT,TensorProto.FLOAT16) for k in selected)
del g.output[:]
for name,shape,typ in selected:
    g.output.append(helper.make_tensor_value_info(name,typ,shape))
required={x[0] for x in selected}
kept=[]
for node in reversed(g.node):
    if any(x in required for x in node.output if x):
        kept.append(node);required.update(x for x in node.input if x)
kept.reverse()
del g.node[:];g.node.extend(kept)
needed={x for node in g.node for x in node.input}
weights=[i for i in g.initializer if i.name in needed]
del g.initializer[:];g.initializer.extend(weights)
del g.value_info[:]
onnx.checker.check_model(m)
raw=work/'v8_320_valo_ihack_v3_RAW6.onnx'
onnx.save(m,str(raw))
print('RAW_ONNX_READY',raw.stat().st_size,'NODES',len(g.node),'WEIGHTS',len(g.initializer),flush=True)

# Validation on synthetic input: correctness of model structure and shape, not detection quality.
import onnxruntime as ort
session=ort.InferenceSession(str(raw),providers=['CPUExecutionProvider'])
inp=session.get_inputs()[0]
print('INPUT',inp.name,inp.shape,inp.type,flush=True)
assert inp.shape==[1,3,320,320]
image=np.zeros((1,3,320,320),dtype=np.float16 if inp.type=='tensor(float16)' else np.float32)
results=session.run(None,{inp.name:image})
for i,(output,expected) in enumerate(zip(results,selected)):
    if list(output.shape)!=expected[1] or not np.all(np.isfinite(output)): raise RuntimeError(f'Bad output {i}')
print('ONNX_RUNTIME_OK',[list(z.shape) for z in results],flush=True)

archive=work/'coco128.zip'
with urllib.request.urlopen('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',timeout=180) as r,archive.open('wb') as f:shutil.copyfileobj(r,f)
imgs_dir=work/'calibration';imgs_dir.mkdir(exist_ok=True)
dataset=[]
with zipfile.ZipFile(archive) as z:
    entries=sorted(s for s in z.namelist() if '/images/' in s and s.lower().endswith(('.jpg','.jpeg','.png')))
    assert len(entries)>=100
    for i in range(200):
        raw_img=np.frombuffer(z.read(entries[i%len(entries)]),np.uint8)
        img=cv2.imdecode(raw_img,cv2.IMREAD_COLOR)
        h,w=img.shape[:2];scale=min(320/w,320/h);nw=max(1,round(w*scale));nh=max(1,round(h*scale))
        res=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        canvas=np.full((320,320,3),114,dtype=np.uint8);x=(320-nw)//2;y=(320-nh)//2
        canvas[y:y+nh,x:x+nw]=res
        if i>=len(entries):canvas=cv2.flip(canvas,1)
        p=imgs_dir/f'c_{i:03d}.jpg'
        assert cv2.imwrite(str(p),canvas)
        dataset.append(str(p))
data_file=work/'dataset.txt';data_file.write_text('\n'.join(dataset)+'\n')
print('CALIBRATION_COUNT',len(dataset),'GENERIC_COCO128',flush=True)

rknn=RKNN(verbose=False)
try:
    rc=rknn.config(target_platform='rk3588',mean_values=[[0,0,0]],std_values=[[255,255,255]],quantized_dtype='asymmetric_quantized-8')
    print('CONFIG_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('config failed')
    rc=rknn.load_onnx(model=str(raw));print('LOAD_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('load failed')
    rc=rknn.build(do_quantization=True,dataset=str(data_file))
    print('BUILD_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('build failed')
    dest=out/'v8_320_valo_ihack_v3_RK3588_INT8_RAW6.rknn'
    rc=rknn.export_rknn(str(dest))
    print('EXPORT_RETURN',rc,flush=True)
    if rc!=0:raise RuntimeError('export failed')
    print('RKNN_INT8_CONCLUIDO',str(dest),dest.stat().st_size,flush=True)
    print('RKNN_SHA256',hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
finally:rknn.release()
