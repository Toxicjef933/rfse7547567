import os, urllib.request, hashlib
from pathlib import Path
import onnx
from rknn.api import RKNN

work=Path("/work"); out=Path("/out")
work.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
url=os.environ["MODEL_URL"]
src=work/"valorant_bestV2_original.onnx"
fixed=work/"valorant_bestV2_640x640_static.onnx"
dest=out/"valorant_bestV2_RK3588_640_FP16.rknn"

print("DOWNLOADING_ONNX", flush=True)
req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
with urllib.request.urlopen(req,timeout=180) as response, open(src,"wb") as dst:
    while True:
        chunk=response.read(1024*1024)
        if not chunk: break
        dst.write(chunk)
print("DOWNLOADED",src.stat().st_size,hashlib.sha256(src.read_bytes()).hexdigest(),flush=True)

m=onnx.load(str(src))
def dims(v):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
print("ORIGINAL",[(x.name,dims(x)) for x in m.graph.input],[(x.name,dims(x)) for x in m.graph.output], flush=True)
print("METADATA",m.ir_version,[(x.domain,x.version) for x in m.opset_import],len(m.graph.node),len(m.graph.initializer), flush=True)
assert len(m.graph.input)==1, "Expected one input"
i=m.graph.input[0].type.tensor_type.shape.dim
assert len(i)==4, "Expected 4D NCHW input"
for d,n in zip(i,[1,3,640,640]):
    d.ClearField("dim_param")
    d.dim_value=n
if len(m.graph.output)==1:
    o=m.graph.output[0].type.tensor_type.shape.dim
    if len(o)==3:
        for d,n in zip(o,[1,5,8400]):
            d.ClearField("dim_param")
            d.dim_value=n
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m,str(fixed))
print("FIXED_SHAPES",[(x.name,dims(x)) for x in m.graph.input],[(x.name,dims(x)) for x in m.graph.output],flush=True)
print("STATIC_SHA256",hashlib.sha256(fixed.read_bytes()).hexdigest(),flush=True)

r=RKNN(verbose=False)
try:
    rc=r.config(mean_values=[[0,0,0]],std_values=[[255,255,255]],target_platform="rk3588")
    print("CONFIG_RESULT",rc,flush=True)
    if rc!=0: raise RuntimeError("Config failed")
    rc=r.load_onnx(model=str(fixed))
    print("LOAD_RESULT",rc,flush=True)
    if rc!=0: raise RuntimeError("Load failed")
    rc=r.build(do_quantization=False)
    print("BUILD_RESULT",rc,flush=True)
    if rc!=0: raise RuntimeError("Build failed")
    rc=r.export_rknn(str(dest))
    print("EXPORT_RESULT",rc,flush=True)
    if rc!=0: raise RuntimeError("Export failed")
    print("RKNN_FP_CONCLUIDO",str(dest),dest.stat().st_size,"bytes",flush=True)
    print("SHA256_RKNN",hashlib.sha256(dest.read_bytes()).hexdigest(),flush=True)
finally:
    r.release()
