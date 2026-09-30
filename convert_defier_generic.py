import os, urllib.request, hashlib, re
from pathlib import Path
import onnx
from rknn.api import RKNN

work=Path('/work'); outdir=Path('/out')
work.mkdir(exist_ok=True); outdir.mkdir(exist_ok=True)

url=os.environ['MODEL_URL']
name=os.environ.get('MODEL_NAME','model.onnx')
base=re.sub(r'[^A-Za-z0-9._-]+','_',Path(name).stem)
src=work/(base+'.onnx')

print('Baixando ONNX...')
urllib.request.urlretrieve(url, src)
print('ONNX baixado:', src.stat().st_size, 'bytes')
print('SHA256 original:', hashlib.sha256(src.read_bytes()).hexdigest())

m=onnx.load(str(src))
def shp(v):
    return [d.dim_value if d.dim_value else d.dim_param for d in v.type.tensor_type.shape.dim]

print('IR/opset:', m.ir_version, [(x.domain, x.version) for x in m.opset_import])
print('producer:', m.producer_name, m.producer_version)
print('input:', [(x.name, shp(x)) for x in m.graph.input])
print('output:', [(x.name, shp(x)) for x in m.graph.output])
print('nodes=', len(m.graph.node), 'initializers=', len(m.graph.initializer), 'value_info=', len(m.graph.value_info))

symbolic=sum(
    1 for vi in m.graph.value_info
    if vi.type.tensor_type.HasField('shape')
    and any(d.dim_param for d in vi.type.tensor_type.shape.dim)
)
print('value_info com dim_param=', symbolic)

clean=work/(base+'_clean.onnx')
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, str(clean))
print('ONNX limpo:', clean.stat().st_size, 'bytes')
print('SHA256 clean:', hashlib.sha256(clean.read_bytes()).hexdigest())

rknn=RKNN(verbose=False)
print('Config RK3588 FP16...')
ret=rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
if ret != 0:
    raise SystemExit(f'config failed: {ret}')

print('Load ONNX...')
ret=rknn.load_onnx(model=str(clean))
if ret != 0:
    raise SystemExit(f'load_onnx failed: {ret}')

print('Build RKNN FP16...')
ret=rknn.build(do_quantization=False)
if ret != 0:
    raise SystemExit(f'build failed: {ret}')

outfile=outdir/(base+'_RK3588_FP16.rknn')
print('Export RKNN...')
ret=rknn.export_rknn(str(outfile))
if ret != 0:
    raise SystemExit(f'export failed: {ret}')
rknn.release()

print('RKNN_FP_CONCLUIDO', outfile, outfile.stat().st_size, 'bytes')
print('SHA256 rknn:', hashlib.sha256(outfile.read_bytes()).hexdigest())
