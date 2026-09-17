#!/bin/sh
set -eu
mkdir -p /work /out /work/calib
cd /work

python3 - <<'PY'
import os, urllib.request
u=os.environ['MODEL_URL']
print('Baixando ONNX...')
urllib.request.urlretrieve(u, 'model.onnx')
print('ONNX baixado:', os.path.getsize('model.onnx'), 'bytes')

names = [
'000000005001.jpg','000000038829.jpg','000000052891.jpg','000000075612.jpg','000000098261.jpg',
'000000181542.jpg','000000215245.jpg','000000277005.jpg','000000288685.jpg','000000301421.jpg',
'000000334371.jpg','000000348481.jpg','000000373353.jpg','000000397681.jpg','000000414673.jpg',
'000000419312.jpg','000000465822.jpg','000000475732.jpg','000000559707.jpg','000000574315.jpg']
base='https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/datasets/COCO/subset/'
paths=[]
for n in names:
    p='/work/calib/'+n
    urllib.request.urlretrieve(base+n, p)
    paths.append(p)
print('Calibracao baixada:', len(paths), 'imagens')
with open('/work/dataset.txt','w') as f:
    f.write('\n'.join(paths)+'\n')
PY

python3 - <<'PY'
import onnx, os, hashlib
from rknn.api import RKNN

src='/work/model.onnx'
clean='/work/v8_640_valo_v3_clean_640.onnx'
out='/out/v8_640_valo_v3_Defier_INT8_640.rknn'
dataset='/work/dataset.txt'

m=onnx.load(src)
print('IR/opset:', m.ir_version, [(x.domain, x.version) for x in m.opset_import])
print('producer:', m.producer_name, m.producer_version)
print('input:', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.input])
print('output:', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.output])
print('nodes=', len(m.graph.node), 'initializers=', len(m.graph.initializer), 'value_info=', len(m.graph.value_info))

symbolic=0
for vi in m.graph.value_info:
    tt=vi.type.tensor_type
    if tt.HasField('shape') and any(d.dim_param for d in tt.shape.dim):
        symbolic += 1
print('value_info com dim_param=', symbolic)

# value_info intermediario e opcional no ONNX; remove apenas anotacoes de shape,
# preservando input, output, nodes, pesos e arquitetura.
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, clean)
print('ONNX limpo:', clean, os.path.getsize(clean), 'bytes')
print('SHA256 clean:', hashlib.sha256(open(clean,'rb').read()).hexdigest())

m2=onnx.load(clean)
onnx.checker.check_model(m2)
print('Reload/check OK')

rknn=RKNN(verbose=True)
print('Config RK3588 INT8...')
ret=rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
if ret != 0: raise SystemExit(f'config failed: {ret}')

print('Load ONNX limpo...')
ret=rknn.load_onnx(model=clean)
if ret != 0: raise SystemExit(f'load_onnx failed: {ret}')

print('Build RKNN INT8...')
ret=rknn.build(do_quantization=True, dataset=dataset)
if ret != 0: raise SystemExit(f'build failed: {ret}')

print('Export RKNN INT8...')
ret=rknn.export_rknn(out)
if ret != 0: raise SystemExit(f'export failed: {ret}')
rknn.release()
print('RKNN_INT8_CONCLUIDO', out, os.path.getsize(out), 'bytes')
print('SHA256 rknn:', hashlib.sha256(open(out,'rb').read()).hexdigest())
PY

cd /out
ls -lah
exec python3 -m http.server "${PORT:-8080}" --bind 0.0.0.0
