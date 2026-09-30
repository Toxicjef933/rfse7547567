#!/bin/sh
set -eu
mkdir -p /work /out
cd /work

python3 - <<'PY'
import os, urllib.request
u=os.environ.get('MODEL_URL','')
if not u:
    print('MODEL_URL ausente; aguardando configuracao...')
    raise SystemExit(2)
print('Baixando ONNX...')
urllib.request.urlretrieve(u, 'model.onnx')
print('ONNX baixado:', os.path.getsize('model.onnx'), 'bytes')
PY

python3 - <<'PY'
import onnx, os, hashlib
from rknn.api import RKNN

src='/work/model.onnx'
clean='/work/GGMWarzone_clean.onnx'
out='/out/GGMWarzone_RK3588_FP16.rknn'

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

# Intermediate value_info is optional. Preserve inputs, outputs, nodes and weights.
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, clean)
print('ONNX limpo:', clean, os.path.getsize(clean), 'bytes')
print('SHA256 clean:', hashlib.sha256(open(clean,'rb').read()).hexdigest())

rknn=RKNN(verbose=True)
print('Config RK3588 FP...')
ret=rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
if ret != 0: raise SystemExit(f'config failed: {ret}')

print('Load ONNX...')
ret=rknn.load_onnx(model=clean)
if ret != 0: raise SystemExit(f'load_onnx failed: {ret}')

print('Build RKNN FP...')
ret=rknn.build(do_quantization=False)
if ret != 0: raise SystemExit(f'build failed: {ret}')

print('Export RKNN...')
ret=rknn.export_rknn(out)
if ret != 0: raise SystemExit(f'export failed: {ret}')
rknn.release()

print('RKNN_FP_CONCLUIDO', out, os.path.getsize(out), 'bytes')
print('SHA256 rknn:', hashlib.sha256(open(out,'rb').read()).hexdigest())
PY

cd /out
ls -lah
exec python3 -m http.server "${PORT:-8080}" --bind 0.0.0.0
