#!/bin/sh
set -eu
mkdir -p /work /out
cd /work
python3 - <<'PY'
import os, urllib.request
u=os.environ['MODEL_URL']
print('Baixando ONNX...')
urllib.request.urlretrieve(u, 'model.onnx')
print('ONNX baixado:', os.path.getsize('model.onnx'), 'bytes')
PY

python3 - <<'PY'
import onnx, os
from rknn.api import RKNN

src='/work/model.onnx'
clean='/work/model_clean.onnx'
out='/out/valorant_bestV2_RK3588_FP.rknn'

m=onnx.load(src)
print('IR/opset:', m.ir_version, [(x.domain, x.version) for x in m.opset_import])
print('input:', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.input])
print('output:', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.output])

symbolic=[]
for vi in m.graph.value_info:
    tt=vi.type.tensor_type
    if tt.HasField('shape'):
        vals=[]
        for d in tt.shape.dim:
            if d.dim_param:
                vals.append(d.dim_param)
        if vals:
            symbolic.append((vi.name, vals))
print('value_info total=', len(m.graph.value_info), 'com dim_param=', len(symbolic))
for name, vals in symbolic[:12]:
    print('SYMBOLIC', name, [repr(x[:220]) for x in vals])

# Intermediate value_info is optional in ONNX. Keep graph inputs/outputs and all nodes/weights unchanged.
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, clean)
print('ONNX limpo salvo:', clean, os.path.getsize(clean), 'bytes', 'value_info=', len(m.graph.value_info))

# Reload and validate the serialized copy itself.
m2=onnx.load(clean)
onnx.checker.check_model(m2)
print('Reload/check OK; input=', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m2.graph.input])

rknn=RKNN(verbose=True)
print('Config RK3588...')
ret=rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
if ret != 0: raise SystemExit(f'config failed: {ret}')
print('Load ONNX limpo...')
ret=rknn.load_onnx(model=clean)
if ret != 0: raise SystemExit(f'load_onnx failed: {ret}')
print('Build RKNN FP...')
ret=rknn.build(do_quantization=False)
if ret != 0: raise SystemExit(f'build failed: {ret}')
print('Export RKNN...')
ret=rknn.export_rknn(out)
if ret != 0: raise SystemExit(f'export failed: {ret}')
rknn.release()
print('RKNN_CONCLUIDO', out, os.path.getsize(out), 'bytes')
PY

cd /out
ls -lah
exec python3 -m http.server "${PORT:-8080}" --bind 0.0.0.0
