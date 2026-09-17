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
from rknn.api import RKNN
import os, sys

out='/out/valorant_bestV2_RK3588_FP.rknn'
rknn = RKNN(verbose=True)
print('Config RK3588...')
ret = rknn.config(
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    target_platform='rk3588'
)
if ret != 0:
    raise SystemExit(f'config failed: {ret}')
print('Load ONNX...')
ret = rknn.load_onnx(model='/work/model.onnx')
if ret != 0:
    raise SystemExit(f'load_onnx failed: {ret}')
print('Build RKNN FP...')
ret = rknn.build(do_quantization=False)
if ret != 0:
    raise SystemExit(f'build failed: {ret}')
print('Export RKNN...')
ret = rknn.export_rknn(out)
if ret != 0:
    raise SystemExit(f'export failed: {ret}')
rknn.release()
print('RKNN_CONCLUIDO', out, os.path.getsize(out), 'bytes')
PY

cd /out
ls -lah
exec python3 -m http.server "${PORT:-8080}" --bind 0.0.0.0
