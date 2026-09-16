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
cat > meta.yaml <<'YAML'
delegate: npu
data_layout: default
inputs:
  - name: images
    shape: [1, 3, 640, 640]
    means: [0, 0, 0]
    scale: 255
    format: rgb keep_proportions=1
outputs:
  - name: output0
    format: yolov8 transposed=1 w_scale=640 h_scale=640
YAML

echo 'Iniciando compilacao SyNAP para SL1680...'
synap convert --model model.onnx --target SL1680 --meta meta.yaml --out-dir /out
cp /out/model.synap /out/valorant_bestV2_SL1680.model
cd /out
echo 'COMPILACAO_CONCLUIDA'
ls -lah
exec python3 -m http.server "${PORT:-8080}" --bind 0.0.0.0
