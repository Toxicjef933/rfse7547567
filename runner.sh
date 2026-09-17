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
import onnx
from onnx import TensorProto, AttributeProto, numpy_helper

p='/work/model.onnx'
m=onnx.load(p)
print('IR', m.ir_version, 'opsets', [(x.domain, x.version) for x in m.opset_import])
print('metadata_props', [(x.key, x.value[:220]) for x in m.metadata_props])
print('graph inputs', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.input])
print('graph outputs', [(x.name, [d.dim_value or d.dim_param for d in x.type.tensor_type.shape.dim]) for x in m.graph.output])

print('--- STRING INITIALIZERS ---')
count=0
for t in m.graph.initializer:
    if t.data_type == TensorProto.STRING:
        count += 1
        vals=[v.decode('utf-8','replace') if isinstance(v,(bytes,bytearray)) else str(v) for v in t.string_data]
        print('INIT', t.name, 'dims=', list(t.dims), 'vals=', vals[:5])
print('string initializer count=', count)

print('--- CONSTANT NODES WITH STRING TENSORS / STRINGS ---')
count=0
for n in m.graph.node:
    for a in n.attribute:
        if a.type == AttributeProto.TENSOR and a.t.data_type == TensorProto.STRING:
            count += 1
            vals=[v.decode('utf-8','replace') if isinstance(v,(bytes,bytearray)) else str(v) for v in a.t.string_data]
            print('NODE_TENSOR_STRING', n.name, n.op_type, 'attr=', a.name, 'dims=', list(a.t.dims), 'vals=', vals[:5])
        elif a.type == AttributeProto.STRING:
            s=a.s.decode('utf-8','replace')
            if len(s) >= 100:
                print('LONG_STRING_ATTR', n.name, n.op_type, 'attr=', a.name, 'len=', len(s), 'repr=', repr(s[:240]))
        elif a.type == AttributeProto.STRINGS:
            vals=[x.decode('utf-8','replace') for x in a.strings]
            print('STRINGS_ATTR', n.name, n.op_type, 'attr=', a.name, 'lens=', [len(x) for x in vals], 'vals=', vals[:5])
print('string tensor constant count=', count)

print('--- ALL CONSTANT VALUE DTYPES ---')
from collections import Counter
c=Counter()
for n in m.graph.node:
    if n.op_type == 'Constant':
        for a in n.attribute:
            if a.type == AttributeProto.TENSOR:
                c[TensorProto.DataType.Name(a.t.data_type)] += 1
print(dict(c))

print('--- SUSPICIOUS 182-LENGTH TEXT ---')
# Search serialized protobuf bytes for readable runs around length ~182.
b=m.SerializeToString()
import re
for mat in re.finditer(rb'[\x20-\x7e]{170,195}', b):
    s=mat.group().decode('ascii','replace')
    print('ASCII_RUN', len(s), repr(s[:240]))
PY
