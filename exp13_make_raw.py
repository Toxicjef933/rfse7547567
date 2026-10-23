from pathlib import Path
import onnx
from onnx import helper, TensorProto
import copy
from collections import defaultdict

src=Path('/work/exp13_sorted.onnx')
root=Path('/work')
original=onnx.load(str(src))
onnx.checker.check_model(original)

conv_names={
    'reg0':'/model.23/cv2.0/cv2.0.2/Conv_output_0',
    'cls0':'/model.23/cv3.0/cv3.0.2/Conv_output_0',
    'reg1':'/model.23/cv2.1/cv2.1.2/Conv_output_0',
    'cls1':'/model.23/cv3.1/cv3.1.2/Conv_output_0',
    'reg2':'/model.23/cv2.2/cv2.2.2/Conv_output_0',
    'cls2':'/model.23/cv3.2/cv3.2.2/Conv_output_0'
}
shapes={
    'reg0':[1,64,40,40],'cls0':[1,1,40,40],
    'reg1':[1,64,20,20],'cls1':[1,1,20,20],
    'reg2':[1,64,10,10],'cls2':[1,1,10,10]
}
available={a for n in original.graph.node for a in n.output}
for k,v in conv_names.items():
    if v not in available: raise RuntimeError(f'Missing head node {k}={v}')

for mode in (6,9):
    m=copy.deepcopy(original)
    g=m.graph
    del g.output[:]
    outputs=[]
    for scale in range(3):
        for k in ('reg','cls'):
            name=conv_names[k+str(scale)]
            g.output.append(helper.make_tensor_value_info(name,TensorProto.FLOAT,shapes[k+str(scale)]))
            outputs.append(name)
        if mode==9:
            raw=conv_names['cls'+str(scale)]
            reduceout=f'exp13_class_sum_{scale}'
            clipout=f'exp13_class_sum_clip_{scale}'
            g.node.append(helper.make_node('ReduceSum',[raw], [reduceout], name=f'exp13_reduce_score_s{scale}',axes=[1],keepdims=1))
            g.node.append(helper.make_node('Relu',[reduceout],[clipout],name=f'exp13_nonnegative_s{scale}'))
            g.output.append(helper.make_tensor_value_info(clipout,TensorProto.FLOAT,shapes['cls'+str(scale)]))
            outputs.append(clipout)

    required=set(outputs)
    keep=[]
    for node in reversed(g.node):
        if any(x in required for x in node.output if x):
            keep.append(node)
            required.update(x for x in node.input if x)
    keep.reverse()
    del g.node[:]
    g.node.extend(keep)

    used={x for node in g.node for x in node.input}
    init_keep=[v for v in g.initializer if v.name in used]
    del g.initializer[:]
    g.initializer.extend(init_keep)
    del g.value_info[:]

    # IR does not require graph output names to be produced by graph's last node.
    onnx.checker.check_model(m)
    dst=root/f'exp13_raw{mode}.onnx'
    onnx.save(m,str(dst))
    print('RAW_ONNX_READY', mode, str(dst),dst.stat().st_size,'nodes',len(g.node),'weights',len(g.initializer),flush=True)
    print('OUTPUTS',mode,[(o.name,list(x.dim_value for x in o.type.tensor_type.shape.dim)) for o in g.output],flush=True)
    print('NO_DFL',mode,not any('/dfl/' in n.name for n in g.node),flush=True)
    # Run ORT for one synthetic input, verifying model loads and all expected outputs exist.
    import onnxruntime as ort
    import numpy as np
    sess=ort.InferenceSession(str(dst),providers=['CPUExecutionProvider'])
    meta=sess.get_inputs()[0]
    shape=meta.shape
    if meta.type != 'tensor(float)': raise RuntimeError(f'Unexpected input type {meta.type}')
    x=np.random.default_rng(0).random((1,3,320,320), dtype=np.float32)
    ys=sess.run(None,{meta.name:x})
    for i,(o,y) in enumerate(zip(g.output,ys)):
        declared=shapes['reg'+str(i//3)] if mode==9 and i%3==0 else None
        if not np.isfinite(y).all(): raise RuntimeError(f'Nonfinite output {i}')
    print('ORT_OK',mode,[tuple(y.shape) for y in ys],flush=True)
