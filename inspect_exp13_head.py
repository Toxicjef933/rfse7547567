from pathlib import Path
import onnx
from collections import Counter,defaultdict
src=Path("/work/exp13_sorted.onnx")
m=onnx.load(str(src))
g=m.graph
shapes={}
def show(v):
    t=v.type.tensor_type
    if not t.HasField("shape"):return None
    return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in t.shape.dim]
for v in list(g.value_info)+list(g.input)+list(g.output):
    shapes[v.name]=show(v)
for t in g.initializer: shapes[t.name]=list(t.dims)
print("META", "nodes",len(g.node),"initializers",len(g.initializer),"graphs",len(g.output), flush=True)
print("COUNTS",sorted(Counter(n.op_type for n in g.node).items()),flush=True)
print("OUTPUTS", [(v.name, show(v)) for v in g.output],flush=True)
cons=defaultdict(list)
for i,n in enumerate(g.node):
    for x in n.input:cons[x].append(i)
print("END_GRAPH")
for i,n in enumerate(g.node):
    if i>=len(g.node)-115:
        print(f"NODE {i:03d} {n.op_type} {n.name} | INPUT {[(x,shapes.get(x)) for x in n.input[:4]]} | OUTPUT {[(x,shapes.get(x),len(cons[x])) for x in n.output]} ",flush=True)
print("CONVS_HEAD")
for i,n in enumerate(g.node):
    if n.op_type=="Conv" and ("/model.22/" in n.name or "cv2" in n.name or "cv3" in n.name):
        print(f"CONV {i:03d} {n.name} {[(x,shapes.get(x),len(cons[x])) for x in n.output]}",flush=True)
print("SHAPE_INFERENCE")
try:
    z=onnx.shape_inference.infer_shapes(m,strict_mode=False)
    sh={v.name:show(v) for v in list(z.graph.value_info)+list(z.graph.input)+list(z.graph.output)}
    for i,n in enumerate(z.graph.node):
        if (i>=len(z.graph.node)-100 and n.op_type in ("Conv","Concat","Reshape","Transpose","Split","Softmax","Sigmoid","Div","Mul","Add")):
            print(f"SHAPE {i:03d} {n.op_type} {n.name} {[ (x,sh.get(x)) for x in n.output]}",flush=True)
except Exception as e:print("SHAPE_FAILED",repr(e))
