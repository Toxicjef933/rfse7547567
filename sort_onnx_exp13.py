import sys
from pathlib import Path
from collections import defaultdict
import heapq
import onnx

source=Path("/work/exp13_preloaded.onnx")
dest=Path("/work/exp13_sorted.onnx")
m=onnx.load(str(source))
g=m.graph
nodes=list(g.node)
producer={}
for idx,node in enumerate(nodes):
    for output in node.output:
        if output:
            if output in producer:
                raise RuntimeError(f"Duplicate producer {output}: {producer[output]} and {idx}")
            producer[output]=idx

known={x.name for x in g.input}
known.update(x.name for x in g.initializer)
known.update(x.values.name for x in g.sparse_initializer)
dependencies=[set() for _ in nodes]
missing=set()
for idx,node in enumerate(nodes):
    for inp in node.input:
        if not inp: continue
        if inp in producer:
            dependencies[idx].add(producer[inp])
        elif inp not in known:
            missing.add(inp)
if missing:
    print("UNDEFINED_INPUTS",sorted(missing)[:40], flush=True)
    raise RuntimeError(f"{len(missing)} referenced values have no producer")
indegree=[len(x) for x in dependencies]
successors=defaultdict(list)
for idx,deps in enumerate(dependencies):
    for p in deps: successors[p].append(idx)
ready=[i for i,d in enumerate(indegree) if not d]
heapq.heapify(ready)
order=[]
while ready:
    i=heapq.heappop(ready)
    order.append(i)
    for j in successors[i]:
        indegree[j]-=1
        if indegree[j]==0: heapq.heappush(ready,j)
if len(order)!=len(nodes):
    raise RuntimeError(f"Graph cycle or unresolved dependencies: sorted {len(order)}/{len(nodes)}")
print("ORIGINAL_NODES",len(nodes),"REORDERED_NODES",sum(i!=original_index for original_index,i in enumerate(order)),flush=True)
del g.node[:]
g.node.extend(nodes[i] for i in order)
onnx.checker.check_model(m)
onnx.save(m,str(dest))
print("ONNX_TOPOLOGICAL_SORT_OK",dest,dest.stat().st_size,flush=True)
print("INPUTS",[(x.name,[d.dim_value if d.HasField("dim_value") else d.dim_param for d in x.type.tensor_type.shape.dim]) for x in g.input],flush=True)
print("OUTPUTS",[(x.name,[d.dim_value if d.HasField("dim_value") else d.dim_param for d in x.type.tensor_type.shape.dim]) for x in g.output],flush=True)
