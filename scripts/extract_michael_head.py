"""Extract the ONNX subgraph feeding the inverse STFT (ConvTranspose 600) of Michael's model, in
topological order, with tensor shapes, to plan the FIX H splice."""
import onnx
from onnx import shape_inference

m = onnx.load('runs/michael_v02.onnx')
m = shape_inference.infer_shapes(m)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
shp = {vi.name: vi for vi in list(g.value_info) + list(g.output) + list(g.input)}


def dims(name):
    vi = shp.get(name)
    if vi is None or not vi.type.tensor_type.HasField('shape'):
        return '?'
    return [d.dim_value if d.HasField('dim_value') else d.dim_param for d in vi.type.tensor_type.shape.dim]


def et(name):
    vi = shp.get(name)
    return vi.type.tensor_type.elem_type if vi else '?'


def consts(n):
    out = []
    for i in n.input:
        if i not in prod:
            for init in g.initializer:
                if init.name == i:
                    v = onnx.numpy_helper.to_array(init)
                    out.append(f'{i}={v.flatten()[:6]}{"..." if v.size > 6 else ""}')
    return ' '.join(out)


# collect ancestors of the ConvTranspose(600)
need = set()
stack = []
for n in g.node:
    if n.op_type == 'ConvTranspose' and [a for a in n.attribute if a.name == 'kernel_shape'][0].ints[0] == 600:
        stack = list(n.input)
seen = set()
while stack:
    t = stack.pop()
    if t in seen or t not in prod:
        continue
    seen.add(t)
    stack.extend(prod[t].input)

nodes = [n for n in g.node if any(o in seen for o in n.output)]
print(f'{len(nodes)} nodes in head subgraph')
for n in nodes:
    ins = []
    for i in n.input:
        ins.append(f'{i}' if i not in prod else i)
    c = consts(n)
    attrs = ' '.join(f'{a.name}={a.ints if a.ints else (a.i if a.HasField("i") else a.f)}' for a in n.attribute)
    outs = []
    for o in n.output:
        outs.append(f'{o}[et={et(o)},d={dims(o)}]')
    print(f'{n.op_type:18s} in={ins} out={outs} {attrs} {c}')
