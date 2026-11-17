"""Trace Michael's compiled ONNX head: find the tensors feeding the inverse STFT convs (n=600) and the
burst path (n=150), and the F0/voiced tensors, so FIX H ops can be spliced in."""
import onnx

m = onnx.load('runs/michael_v02.onnx')
g = m.graph

prod = {}
for n in g.node:
    for o in n.output:
        prod[o] = n

def shape_of(name):
    for vi in list(g.value_info) + list(g.output) + list(g.input):
        if vi.name == name:
            t = vi.type.tensor_type
            dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in t.shape.dim]
            return t.elem_type, dims
    return None, None


def chain(name, depth=6, seen=None):
    """Print the producer chain of a tensor."""
    if seen is None:
        seen = set()
    if name in seen or depth < 0:
        return
    seen.add(name)
    n = prod.get(name)
    if n is None:
        return
    et, dims = shape_of(name)
    print(f"  <- {n.op_type:16s} {name[:70]:70s} et={et} dims={dims}")
    for i in n.input:
        chain(i, depth - 1, seen)


# the two inverse-STFT ConvTranspose(600) nodes
for n in g.node:
    if n.op_type == 'ConvTranspose':
        ks = [a for a in n.attribute if a.name == 'kernel_shape'][0].ints[0]
        if ks in (600, 150):
            print(f'\n== {n.op_type} kernel {ks}: {n.name}')
            for i in n.input:
                chain(i)
