import onnx

m = onnx.load('runs/k16_final/fixh_export/paradee.onnx')
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cands = sorted({o for n in g.node if n.op_type == 'MatMul' for o in n.output if 'head' in n.name})
print('head MatMul outputs:', cands[:12])
# for each, show the producer's inputs
for c in cands[:12]:
    n = prod[c]
    print(' ', c, '<-', n.name, n.input)
for n2 in g.node:
    if outn in n2.input:
        print('consumer:', n2.op_type, n2.name, n2.input)
