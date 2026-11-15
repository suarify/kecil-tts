"""Make the exported ONNX WebGPU-safe: WebGPU has no float64, and the pitch-phase accumulator in the head runs in double.
usage: webgpu_patch.py IN.onnx OUT.onnx   (double items become float32; int64 values outside 32 bits are clamped)"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, numpy_helper

m = onnx.load(sys.argv[1])
n_fix = 0
for n in m.graph.node:
    for a in n.attribute:
        if a.name == 'to' and a.i == T.DOUBLE:
            a.i = T.FLOAT
            n_fix += 1
        if a.t is not None and a.t.data_type == T.DOUBLE:
            arr = numpy_helper.to_array(a.t).astype(np.float32)
            a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
            n_fix += 1
for i in list(m.graph.initializer):
    if i.data_type == T.DOUBLE:
        m.graph.initializer.remove(i)
        m.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float32), i.name))
        n_fix += 1
# WebGPU keeps Slice bounds in 32-bit integers, so an end of INT64_MAX (meaning "to the end") wraps to -1 and the
# slice comes out the wrong length (the same happens to the INT64_MIN used by reversing slices, which dropped one
# sample of the noise source). Clamp every int64 constant outside the 32-bit range to +-(2**31 - 1).
BIG = 2 ** 31 - 1
for n in m.graph.node:
    for a in n.attribute:
        if a.t is not None and a.t.data_type == T.INT64:
            arr = numpy_helper.to_array(a.t)
            if (arr >= 2 ** 31).any() or (arr <= -2 ** 31).any():
                a.t.CopyFrom(numpy_helper.from_array(np.clip(arr, -BIG, BIG).astype(np.int64), a.t.name))
                n_fix += 1
for i in list(m.graph.initializer):
    if i.data_type == T.INT64:
        arr = numpy_helper.to_array(i)
        if (arr >= 2 ** 31).any() or (arr <= -2 ** 31).any():
            m.graph.initializer.remove(i)
            m.graph.initializer.append(numpy_helper.from_array(np.clip(arr, -BIG, BIG).astype(np.int64), i.name))
            n_fix += 1
# ORT-web's WebGPU Slice drops one element of a reversing slice (step -1) that runs to the start. The one that reverses
# the head's random noise is pointless (reversed noise is noise), so it becomes an Identity. The other reversing
# slices are small reflect-padding slices and are left alone.
cons = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type == 'Constant'}
prod = {o: n for n in m.graph.node for o in n.output}
for n in m.graph.node:
    if n.op_type == 'Slice' and len(n.input) >= 5 and n.input[4] in cons and (cons[n.input[4]] < 0).any():
        src = prod.get(n.input[0])
        if src is not None and src.op_type == 'RandomNormalLike':
            first = n.input[0]
            del n.input[:]
            n.input.append(first)
            del n.attribute[:]
            n.op_type = 'Identity'
            n_fix += 1
# value_info entries still say double: drop them so shape inference recomputes
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
print('switched', n_fix, 'double items to float32 ->', sys.argv[2])
