"""Step 0: how big/slow is a wider head? Instantiate candidates, count parameters and time a forward pass."""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
os.chdir(ROOT)

from student import TextStudent  # noqa: E402
from vhead import VDecoder  # noqa: E402

ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load('runs/kaggle_sm07/textA_delta_last.pt', map_location='cpu', weights_only=False)['model'])
n_ts = sum(p.numel() for p in ts.parameters())

print(f'{"variant":16s} {"trunk+head":>12s} {"total":>10s} {"est int8 MB":>12s} {"time per s audio":>17s}')
for label, dim, blocks in (('current', 160, 4), ('wide-224', 224, 4), ('wide-320', 320, 4), ('wide-320-6b', 320, 6)):
    vd = VDecoder(head='v2', dim=dim, blocks=blocks, K=128).eval()
    n_vd = sum(p.numel() for p in vd.parameters())
    total = n_vd + n_ts
    est = total * 1.05 / 1e6      # int8 weights (+ scales/vectors), rough
    x = torch.randn(1, 512, 400)
    f0 = torch.rand(1, 800) * 150 + 80
    nz = torch.randn(1, 800)
    with torch.no_grad():
        t0 = time.perf_counter()
        y = vd(x, f0, nz)
        dt = time.perf_counter() - t0
    secs = y.shape[-1] / 24000
    print(f'{label:16s} {n_vd / 1e6:11.2f}M {total / 1e6:9.2f}M {est:11.1f} {secs / dt:14.1f}x')
