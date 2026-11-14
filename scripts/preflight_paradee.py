"""CPU preflight for the Paradee route (no GPU, no Kaggle quota).

Checks that the released Paradee-8M v1.3 PyTorch weights load into the public training code
(commit 2de547e) after slicing the eight-voice style table down to one row, and that a forward
pass of both halves runs. Run with the project venv:

    .venv/Scripts/python scripts/preflight_paradee.py <path to paradee/training/scripts>
"""
import os
import sys
import types

import torch

SCRIPTS = sys.argv[1]
ROW = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# kokoro imports misaki at import time; the model code does not need it.
stubs = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.stubs', 'misaki')
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    with open(os.path.join(stubs, name + '.py'), 'w') as f:
        f.write('def __getattr__(name):\n    return type(name, (), {})\n')
# The package must import its submodules itself: a package-level __getattr__ would hand back a
# dummy class for `from misaki import en`, and kokoro then fails on `en.MToken`.
with open(os.path.join(stubs, '__init__.py'), 'w') as f:
    f.write('from . import en, espeak\n')
sys.path.insert(0, os.path.dirname(stubs))
sys.path.insert(0, SCRIPTS)

from huggingface_hub import hf_hub_download  # noqa: E402

from student import TextStudent  # noqa: E402
from vhead import VDecoder  # noqa: E402

REPO = 'sahilmahendrakar/Paradee-8M-v1.3'
out = {}
for half, fname in (('text', 'pytorch/text_side.pt'), ('dec', 'pytorch/decoder.pt')):
    ck = torch.load(hf_hub_download(REPO, fname), map_location='cpu')
    st = ck['model']
    print(half, 'checkpoint keys:', list(ck.keys()), '| style table', tuple(st['style'].shape), '| tensors', len(st))
    st = dict(st)
    st['style'] = st['style'][ROW:ROW + 1].clone()
    out[half] = st

ts = TextStudent('s+mlp')
res = ts.load_state_dict(out['text'], strict=True)
print('text side loaded:', res, '| params', round(sum(p.numel() for p in ts.parameters()) / 1e6, 2), 'M')

vd = VDecoder(head='v2', dim=160, blocks=4)
res = vd.load_state_dict(out['dec'], strict=True)
print('decoder loaded:', res, '| params', round(sum(p.numel() for p in vd.parameters()) / 1e6, 2), 'M')

# forward pass: 20 tokens, teacher-free
ts.eval(); vd.eval()
ids = torch.tensor([[0] + [43, 55, 51, 56, 16, 53, 43, 62, 47, 54, 16, 61, 43, 52, 43, 4] + [0]])
L = torch.tensor([ids.shape[1]])
with torch.no_grad():
    dur, _, _, asr_tok = ts(ids, L)
    pred = torch.round(dur[0]).clamp(min=1).long()
    from student import alignment
    aln = alignment(pred)[None]
    _, F0, N, _ = ts(ids, L, aln)
    torch.manual_seed(0)
    y = vd(asr_tok @ aln, F0, N)
print('forward ok: frames', int(pred.sum()), '| audio samples', tuple(y.shape), '=', round(y.shape[-1] / 24000, 2), 's',
      '| finite', bool(torch.isfinite(y).all()), '| peak', round(float(y.abs().max()), 3))
torch.save({'model': out['text']}, os.path.join(os.path.dirname(stubs), 'text_row.pt'))
print('PREFLIGHT PASS')
