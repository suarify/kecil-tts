"""Download Paradee-8M v1.3 PyTorch weights and slice the eight-voice style tables down to one voice.
usage: ms_init.py OUTDIR [ROW]      (row 0 = af_heart)"""
import os
import sys

import torch
from huggingface_hub import hf_hub_download

out = sys.argv[1]
row = int(sys.argv[2]) if len(sys.argv) > 2 else 0
os.makedirs(out, exist_ok=True)
for half, fname, dst in (('text', 'pytorch/text_side.pt', 'text_row0.pt'), ('dec', 'pytorch/decoder.pt', 'dec_row0.pt')):
    st = dict(torch.load(hf_hub_download('sahilmahendrakar/Paradee-8M-v1.3', fname), map_location='cpu')['model'])
    st['style'] = st['style'][row:row + 1].clone()
    torch.save({'model': st}, os.path.join(out, dst))
    print(half, 'style', tuple(st['style'].shape), '->', os.path.join(out, dst))
