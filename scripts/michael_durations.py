"""Inspect per-phoneme durations for the problem sentences: are III / atau getting too few frames
(staccato, robotic) or a broken curve? Also prints the pitch curve over the word."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import VDecoder  # noqa: E402
from torch.nn.utils import remove_weight_norm  # noqa: E402
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations  # noqa: E402


def strip(mod):
    for x in mod.modules():
        try:
            remove_weight_norm(x)
        except Exception:
            pass
        if is_parametrized(x, 'weight'):
            try:
                remove_parametrizations(x, 'weight', leave_parametrized=True)
            except Exception:
                pass


ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
ts = TextStudent('s+mlp').eval(); strip(ts)
ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
vd = VDecoder(head='v2', dim=160, blocks=4, K=128).eval(); strip(vd)
vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')

TEXTS = [
    ('iii', 'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.'),
    ('atau2', 'Kita boleh pergi sekarang atau tunggu sehingga petang.'),
]

with torch.no_grad():
    for key, text in TEXTS:
        ph = g2p(normalize(text))
        ids = torch.tensor([g2p.tokens(ph)], dtype=torch.long)
        t, p, s = ts, ts.predictor, ts.style
        d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
        x, T = d_en, ids.shape[1]
        sx = s[:, :, None].expand(-1, -1, T)
        x = torch.cat([x, sx], 1)
        for b in p.text_encoder.lstms:
            if isinstance(b, nn.LSTM):
                x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
            else:
                x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
        d = x.transpose(-1, -2)
        dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
        n = torch.clamp(torch.round(dur), min=1).long()
        toks = g2p.tokens(ph)
        print(f'\n{key}: {text}')
        print(f'  phonemes ({len(toks)}): {ph}')
        print('  durations (frames, 25 ms each):')
        print('   ', [int(v) for v in n.tolist()])
        # P(1-frame) share: staccato indicator
        ones = (n == 1).float().mean().item()
        print(f'  share of 1-frame phonemes: {ones:.0%} | total frames {int(n.sum())} = {int(n.sum()) * 25} ms')
        # locate 'ataw'
        idx = ph.find('ataw')
        if idx >= 0:
            seg = n[idx:idx + 4]
            print(f'  atau spans phoneme idx {idx}..{idx + 3}: durations {seg.tolist()} frames '
                  f'({sum(int(v) for v in seg.tolist()) * 25} ms)')
        # pitch curve over the whole sentence
        end = torch.cumsum(n, 0); start = end - n
        f = torch.arange(int(end[-1]))[None]
        aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
        F0, N_ = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
        f0 = F0[0].numpy()
        voiced = f0 > 10
        print(f'  F0: voiced frames {int(voiced.sum())}/{len(f0)}, '
              f'range {f0[voiced].min():.0f}-{f0[voiced].max():.0f} Hz, '
              f'frame-to-frame median |dF0| {np.median(np.abs(np.diff(f0[voiced]))):.1f} Hz')
