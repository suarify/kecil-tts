"""A1 follow-up part 2: quantify how the PREDICTED F0 contour differs from the MEASURED one.

    PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe scripts/f0n_diag2.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402

torch.set_num_threads(4)
ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(os.path.join(ROOT, 'runs/k12/ckpt/textA_last.pt'), map_location='cpu')['model'])
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
rows = torch.load(os.path.join(ROOT, 'runs/k12_rows/rows/shard_0000.pt'), map_location='cpu', weights_only=False)[:30]


def text_side(ids):
    t, p, s = ts, ts.predictor, ts.style
    d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
    x, T = d_en, ids.shape[1]
    sx = s[:, :, None].expand(-1, -1, T)
    x = torch.cat([x, sx], 1)
    for b in p.text_encoder.lstms:
        x = (b(x.transpose(-1, -2))[0].transpose(-1, -2) if isinstance(b, nn.LSTM)
             else torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1))
    d = x.transpose(-1, -2)
    dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
    e = t.text_encoder
    h = e.embedding(ids).transpose(1, 2)
    for c in e.cnn:
        h = c(h)
    asr = t.asr_proj(e.lstm(h.transpose(1, 2))[0]).transpose(-1, -2)
    n = torch.clamp(torch.round(dur), min=1).long()
    end = torch.cumsum(n, 0)
    start = end - n
    f = torch.arange(int(end[-1]))[None]
    aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
    F0, Nn = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return F0[0].numpy(), Nn[0].numpy()


drop, jag, pspan, mspan = [], [], [], []
with torch.no_grad():
    for r in rows:
        ids = torch.tensor([g2p.tokens(g2p(normalize(r['text'])))], dtype=torch.long)
        pf, _ = text_side(ids)
        mf = r['F0'].numpy()
        L = min(len(pf), len(mf))
        v = mf[:L] > 0
        pv = pf[:L][v]
        drop.append(float((pv < 50).mean()))
        jag.append(float((np.abs(np.diff(pv)) > 25).mean()))
        pspan.append(float(np.percentile(pv, 90) - np.percentile(pv, 10)))
        m = mf[mf > 0]
        mspan.append(float(np.percentile(m, 90) - np.percentile(m, 10)))

print('share of voiced frames with predicted F0 < 50 Hz (dropouts):', round(float(np.median(drop)) * 100, 1), '%')
print('share of voiced-to-voiced F0 jumps > 25 Hz (jaggedness):     ', round(float(np.median(jag)) * 100, 1), '%')
print('predicted voiced-F0 span 10-90 pct:', round(float(np.median(pspan)), 1), 'Hz')
print('measured  voiced-F0 span 10-90 pct:', round(float(np.median(mspan)), 1), 'Hz')
