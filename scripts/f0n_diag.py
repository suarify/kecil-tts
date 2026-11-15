"""A1 follow-up: why does the text side sound damp? Compare the text side's PREDICTED F0/N with the
MEASURED clip F0/N on the same held-out rows (median, 10-90th percentile span, per-sentence dynamics).

Decoder-only listening showed predicted durations are fine and measured F0/N sound right, so the damp
must live in the predicted F0/N contours. This quantifies what differs: overall level, span, and how
flat the contour is.

    PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe scripts/f0n_diag.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import SR  # noqa: E402

N = 30
SHARD = os.path.join(ROOT, 'runs/k12_rows/rows/shard_0000.pt')
TEXT = os.path.join(ROOT, 'runs/k12/ckpt/textA_last.pt')
torch.set_num_threads(4)

ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(TEXT, map_location='cpu')['model'])
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')

rows = torch.load(SHARD, map_location='cpu', weights_only=False)[:N]

import torch.nn as nn  # noqa: E402


def text_side(ids):
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


def contour_stats(x, voiced=None):
    x = x[voiced] if voiced is not None else x
    return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                span=float(np.percentile(x, 90) - np.percentile(x, 10)),
                flat=float(np.mean(np.abs(np.diff(x)) < 0.01 * (np.percentile(x, 90) - np.percentile(x, 10) + 1e-9))))


pred_f0, meas_f0, pred_n, meas_n = [], [], [], []
with torch.no_grad():
    for r in rows:
        ids = torch.tensor([g2p.tokens(g2p(normalize(r['text'])))], dtype=torch.long)
        pf, pn = text_side(ids)
        # measured rows are at 2x the decoder frame rate; predicted F0/N are at the same rate as the render
        mf, mn = r['F0'].numpy(), r['N'].numpy()
        # voice flag from the measured row: F0 > 0 marks voiced frames
        v = mf[:len(pf)] > 0 if len(mf) >= len(pf) else mf > 0
        pred_f0.append(contour_stats(pf[:len(v)], v))
        meas_f0.append(contour_stats(mf[:len(v)], v))
        pred_n.append(contour_stats(pn[:len(v)]))
        meas_n.append(contour_stats(mn[:len(v)]))

for name, group in (('F0 predicted', pred_f0), ('F0 measured', meas_f0), ('N predicted', pred_n), ('N measured', meas_n)):
    agg = {k: round(float(np.median([g[k] for g in group])), 3) for k in group[0]}
    print(name, agg)
print('F0 predicted span, median over rows:', round(float(np.median([g['span'] for g in pred_f0])), 1), 'Hz-ish units')
print('F0 measured  span, median over rows:', round(float(np.median([g['span'] for g in meas_f0])), 1))
print('N predicted  span, median over rows:', round(float(np.median([g['span'] for g in pred_n])), 3))
print('N measured   span, median over rows:', round(float(np.median([g['span'] for g in meas_n])), 3))
