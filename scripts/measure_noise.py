"""Measure where the remaining voiced noise energy lives in v0.4: plain / pulsed / burst paths."""
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402

torch.set_num_threads(4)
ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(os.path.join(ROOT, 'runs/k16_final/textA_c4_last.pt'), map_location='cpu')['model'])
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(os.path.join(ROOT, 'runs/k16_final/dec_c4_last.pt'))
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')


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
    F0, N = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return asr @ aln, F0, N


def trunk(asr, F0c, N):
    s = vd.style[:1]
    F0, Nn = vd.F0_conv(F0c.unsqueeze(1)), vd.N_conv(N.unsqueeze(1))
    x = vd.encode(torch.cat([asr, F0, Nn], 1), s)
    asr_res, res = vd.asr_res(asr), True
    for block in vd.decode:
        if res:
            x = torch.cat([x, asr_res, F0, Nn], 1)
        x = block(x, s)
        if block.upsample_type != 'none':
            res = False
    return x


clips = torch.load(os.path.join(ROOT, 'runs/k15_v3/cb_clips2.pt'), map_location='cpu', weights_only=False)
plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open(os.path.join(ROOT, 'runs/k15_v3/gen_plan.csv'), encoding='utf-8'))}
ids = [i for i in clips if i in plan][:6]
SHARE = []
with torch.no_grad():
    for i in ids:
        ids_t = torch.tensor([g2p.tokens(g2p(normalize(plan[i][0])))], dtype=torch.long)
        asr, F0, Nc = text_side(ids_t)
        x = trunk(asr, F0, Nc)
        h = vd.head
        B, _, n = x.shape
        T = n * h.up
        z = h.norm0(h.inp(x).transpose(1, 2)).transpose(1, 2)
        for b in h.blocks:
            z = b(z)
        hn = h.norm1(z.transpose(1, 2))
        o = h.out(hn).view(B, T, h.C * h.G + 1)
        o = torch.cat([o, o[:, -1:]], 1)
        g = (o[..., :-1].reshape(B, T + 1, h.C, h.G) @ h.M.T).permute(0, 2, 3, 1)
        gh, gn = torch.exp(g[:, 0].clamp(max=6)), torch.exp(g[:, 3].clamp(max=6))
        e, cyc, uv = sine_bank(F0, h.K, None)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        Nz0 = h.stft(nz)
        Nm = [h.stft(nz * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
        mod = sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))
        f0raw = torch.cat([F0.repeat_interleave(h.up, 1), F0[:, -1:]], 1)
        voiced = (f0raw > 10).float()[:, None, :]
        fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
        band = (fr >= 1500).float()
        Nb = torch.stft(nz.flip(-1), 150, 75, window=h.wb, return_complex=True)
        gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
        gb = torch.cat([gb, gb[..., -1:]], -1)
        yb = torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)          # burst path alone
        Sb = torch.stft(yb, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs() ** 2
        # burst STFT has 81 frames of 256 hop; align count to head frames T+1
        Sb = torch.nn.functional.interpolate(Sb[None], size=(h.nb, T + 1), mode='bilinear', align_corners=False)[0]
        P_main = ((Nz0 * gn) ** 2 * band).sum(1)
        P_mod = ((mod * gn) ** 2 * band).sum(1)
        P_burst = (Sb * band).sum(1)
        v = (voiced[:, 0, :] > 0)[0]
        pm = P_main[:, v].sum().double(); pmo = P_mod[:, v].sum().double(); pb = P_burst[:, v].sum().double()
        print('  raw:', float(pm), float(pmo), float(pb), flush=True)
        tot = pm + pmo + pb
        SHARE.append((float(pm / tot), float(pmo / tot), float(pb / tot)))
        print(i, 'voiced >1.5k noise shares: plain %.0f%%  pulsed %.0f%%  burst %.0f%%' % (SHARE[-1][0] * 100, SHARE[-1][1] * 100, SHARE[-1][2] * 100), flush=True)
a = np.mean(SHARE, 0)
print('\nmean voiced >1.5 kHz noise energy: plain %.0f%% | pulsed(mod) %.0f%% | burst(3ms) %.0f%%' % (a[0] * 100, a[1] * 100, a[2] * 100))
