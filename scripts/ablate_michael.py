"""Michael ablation: his v0.2.1 (FIX H) still sounds watery/phasey to the owner, so test where it comes from.

Variants (all same noise seed, loudness-matched to the v0.2.1 output):
  old      no FIX H at all (v0.2 behaviour: noise smeared over harmonics, burst everywhere)
  fixH     current v0.2.1 (gap-masked noise + burst gated to unvoiced)
  noPhase  fixH with the learned phase removed (u = 1, pulse shift kept)
  noShift  fixH with the pulse shift removed (learned phase u kept)
  keepHalf fixH + half the voiced noise energy moved into the harmonics (energy-compensated)
  noBurst  fixH with the 3 ms burst path removed entirely
  noMod    fixH with the pulsed-noise sources removed (plain noise only)

Writes runs/listen/michael_ablate/*.wav + items.json.
"""
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
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
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402

OUT, N = 'runs/listen/michael_ablate', 6
torch.set_num_threads(4)

ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
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


ts = TextStudent('s+mlp').eval(); strip(ts)
ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
vd = VDecoder(head='v2', dim=160, blocks=4, K=128).eval(); strip(vd)
vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])
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


def head_parts(h, x, f0, noise, interharm=False, keep=1.0, no_mod=False, burst_uv=False, comp_burst=False,
               phase=True, shift_on=True, burst_off=False, harm_only=False):
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
    u = torch.complex(g[:, 1], g[:, 2])
    u = u / u.abs().clamp_min(1e-4)
    if not phase:
        u = torch.ones_like(u)
    e, cyc, uv = sine_bank(f0, h.K, None)
    E = h.stft(e)
    Nz0 = h.stft(noise)
    Nm = [] if no_mod else [h.stft(noise * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([f0.repeat_interleave(h.up, 1), f0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None] if shift_on else torch.zeros_like(o[..., -1])[:, None]
    Nz = Nz0 + sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))
    voiced = (f0raw > 10).float()
    band_mask = (fr[0, :, 0] >= 800).float()[None, :, None]
    if interharm:
        frac = (ratio - ratio.round()).abs()
        hp = torch.exp(-((frac / 0.22) ** 2)) * band_mask * voiced[:, None, :]
        w = 1 - hp.clamp(max=1)
        Nz2 = Nz * w
        P0 = ((Nz.abs() ** 2) * band_mask).sum(1, keepdim=True)
        P1 = ((Nz2.abs() ** 2) * band_mask).sum(1, keepdim=True)
        s = torch.sqrt(P0 / P1.clamp_min(1e-12)).clamp(max=4.0)
        s = nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)
        Nz = Nz2 * s
    if h.burst and not burst_off:
        gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
        gb = torch.cat([gb, gb[..., -1:]], -1)
        Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
        Yb = Nb * gb
        band_b = (torch.linspace(0, SR / 2, 76)[None, :, None] >= 1500).float()
        Pb = (Yb.abs() ** 2 * band_b).sum(1)
        Pb = nn.functional.interpolate(Pb[:, None, :], size=T + 1, mode='linear', align_corners=False)[:, 0]
        Pb = (Pb * voiced).reshape(B, T + 1)
        if burst_uv:
            vb = torch.cat([f0.repeat_interleave(4, 1), f0[:, -1:]], 1) > 10
            gb = gb * (~vb).float()[:, None, :]
    if keep != 1:
        band = (fr >= 1500).float()
        Ph = ((gh * E.abs()) ** 2 * band).sum(1, keepdim=True)
        Pn = ((Nz.abs() * gn) ** 2 * band).sum(1, keepdim=True)
        if h.burst and comp_burst:
            Pn = Pn + Pb[:, None, :]
        sc = torch.sqrt((Ph + (1 - keep ** 2) * Pn) / Ph.clamp_min(1e-9)).clamp(max=4.0)
        sc = nn.functional.avg_pool1d(sc, 5, 1, 2, count_include_pad=False)
        m = band * voiced[:, None, :]
        gh = gh * (1 + m * (sc - 1))
        gn = gn * (1 + m * (keep - 1))
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    yn = torch.istft(Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    if h.burst and not burst_off:
        yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    if harm_only:
        return yh, torch.zeros_like(yn)
    return yh, yn


def rms(a):
    return float(np.sqrt((np.asarray(a, np.float32) ** 2).mean()))


def periodicity(a):
    a = np.asarray(a, np.float32)
    a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480
    vals = []
    for i in range(0, len(a) - win, hop):
        fr_ = a[i:i + win]
        if np.sqrt((fr_ ** 2).mean()) < 0.02:
            continue
        fr_ = fr_ - fr_.mean()
        ac = np.correlate(fr_, fr_, 'full')[win - 1:]
        ac /= ac[0] + 1e-12
        vals.append(float(ac[int(SR / 200):int(SR / 80)].max()))
    return float(np.median(vals)) if vals else 0.0


def texture(a):
    import librosa
    S = np.abs(librosa.stft(np.asarray(a, np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    over4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    return flat, over4


plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = ['c0313', 'c0551', 'c0571', 'c0726', 'c0746', 'c0747'][:N]

os.makedirs(OUT, exist_ok=True)
VARIANTS = ['old', 'fixH', 'harmOnly', 'noPhase', 'noShift', 'keepHalf', 'noBurst', 'noMod']
items, met = [], {v: [] for v in VARIANTS}
with torch.no_grad():
    for i in ids:
        text, cat = plan[i]
        ids_t = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
        asr, F0, Nc = text_side(ids_t)
        x = trunk(asr, F0, Nc)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        it = dict(id=i, cat=cat, text=text, files={})
        yref = None
        for v in VARIANTS:
            if v == 'old':
                yh, yn = head_parts(vd.head, x, F0, nz)
            elif v == 'fixH':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True)
            elif v == 'harmOnly':
                yh, yn = head_parts(vd.head, x, F0, nz, harm_only=True)
            elif v == 'noPhase':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True, phase=False)
            elif v == 'noShift':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True, shift_on=False)
            elif v == 'keepHalf':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True, comp_burst=True, keep=0.5)
            elif v == 'noBurst':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_off=True)
            elif v == 'noMod':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True, no_mod=True)
            y = (yh + yn).squeeze().numpy()
            if v == 'fixH':
                yref = y.copy()
            if yref is not None:
                y = y * (rms(yref) / (rms(y) + 1e-9))
            fn = f'{i}_{v}.wav'
            sf.write(os.path.join(OUT, fn), np.clip(y, -1, 1), SR)
            it['files'][v] = f'{OUT}/{fn}'
            met[v].append([periodicity(y), *texture(y)])
        items.append(it)
        print(i, cat, 'done', flush=True)

res = {v: [round(float(x), 3) for x in np.array(m).mean(0)] for v, m in met.items()}
json.dump(dict(items=items, metrics=res), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n            periodicity  flatness2-6k  >4k share')
for v in VARIANTS:
    print(f'  {v:10s} {res[v][0]:.3f}       {res[v][1]:.3f}        {res[v][2]:.3f}')
