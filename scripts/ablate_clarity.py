"""Clarity pass on the FAITHFUL student (what the owner preferred): no clamp, no burst gate, no smoothing.
Tests how much murk can be removed by air shelf and by moving noise out of the gaps."""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402

OUT = 'runs/listen/clarity'
torch.set_num_threads(4)
TEXTS = [
    ('iii', 'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.'),
    ('call', 'Booking encik untuk hari Selasa dah confirmed.'),
]


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


ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load('runs/kaggle_sm07/textA_delta_last.pt', map_location='cpu', weights_only=False)['model'])
vd = VDecoder(head='v2', K=128).eval(); vd.load_ckpt('runs/kaggle_sm07/dec_delta_last.pt'); strip(ts); strip(vd)
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
    end = torch.cumsum(n, 0); start = end - n
    f = torch.arange(int(end[-1]))[None]
    aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
    F0, N_ = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return asr @ aln, F0, N_


def trunk(v_, asr, F0c, N):
    s = v_.style[:1]
    F0, Nn = v_.F0_conv(F0c.unsqueeze(1)), v_.N_conv(N.unsqueeze(1))
    x = v_.encode(torch.cat([asr, F0, Nn], 1), s)
    asr_res, res = v_.asr_res(asr), True
    for block in v_.decode:
        if res:
            x = torch.cat([x, asr_res, F0, Nn], 1)
        x = block(x, s)
        if block.upsample_type != 'none':
            res = False
    return x


def head_render(h, x, f0, noise, gap_keep=1.0, harmonic_boost_db=0.0):
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
    u = torch.complex(g[:, 1], g[:, 2]); u = u / u.abs().clamp_min(1e-4)
    e, cyc, uv = sine_bank(f0, h.K, None)
    E = h.stft(e); Nz0 = h.stft(noise)
    Nm = [h.stft(noise * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([f0.repeat_interleave(h.up, 1), f0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None]
    Nz = Nz0 + sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))
    voiced = (f0raw > 10).float()
    Nz = Nz * gap_keep                     # simply scale the noise part (no gap shaping)
    if harmonic_boost_db:
        band15 = (fr >= 1500).float()
        gh = gh * (1 + band15 * voiced[:, None, :] * (10 ** (harmonic_boost_db / 20) - 1))
    gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
    gb = torch.cat([gb, gb[..., -1:]], -1)
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    yn = torch.istft(Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
    yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    return (yh + yn).squeeze().numpy()


def air(y, db=3.0, fc=6500.0):
    b, a = butter(4, fc / (SR / 2), 'high')
    return y + (10 ** (db / 20) - 1) * filtfilt(b, a, y)


def rms(a):
    return float(np.sqrt((np.asarray(a, np.float32) ** 2).mean()))


def periodicity(a):
    a = np.asarray(a, np.float32); a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480; vals = []
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
    P = S ** 2; tot = P.sum(0); keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    o4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    o7 = float((P[f > 7000][:, keep].sum(0) / tot[keep]).mean())
    return flat, o4, o7


VARIANTS = {
    'faithful': dict(),
    'air3': dict(),
    'air5': dict(),
    'noise60': dict(gap_keep=0.6),
    'air3noise60': dict(gap_keep=0.6),
    'air5noise60': dict(gap_keep=0.6),
    'harmBoost': dict(harmonic_boost_db=3.0),
}
os.makedirs(OUT, exist_ok=True)
items, metrics = [], {k: [] for k in VARIANTS}
with torch.no_grad():
    for key, text in TEXTS:
        ids = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
        asr, F0, Nc = text_side(ids)
        x = trunk(vd, asr, F0, Nc)
        it = dict(id=key, cat=key, text=text, files={})
        yf = None
        for v, kw in VARIANTS.items():
            torch.manual_seed(0)
            nz = torch.randn(1, x.shape[2] * FHOP)
            y = head_render(vd.head, x, F0, nz, **kw)
            if v in ('air3', 'air3noise60'):
                y = air(y, 3.0)
            elif v in ('air5', 'air5noise60'):
                y = air(y, 5.0)
            if v == 'faithful':
                yf = y.copy()
            else:
                y = y * (rms(yf) / (rms(y) + 1e-9))
            fn = f'{key}_{v}.wav'
            sf.write(os.path.join(OUT, fn), np.clip(y, -1, 1), SR)
            it['files'][v] = f'{OUT}/{fn}'
            metrics[v].append([periodicity(y), *texture(y)])
        items.append(it)
        print(key, 'done', flush=True)

res = {v: [round(float(t), 3) for t in np.mean(m, 0)] for v, m in metrics.items()}
json.dump(dict(items=items, metrics=res), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n                periodicity  flat2-6k  >4k    >7k')
for v in VARIANTS:
    print(f'  {v:12s} {res[v][0]:.3f}       {res[v][1]:.3f}     {res[v][2]:.3f}  {res[v][3]:.3f}')
