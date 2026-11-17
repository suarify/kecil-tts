"""Michael: fix the timing, not the spectrum. The duration model gives some phonemes 1 frame (25 ms, rushed)
and others 16 frames (400 ms, a frozen held vowel = robotic/damp/watery), and a fast vibrato lands between the
25 ms frames that carry the phase, which is why it smears.

Variants:
  base      published hang7 + noise-gain smoothing
  clamp      per-phoneme durations clamped to [2, 8] frames (50-200 ms)
  med3      durations median-filtered (3) then clamped to [2, 10]
  clampair  clamp + 3 dB air shelf above 6.5 kHz
  trem      clamp + slow amplitude tremor on the gap noise in voiced frames (micro life, no pitch change)
  vib03     clamp + gentle 0.3% vibrato at 3.0 Hz (slow enough for the frame grid)
"""
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

OUT = 'runs/listen/dur'
torch.set_num_threads(4)
TEXTS = [
    ('iii', 'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.'),
    ('atau2', 'Kita boleh pergi sekarang atau tunggu sehingga petang.'),
    ('mixed', 'Buku ini untuk murid tahun lima di sekolah kebangsaan.'),
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


ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
ts = TextStudent('s+mlp').eval(); strip(ts)
ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
vd = VDecoder(head='v2', dim=160, blocks=4, K=128).eval(); strip(vd)
vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')


def text_side(t_, ids, dur_mode='base', vlo=2, vhi=8):
    t, p, s = t_, t_.predictor, t_.style
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
    if dur_mode in ('clamp', 'clampair', 'trem', 'vib03'):
        n = n.clamp(vlo, vhi)
    elif dur_mode == 'med3':
        pad = torch.cat([n[:1], n, n[-1:]])
        med = torch.stack([pad[0:-2], pad[1:-1], pad[2:]]).median(0).values
        n = med.clamp(vlo, 10).long()
    end = torch.cumsum(n, 0); start = end - n
    f = torch.arange(int(end[-1]))[None]
    aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
    F0, N_ = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return asr @ aln, F0, N_, n


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


def smooth_time(t, kernel):
    B, F, T = t.shape
    k = torch.tensor(kernel, dtype=t.dtype).expand(F, 1, len(kernel))
    pad = len(kernel) // 2
    return nn.functional.conv1d(nn.functional.pad(t, (pad, pad), mode='replicate'), k, groups=F)


def head_render(h, x, f0, noise, hang=7, trem=0.0, vib=0.0, vib_hz=3.0):
    if vib:
        tt = torch.arange(f0.shape[1], dtype=f0.dtype)[None] * (FHOP / SR)
        f0 = f0 * (1 + vib * torch.sin(2 * torch.pi * vib_hz * tt))
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
    gn = smooth_time(gn, [1 / 3, 1 / 3, 1 / 3])
    if trem:                                             # slow random amplitude sway (frame-rate, per frame)
        torch.manual_seed(7)
        sway = 1 + trem * (torch.randn(1, 1, gn.shape[2]) * 0.5).clamp(-1, 1)
        gn = gn * sway
        gh = gh * (1 + 0.5 * (sway - 1))
    u = torch.complex(g[:, 1], g[:, 2])
    u = u / u.abs().clamp_min(1e-4)
    e, cyc, uv = sine_bank(f0, h.K, None)
    E = h.stft(e)
    Nz0 = h.stft(noise)
    Nm = [h.stft(noise * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([f0.repeat_interleave(h.up, 1), f0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None]
    Nz = Nz0 + sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))
    voiced = (f0raw > 10).float()
    band_mask = (fr[0, :, 0] >= 800).float()[None, :, None]
    frac = (ratio - ratio.round()).abs()
    hp = torch.exp(-((frac / 0.22) ** 2)) * band_mask * voiced[:, None, :]
    w = 1 - hp.clamp(max=1)
    Nz2 = Nz * w
    P0 = ((Nz.abs() ** 2) * band_mask).sum(1, keepdim=True)
    P1 = ((Nz2.abs() ** 2) * band_mask).sum(1, keepdim=True)
    s = torch.sqrt(P0 / P1.clamp_min(1e-12)).clamp(max=4.0)
    s = nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)
    Nz = Nz2 * s
    gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
    gb = torch.cat([gb, gb[..., -1:]], -1)
    vbb = torch.cat([f0.repeat_interleave(4, 1), f0[:, -1:]], 1) > 10
    vb = nn.functional.max_pool1d(vbb.float()[None], hang, 1, hang // 2)[0] > 0
    gb = gb * (~vb).float()[:, None, :]
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    yn = torch.istft(Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
    yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    return (yh + yn).squeeze().numpy()


def air(y, db, fc=6500.0):
    b, a = butter(4, fc / (SR / 2), 'high')
    return y + (10 ** (db / 20) - 1) * filtfilt(b, a, y)


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


VARIANTS = ['base', 'clamp', 'med3', 'clampair', 'trem', 'vib03']
os.makedirs(OUT, exist_ok=True)
items, metrics = [], {v: [] for v in VARIANTS}
with torch.no_grad():
    for key, text in TEXTS:
        ph = g2p(normalize(text))
        idt = torch.tensor([g2p.tokens(ph)], dtype=torch.long)
        it = dict(id=key, cat=key, text=text, files={})
        yr = {}
        for v in VARIANTS:
            dm = {'base': 'base', 'clamp': 'clamp', 'med3': 'med3', 'clampair': 'clamp',
                  'trem': 'clamp', 'vib03': 'clamp'}[v]
            asr, F0, Nc, n = text_side(ts, idt, dur_mode=dm)
            x = trunk(vd, asr, F0, Nc)
            torch.manual_seed(0)
            nz = torch.randn(1, x.shape[2] * FHOP)
            kw = {}
            if v == 'trem':
                kw = dict(trem=0.18)
            elif v == 'vib03':
                kw = dict(vib=0.003, vib_hz=3.0)
            y = head_render(vd.head, x, F0, nz, **kw)
            if v == 'clampair':
                y = air(y, 3.0)
            yr[v] = y
            metrics[v].append([periodicity(y), *texture(y)])
        base = yr['base']
        for v in VARIANTS:
            y = yr[v]
            if v != 'base':
                y = y * (rms(base) / (rms(y) + 1e-9))
            fn = f'{key}_{v}.wav'
            sf.write(os.path.join(OUT, fn), np.clip(y, -1, 1), SR)
            it['files'][v] = f'{OUT}/{fn}'
        items.append(it)
        print(key, 'len base', len(base), 'clamp', len(yr['clamp']), 'done', flush=True)

res = {v: [round(float(t), 3) for t in np.mean(m, 0)] for v, m in metrics.items()}
json.dump(dict(items=items, metrics=res), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n           periodicity  flat2-6k  >4k')
for v in VARIANTS:
    print(f'  {v:10s} {res[v][0]:.3f}       {res[v][1]:.3f}     {res[v][2]:.3f}')
