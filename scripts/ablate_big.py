"""Render the BIG student (320/6 blocks) with clarity options and compare with v0.3.0 + teacher."""
import json
import os
import shutil
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

OUT = 'runs/listen/big'
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
vd = VDecoder(head='v2', K=128, dim=320, blocks=6).eval()
vd.load_ckpt('runs/kaggle_sm11/dec_big_last.pt'); strip(ts); strip(vd)
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


def head_render(h, x, f0, noise, gap_keep=1.0):
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
    Nz = (Nz0 + sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))) * gap_keep
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


def mcd(a, b):
    import librosa
    M = lambda x: np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR, n_mels=64) + 1e-8)  # noqa: E731
    A, B = M(a), M(b)
    T = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :T] - B[:, :T]).mean())


os.makedirs(OUT, exist_ok=True)
items, metrics = [], {k: [] for k in ('big', 'bigAir3', 'bigAir5')}
dists = []
with torch.no_grad():
    for key, text in TEXTS:
        ids = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
        asr, F0, Nc = text_side(ids)
        x = trunk(vd, asr, F0, Nc)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        y0 = head_render(vd.head, x, F0, nz)
        it = dict(id=key, cat=key, text=text, files={'big': f'{OUT}/{key}_big.wav',
                                                     'bigAir3': f'{OUT}/{key}_bigAir3.wav',
                                                     'bigAir5': f'{OUT}/{key}_bigAir5.wav'})
        for v, y in (('big', y0), ('bigAir3', air(y0, 3.0)), ('bigAir5', air(y0, 5.0))):
            sf.write(it['files'][v], np.clip(y, -1, 1), SR)
            metrics[v].append([periodicity(y), *texture(y)])
        teacher = {'iii': 'runs/listen/vc10/b/c0571.wav', 'call': 'runs/listen/vc15/old/c0184.wav'}[key]
        if os.path.exists(teacher):
            t, _ = sf.read(teacher)
            dists.append((key, mcd(y0, t)))
        small = f'runs/listen/clarity/{key}_faithful.wav'
        if os.path.exists(small):
            shutil.copy(small, f'{OUT}/{key}_small.wav')
            it['files']['small'] = f'{OUT}/{key}_small.wav'
        items.append(it)
        print(key, 'done', flush=True)

res = {v: [round(float(t), 3) for t in np.mean(m, 0)] for v, m in metrics.items()}
json.dump(dict(items=items, metrics=res, dist=dists), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n            periodicity  flat2-6k  >4k    >7k')
for v in res:
    print(f'  {v:8s} {res[v][0]:.3f}       {res[v][1]:.3f}     {res[v][2]:.3f}  {res[v][3]:.3f}')
print('big vs teacher distance:', [(k, round(d, 2)) for k, d in dists])
