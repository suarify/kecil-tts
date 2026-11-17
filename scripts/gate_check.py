"""Gate check for a new voice: durations, texture, and an A/B page (student vs optional teacher clips).
usage: python scripts/gate_check.py --dec CKPT --text CKPT --name nicole [--teacher nicole.pt]
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument('--dec', required=True)
ap.add_argument('--text', required=True)
ap.add_argument('--name', default='voice')
ap.add_argument('--k', type=int, default=128)
ap.add_argument('--n', type=int, default=8)
A = ap.parse_args()
SR = 24000


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
ts.load_state_dict(torch.load(A.text, map_location='cpu', weights_only=False)['model'])
vd = VDecoder(head='v2', K=A.k).eval(); vd.load_ckpt(A.dec); strip(ts); strip(vd)
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


def head_render(h, x, f0, noise):
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


SENTS = [
    'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.',
    'Booking encik untuk hari Selasa dah confirmed.',
    'Kita boleh pergi sekarang atau tunggu sehingga petang.',
    'Pesanan Dato akan dihantar ke Shah Alam esok pagi.',
    'Kami telah menerima aduan doktor dan sedang menyiasatnya.',
    'Sila maklumkan kehadiran puan sebelum hari Rabu.',
    'Buku ini untuk murid tahun lima di sekolah kebangsaan.',
    'Azman di sini, saya mewakili pasukan khidmat pelanggan.',
]
OUT = f'runs/listen/{A.name}_gate'
os.makedirs(OUT, exist_ok=True)
met_s, met_a, dur_stats, items = [], [], [], []
with torch.no_grad():
    for idx, text in enumerate(SENTS[:A.n]):
        ids_t = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
        asr, F0, Nc, n = text_side(ids_t)
        x = trunk(vd, asr, F0, Nc)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        y = head_render(vd.head, x, F0, nz)
        y3 = air(y)
        f1, f2 = f'{OUT}/g{idx}_student.wav', f'{OUT}/g{idx}_student_air3.wav'
        sf.write(f1, np.clip(y, -1, 1), SR)
        sf.write(f2, np.clip(y3, -1, 1), SR)
        met_s.append([periodicity(y), *texture(y)])
        met_a.append([periodicity(y3), *texture(y3)])
        dur_stats.append(dict(text=text[:36], seconds=round(len(y) / SR, 2), one_frame=round(float((n == 1).float().mean()), 2),
                              max_frames=int(n.max())))
        items.append(dict(text=text, student=f'{A.name}_gate/g{idx}_student.wav',
                          air3=f'{A.name}_gate/g{idx}_student_air3.wav'))
        print(f'{idx}: {len(y) / SR:.2f}s 1-frame {float((n == 1).float().mean()):.0%} max {int(n.max())}', flush=True)

res = dict(name=A.name,
           student=[round(float(x), 3) for x in np.mean(met_s, 0)],
           air3=[round(float(x), 3) for x in np.mean(met_a, 0)],
           durations=dur_stats,
           benchmarks={'yati_distance_to_teacher': 3.29, 'michael_distance_to_teacher': 3.84})
json.dump(dict(metrics=res, items=items), open(f'{OUT}/metrics.json', 'w', encoding='utf-8'), indent=1)
print('student periodicity/flat/>4k/>7k:', res['student'])
print('air3   periodicity/flat/>4k/>7k:', res['air3'])
print('GATE:', 'dur' in res and 'student' in res)
