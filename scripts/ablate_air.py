"""Round 4: Michael de-chop + Yati brightness (the 'got room for improvement' pass).

Two problems, one harness:
  MICHAEL  harmonics sound good but choppy  -> burst/none, soft bursts, less gating, gain smoothing in time
  YATI     muffled / dark (target is dark)  -> air shelf (linear-phase FIR), harmonic exciter, gap_keep

Variants per voice (all loudness-matched to the base of that voice):
  base       Michael: hang7 gated | Yati: published FIX H (21-frame)
  burstSoft  bursts kept everywhere at 30% inside voiced frames (nothing can be chopped)
  burstFree  bursts kept everywhere at full gain (the old v0.2 behaviour)
  smooth     base + 3-frame temporal smoothing of the harmonic/noise gains (removes stairsteps)
  air4       base + +4 dB high shelf above 6.5 kHz (adds brightness, zero-phase filtfilt here)
  air6x      base + +6 dB shelf + harmonic exciter (2nd-harmonic, 5-10 kHz, 12%)
  keep60     base + 60% of the gap noise moved into the harmonics (a little more solidity)

Writes runs/listen/air/*.wav + items.json (two voice sections) + runs/listen/air.html.
"""
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402
from torch.nn.utils import remove_weight_norm  # noqa: E402
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations  # noqa: E402

OUT, N = 'runs/listen/air', 4
torch.set_num_threads(4)


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


def load_michael():
    ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
    ts = TextStudent('s+mlp').eval(); strip(ts)
    ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
    vd = VDecoder(head='v2', dim=160, blocks=4, K=128).eval(); strip(vd)
    vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
    vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])
    return ts, vd


def load_yati():
    ts = TextStudent('s+mlp').eval()
    ts.load_state_dict(torch.load('runs/k16_final/textA_c4_last.pt', map_location='cpu')['model'])
    vd = VDecoder(head='v2').eval(); vd.load_ckpt('runs/k16_final/dec_c4_last.pt')
    return ts, vd


def text_side(t_, ids):
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


def head_render(h, x, f0, noise, hang=7, burst_mode='gated', soft=0.3, gap_keep=1.0, smooth=0, harm_only=False):
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
    if smooth:
        gh = nn.functional.avg_pool1d(gh, smooth, 1, smooth // 2, count_include_pad=False)
        gn = nn.functional.avg_pool1d(gn, smooth, 1, smooth // 2, count_include_pad=False)
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
    if gap_keep != 1:
        band15 = (fr >= 1500).float()
        Ph = ((gh * E.abs()) ** 2 * band15).sum(1, keepdim=True)
        Pn = ((Nz.abs() * gn) ** 2 * band15).sum(1, keepdim=True)
        sc = torch.sqrt((Ph + (1 - gap_keep ** 2) * Pn) / Ph.clamp_min(1e-9)).clamp(max=3.0)
        sc = nn.functional.avg_pool1d(sc, 5, 1, 2, count_include_pad=False)
        m = band15 * voiced[:, None, :]
        gh = gh * (1 + m * (sc - 1))
        Nz = Nz * gap_keep
    gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
    gb = torch.cat([gb, gb[..., -1:]], -1)
    vbb = torch.cat([f0.repeat_interleave(4, 1), f0[:, -1:]], 1) > 10
    if burst_mode == 'gated':
        if hang > 0:
            vb = nn.functional.max_pool1d(vbb.float()[None], hang, 1, hang // 2)[0] > 0
        else:
            vb = vbb
        gb = gb * (~vb).float()[:, None, :]
    elif burst_mode == 'soft':
        gb = gb * torch.where(vbb, torch.tensor(soft), torch.tensor(1.0))[None, :, :]
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    if harm_only:
        return yh.squeeze().numpy()
    yn = torch.istft(Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
    yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    return (yh + yn).squeeze().numpy()


def air(y, db, fc=6500.0):
    """zero-phase high-shelf approximation: y + (10^(db/20)-1) * highpass(y)"""
    b, a = butter(4, fc / (SR / 2), 'high')
    return y + (10 ** (db / 20) - 1) * filtfilt(b, a, y)


def exciter(y, mix=0.12, lo=5000.0, hi=10500.0):
    d = np.sign(y) * y ** 2                     # 2nd harmonic generation
    b, a = butter(4, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], 'band')
    return y + mix * filtfilt(b, a, d) * (np.abs(y).max() + 1e-9)


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
    hi = float((P[f > 7000][:, keep].sum(0) / tot[keep]).mean())
    return flat, over4, hi


def cons_energy(y):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    fq = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(fq > 2000) & (fq < 8000)].sum(0)
    flat = np.exp(np.log(S + 1e-9).mean(0)) / (S.mean(0) + 1e-9)
    return float(band[flat > np.percentile(flat, 70)].mean())


plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = ['c0313', 'c0551', 'c0571', 'c0726'][:N]
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')

VOICES = {
    'michael': dict(load=load_michael, base=dict(hang=7),
                    variants={'base': {}, 'burstSoft': dict(burst_mode='soft'),
                              'burstFree': dict(burst_mode='free'), 'smooth': dict(smooth=3),
                              'air4': {}, 'air6x': {}, 'keep60': dict(gap_keep=0.6)}),
    'yati': dict(load=load_yati, base=dict(hang=21),
                 variants={'base': {}, 'burstSoft': dict(burst_mode='soft'), 'smooth': dict(smooth=3),
                           'air4': {}, 'air6x': {}, 'keep60': dict(gap_keep=0.6), 'hang7': dict(hang=7)}),
}

os.makedirs(OUT, exist_ok=True)
items, metrics = [], {}
with torch.no_grad():
    for vname, cfg in VOICES.items():
        ts, vd = cfg['load']()
        for i in ids:
            text, cat = plan[i]
            idt = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
            asr, F0, Nc = text_side(ts, idt)
            x = trunk(vd, asr, F0, Nc)
            torch.manual_seed(0)
            nz = torch.randn(1, x.shape[2] * FHOP)
            it = dict(voice=vname, id=i, cat=cat, text=text, files={})
            ybase = None
            for v, over in cfg['variants'].items():
                kw = dict(cfg['base']); kw.update(over)
                if kw.get('burst_mode') == 'free':
                    kw['burst_mode'] = 'soft'; kw['soft'] = 1.0
                y = head_render(vd.head, x, F0, nz, **kw)
                if v == 'air4':
                    y = air(y, 4.0)
                elif v == 'air6x':
                    y = exciter(air(y, 6.0))
                if v == 'base':
                    ybase = y.copy()
                else:
                    y = y * (rms(ybase) / (rms(y) + 1e-9))
                fn = f'{vname}_{i}_{v}.wav'
                sf.write(os.path.join(OUT, fn), np.clip(y, -1, 1), SR)
                it['files'][v] = f'{OUT}/{fn}'
                m = metrics.setdefault(vname, {}).setdefault(v, [])
                m.append([periodicity(y), *texture(y), cons_energy(y)])
            items.append(it)
            print(vname, i, 'done', flush=True)

summary = {v: {k: [round(float(t), 3) for t in np.mean(m, 0)] for k, m in d.items()} for v, d in metrics.items()}
json.dump(dict(items=items, metrics=summary), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for v, d in summary.items():
    print('\n', v, '(periodicity, flat2-6k, >4k, >7k, consonant energy)')
    for k, m in d.items():
        print(f'  {k:10s}', m)
