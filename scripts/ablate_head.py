"""Ablation round 2: the noise part is the watery culprit (owner listening test).

Owner verdict on round 1: 'harmonics only' sounds best but muffled (no noise = no brightness fill);
phase OFF adds distortion (learned phase is good). So: keep the learned phase, keep the noise ENERGY
(brightness), but move the noise out of the harmonic bins into the gaps between them in voiced frames.

  target    the original Chatterbox clip
  v04       full v0.4 (watery/phaser)
  harm      harmonics only, learned phase (owner's pick; muffled)
  fixC      v0.4 with noise masked into inter-harmonic gaps (voiced, >800 Hz), energy preserved
  fixC2     fixC + only half the noise energy (rest moved to harmonics, energy-compensated)
  fixD      fixC2 + pulsed-noise sources removed (plain noise only, gap-masked)

Writes runs/listen/ablate/*.wav + items.json; page via scripts/ablate_page.py.
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
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402

DEC, TEXT = 'runs/k16_final/dec_c4_last.pt', 'runs/k16_final/textA_c4_last.pt'
OUT, N = 'runs/listen/ablate', 6
torch.set_num_threads(4)

ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(os.path.join(ROOT, TEXT), map_location='cpu')['model'])
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(os.path.join(ROOT, DEC))
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


def head_parts(h, x, f0, noise, harm_only=False, noise_only=False, interharm=False, keep=1.0, no_mod=False,
               burst_uv=False, comp_burst=False):
    """SourceFilterHead2.forward with options:
    interharm: mask the noise out of the harmonic bins into the gaps (voiced, >800 Hz), energy preserved.
    keep<1: energy-compensated noise->harmonic move above 1.5 kHz. no_mod: drop the pulsed-noise sources.
    burst_uv: the 3 ms burst path only in UNVOICED frames. comp_burst: the gated burst energy counts in the
    keep move, so harmonic gains are boosted to hold the band energy (full energy conservation)."""
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
    e, cyc, uv = sine_bank(f0, h.K, None)
    E = h.stft(e)
    Nz0 = h.stft(noise)
    Nm = [] if no_mod else [h.stft(noise * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([f0.repeat_interleave(h.up, 1), f0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None]
    Nz = Nz0 + sum(g[:, 4 + j] * Nm[j] for j in range(len(Nm)))
    voiced = (f0raw > 10).float()
    band_mask = (fr[0, :, 0] >= 800).float()[None, :, None]
    if interharm:
        frac = (ratio - ratio.round()).abs()
        hp = torch.exp(-((frac / 0.22) ** 2)) * band_mask * voiced[:, None, :]     # near-harmonic weight
        w = 1 - hp.clamp(max=1)                                                    # 1 in the gaps
        Nz2 = Nz * w
        P0 = ((Nz.abs() ** 2) * band_mask).sum(1, keepdim=True)
        P1 = ((Nz2.abs() ** 2) * band_mask).sum(1, keepdim=True)
        s = torch.sqrt(P0 / P1.clamp_min(1e-12)).clamp(max=4.0)                    # keep the noise energy
        s = nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)
        Nz = Nz2 * s
    # burst path (3 ms), computed early so its energy can join the keep compensation
    if h.burst:
        gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
        gb = torch.cat([gb, gb[..., -1:]], -1)
        Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
        Yb = Nb * gb
        band_b = (torch.linspace(0, SR / 2, 76)[None, :, None] >= 1500).float()
        Pb = (Yb.abs() ** 2 * band_b).sum(1)                                       # [B, Tb] burst band energy
        Pb = nn.functional.interpolate(Pb[:, None, :], size=T + 1, mode='linear', align_corners=False)[:, 0]
        Pb = Pb * voiced                       # [B, 1, T+1] by broadcast
        Pb = Pb.reshape(B, T + 1)
        if burst_uv:
            vb = torch.cat([f0.repeat_interleave(4, 1), f0[:, -1:]], 1) > 10       # voiced, at burst-frame rate
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
    if h.burst:
        yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    if harm_only:
        return yh, None
    if noise_only:
        return None, yn
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
    """(flatness 2-6 kHz, share of energy >4 kHz) — target: 0.62 / 0.084."""
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


clips = torch.load(os.path.join(ROOT, 'runs/k15_v3/cb_clips2.pt'), map_location='cpu', weights_only=False)
plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open(os.path.join(ROOT, 'runs/k15_v3/gen_plan.csv'), encoding='utf-8'))}
ids = [i for i in clips if i in plan][:N]

os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
VARIANTS = ['target', 'v04', 'harm', 'fixC', 'fixG', 'fixH']
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
        y04 = None
        for v in VARIANTS:
            if v == 'target':
                y = clips[i].numpy()
            elif v == 'v04':
                yh, yn = head_parts(vd.head, x, F0, nz)
                y = (yh + yn).squeeze().numpy()
                y04 = y.copy()
            elif v == 'harm':
                yh, _ = head_parts(vd.head, x, F0, nz, harm_only=True)
                y = yh.squeeze().numpy()
            elif v == 'fixC':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True)
                y = (yh + yn).squeeze().numpy()
            elif v == 'fixG':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True, comp_burst=True, keep=0.5)
                y = (yh + yn).squeeze().numpy()
            elif v == 'fixH':
                yh, yn = head_parts(vd.head, x, F0, nz, interharm=True, burst_uv=True)
                y = (yh + yn).squeeze().numpy()
            if v != 'target' and y04 is not None:
                y = y * (rms(y04) / (rms(y) + 1e-9))
            fn = f'{i}_{v}.wav'
            sf.write(os.path.join(ROOT, OUT, fn), np.clip(y, -1, 1), SR)
            it['files'][v] = f'{OUT}/{fn}'
            met[v].append([periodicity(y), *texture(y)])
        items.append(it)
        print(i, cat, 'done', flush=True)

res = {v: [float(x) for x in np.array(m).mean(0)] for v, m in met.items()}
json.dump(dict(items=items, variants=VARIANTS, metrics=res),
          open(os.path.join(ROOT, OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\n            periodicity   flatness2-6k   >4k share     (target ~0.70 / 0.62 / 0.084)')
for v in VARIANTS:
    p, fl, o4 = res[v]
    print(f'  {v:10s} {p:6.3f}      {fl:6.3f}       {o4:6.3f}')
