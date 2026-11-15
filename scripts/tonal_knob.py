"""Zero-training test for the damp sound: move energy from the decoder's NOISE part back to its HARMONIC part.

The v0.2 decoder writes voiced sound as (sine waves x harmonic gain) + (random noise x noise gain). Training on
real clips with a spectral loss let noise creep in between the harmonics. Here, in voiced frames above --fc Hz,
the noise gain is multiplied by `keep` and the harmonic gain is raised so the energy in that band is unchanged.
keep = 1 is v0.2 exactly. Nothing is retrained; if a setting sounds right it can be baked into the ONNX export.

    python scripts/tonal_knob.py --n 20 --keep 1 0.5 0.25      # writes runs/listen/v03/*.wav + runs/listen/v03.html
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder, sine_bank  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dec', default='runs/k12/ckpt/dec_last.pt')
ap.add_argument('--text', default='runs/k12/ckpt/textA_last.pt')
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--keep', type=float, nargs='+', default=[1.0, 0.5, 0.25])
ap.add_argument('--fc', type=float, default=1500.0)
ap.add_argument('--cap', type=float, default=4.0, help='largest allowed rise of the harmonic gain (x)')
ap.add_argument('--out', default='runs/listen/v03')
ap.add_argument('--stats-only', action='store_true')
A = ap.parse_args()
torch.set_num_threads(4)

ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(os.path.join(ROOT, A.text), map_location='cpu')['model'])
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(os.path.join(ROOT, A.dec))
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
SHARE = []


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


def head(h, x, f0, keep, noise):
    """SourceFilterHead2.forward with the noise-to-harmonic move."""
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
    Nz = h.stft(noise)
    Nm = [h.stft(noise * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, h.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([f0.repeat_interleave(h.up, 1), f0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, h.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None]
    Nz = Nz + sum(g[:, 4 + j] * Nm[j] for j in range(2 * h.mod))
    band = (fr >= A.fc).float()
    voiced = (f0raw > 10).float()[:, None]
    Ph = ((gh * E.abs()) ** 2 * band).sum(1, keepdim=True)
    Pn = ((gn * Nz.abs()) ** 2 * band).sum(1, keepdim=True)
    v = voiced[0, 0] > 0
    SHARE.append((float(Pn[0, 0, v].sum()), float(Ph[0, 0, v].sum())))
    if keep != 1:
        s = torch.sqrt((Ph + (1 - keep ** 2) * Pn) / Ph.clamp_min(1e-9)).clamp(max=A.cap)
        s = torch.nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)          # no frame-to-frame flicker
        m = band * voiced
        gh = gh * (1 + m * (s - 1))
        gn = gn * (1 + m * (keep - 1))
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    y = torch.istft(E * Hh + Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    if h.burst:
        gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
        gb = torch.cat([gb, gb[..., -1:]], -1)
        Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
        y = y + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    return y


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


def flat(a):
    import librosa
    S = np.abs(librosa.stft(np.asarray(a, dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    return float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean()), float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())


t200 = pd.read_csv(os.path.join(ROOT, 'kaggle/ds_g2p/test_200.tsv'), sep='\t', dtype=str, keep_default_na=False)
pick = t200.iloc[np.linspace(0, len(t200) - 1, A.n).round().astype(int)]
os.makedirs(os.path.join(ROOT, A.out), exist_ok=True)
items, met = [], {k: [] for k in A.keep}
with torch.no_grad():
    for r in pick.itertuples():
        ref = normalize(r.text)
        ids = torch.tensor([g2p.tokens(g2p(ref))], dtype=torch.long)
        asr, F0, N = text_side(ids)
        x = trunk(asr, F0, N)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        it = dict(id=r.id, text=r.text, files={})
        for k in A.keep:
            y = head(vd.head, x, F0, k, nz).squeeze().numpy()
            met[k].append(flat(y))
            if not A.stats_only:
                name = f'{r.id}_keep{int(round(k * 100)):03d}.wav'
                sf.write(os.path.join(ROOT, A.out, name), np.clip(y, -1, 1), SR)
                it['files'][str(k)] = f'{os.path.basename(A.out)}/{name}'
        items.append(it)
        print(r.id, len(ids[0]), 'tokens', flush=True)
pn = sum(a for a, _ in SHARE[::len(A.keep)])
ph = sum(b for _, b in SHARE[::len(A.keep)])
print(f'voiced frames above {A.fc:.0f} Hz in v0.2: noise part {pn / (pn + ph):.0%} of the energy, harmonic part {ph / (pn + ph):.0%}')
for k in A.keep:
    m = np.array(met[k]).mean(0)
    print(f'keep {k:4.2f}: flatness 2-6 kHz {m[0]:.3f}   energy above 4 kHz {m[1]:.3f}')
json.dump(dict(items=items, keep=A.keep, fc=A.fc, metrics={str(k): np.array(met[k]).mean(0).tolist() for k in A.keep}),
          open(os.path.join(ROOT, A.out, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
