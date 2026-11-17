"""Knob test on the REAL v0.4 weights: Chatterbox original vs v0.4 (watery/phaser) vs knob variants.

Same mechanism as scripts/tonal_knob.py, but:
  - uses the actual v0.4 checkpoints (runs/k16_final/dec_c4_last.pt + textA_c4_last.pt)
  - the comparison sentence list is chosen from clips that HAVE a Chatterbox original
    (runs/k15_v3/cb_clips2.pt), so the page can play 'original Chatterbox' vs 'after training'.
  - writes a test page runs/listen/knob_v04.html with play buttons + flatness stats.

    python scripts/knob_v04.py          # CPU, ~1-2 min
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

DEC = 'runs/k16_final/dec_c4_last.pt'
TEXT = 'runs/k16_final/textA_c4_last.pt'
KEEP = [1.0, 0.5, 0.25]
FC, CAP = 1500.0, 4.0
N = 10
OUT = 'runs/listen/knob_v04'

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


def head(h, x, f0, keep, noise):
    """SourceFilterHead2.forward with the noise-to-harmonic move (keep=1 -> v0.4 exactly)."""
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
    band = (fr >= FC).float()
    voiced = (f0raw > 10).float()[:, None]
    if keep != 1:
        Ph = ((gh * E.abs()) ** 2 * band).sum(1, keepdim=True)
        Pn = ((gn * Nz.abs()) ** 2 * band).sum(1, keepdim=True)
        s = torch.sqrt((Ph + (1 - keep ** 2) * Pn) / Ph.clamp_min(1e-9)).clamp(max=CAP)
        s = nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)   # no frame-to-frame flicker
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


def shelf(a, db=4.0, lo=1500.0, hi=4000.0):
    """FFT high-shelf: +db above hi Hz, ramp between lo and hi (keeps loudness of the rest untouched)."""
    a = np.asarray(a, dtype=np.float32)
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(len(a), 1 / SR)
    t = np.clip((f - lo) / (hi - lo), 0, 1)
    t = t * t * (3 - 2 * t)                        # smoothstep ramp
    return np.fft.irfft(A * (10 ** ((db / 20) * t)), len(a)).astype(np.float32)


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
    return (float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean()),
            float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean()))


# --- pick sentences that have a Chatterbox original, spread over categories ---
clips = torch.load(os.path.join(ROOT, 'runs/k15_v3/cb_clips2.pt'), map_location='cpu', weights_only=False)
plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open(os.path.join(ROOT, 'runs/k15_v3/gen_plan.csv'), encoding='utf-8'))}
ids = [i for i in clips if i in plan]
picked, seen = [], set()
for i in ids:                                   # round-robin across categories for variety
    c = plan[i][1]
    if c not in seen:
        seen.add(c); picked.append(i)
    if len(picked) >= N:
        break
for i in ids:                                   # top up
    if len(picked) >= N:
        break
    if i not in picked:
        picked.append(i)

os.makedirs(os.path.join(ROOT, OUT), exist_ok=True)
items, met = [], {k: [] for k in list(KEEP) + ['eq']}
tgt_met = []
with torch.no_grad():
    for i in picked:
        text, cat = plan[i]
        ref = normalize(text)
        ids_t = torch.tensor([g2p.tokens(g2p(ref))], dtype=torch.long)
        asr, F0, Nc = text_side(ids_t)
        x = trunk(asr, F0, Nc)
        torch.manual_seed(0)                    # same noise seed for every variant of a sentence
        nz = torch.randn(1, x.shape[2] * FHOP)
        it = dict(id=i, cat=cat, text=text, files={})
        tgt = clips[i].numpy()
        tf = f'{i}_chatterbox.wav'
        sf.write(os.path.join(ROOT, OUT, tf), np.clip(tgt, -1, 1), SR)
        it['files']['target'] = f'{OUT}/{tf}'
        tgt_met.append(flat(tgt))
        for k in KEEP:
            y = head(vd.head, x, F0, k, nz).squeeze().numpy()
            met[k].append(flat(y))
            name = f'{i}_keep{int(round(k * 100)):03d}.wav'
            sf.write(os.path.join(ROOT, OUT, name), np.clip(y, -1, 1), SR)
            it['files'][str(k)] = f'{OUT}/{name}'
        ye = shelf(head(vd.head, x, F0, 0.5, nz).squeeze().numpy(), db=4.0)
        met['eq'].append(flat(ye))
        sf.write(os.path.join(ROOT, OUT, f'{i}_knob50_eq4.wav'), np.clip(ye, -1, 1), SR)
        it['files']['eq'] = f'{OUT}/{i}_knob50_eq4.wav'
        items.append(it)
        print(i, cat, len(ids_t[0]), 'tokens', flush=True)

res = {('target' if k == 'target' else str(k)): [float(x) for x in np.array(v).mean(0)]
       for k, v in [('target', tgt_met)] + list(met.items())}
json.dump(dict(items=items, keep=KEEP, fc=FC, metrics=res),
          open(os.path.join(ROOT, OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nflatness 2-6 kHz (target 0.62-ish) / energy >4 kHz:')
print('  chatterbox target :', ' / '.join(f'{v:.3f}' for v in res['target']))
for k in list(KEEP) + ['eq']:
    print(f'  {str(k):8s}        :', ' / '.join(f'{v:.3f}' for v in res[str(k)]))
