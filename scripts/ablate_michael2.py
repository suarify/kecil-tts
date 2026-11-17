"""Michael ablation round 2: everything still watery -> test the LOW band and the mask width.

Michael's male F0 (~120 Hz) means the 0-800 Hz band (left untouched by FIX H) carries wide
inter-harmonic noise gaps, and the 0.22-f0 mask may be too narrow. Variants:
  fixH      current v0.2.1
  maskAll   gap mask over the whole band (>=150 Hz), energy preserved
  maskWide  wider mask (sigma 0.35 f0) + whole band
  gainSmooth harmonic gains averaged over 5 bins across frequency (kills comb notches)
  combo     maskAll + gainSmooth
  lowOff    noise gains zeroed below 800 Hz
  yati      Yati v0.4.1 rendering of the same text (reference of acceptable wateriness)
"""
import csv
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
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

OUT, N = 'runs/listen/michael_ablate2', 4
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


ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
ts = TextStudent('s+mlp').eval(); strip(ts)
ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
vd = VDecoder(head='v2', dim=160, blocks=4).eval(); strip(vd)
vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])

ty = TextStudent('s+mlp').eval()
ty.load_state_dict(torch.load('runs/k16_final/textA_c4_last.pt', map_location='cpu')['model'])
vy = VDecoder(head='v2').eval(); vy.load_ckpt('runs/k16_final/dec_c4_last.pt')

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')


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


def head_parts(h, x, f0, noise, band_lo=800.0, sigma=0.22, gain_smooth=0, low_off=False):
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
    if gain_smooth:
        ghf = nn.functional.avg_pool1d(gh.permute(0, 2, 1), gain_smooth, 1, gain_smooth // 2, count_include_pad=False)
        gh = ghf.permute(0, 2, 1)
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
    band_mask = (fr[0, :, 0] >= band_lo).float()[None, :, None]
    frac = (ratio - ratio.round()).abs()
    hp = torch.exp(-((frac / sigma) ** 2)) * band_mask * voiced[:, None, :]
    w = 1 - hp.clamp(max=1)
    Nz2 = Nz * w
    P0 = ((Nz.abs() ** 2) * band_mask).sum(1, keepdim=True)
    P1 = ((Nz2.abs() ** 2) * band_mask).sum(1, keepdim=True)
    s = torch.sqrt(P0 / P1.clamp_min(1e-12)).clamp(max=4.0)
    s = nn.functional.avg_pool1d(s, 5, 1, 2, count_include_pad=False)
    Nz = Nz2 * s
    if low_off:
        gn = gn * (fr >= 800).float()
    gb = torch.exp((h.out_b(hn).view(B, n * 4, h.Gb) @ h.Mb.T).clamp(max=6)).transpose(1, 2)
    gb = torch.cat([gb, gb[..., -1:]], -1)
    vb = torch.cat([f0.repeat_interleave(4, 1), f0[:, -1:]], 1) > 10
    gb = gb * (~vb).float()[:, None, :]
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    yn = torch.istft(Nz * gn, h.n_fft, h.hop, window=h.window, length=n * FHOP)
    Nb = torch.stft(noise.flip(-1), 150, 75, window=h.wb, return_complex=True)
    yn = yn + torch.istft(Nb * gb, 150, 75, window=h.wb, length=n * FHOP)
    return yh + yn


def rms(a):
    return float(np.sqrt((np.asarray(a, np.float32) ** 2).mean()))


plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = ['c0313', 'c0551', 'c0571', 'c0726'][:N]

os.makedirs(OUT, exist_ok=True)
VARIANTS = ['fixH', 'maskAll', 'maskWide', 'gainSmooth', 'combo', 'lowOff', 'yati']
items = []
with torch.no_grad():
    for i in ids:
        text, cat = plan[i]
        ids_t = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
        asr, F0, Nc = text_side(ts, ids_t)
        x = trunk(vd, asr, F0, Nc)
        torch.manual_seed(0)
        nz = torch.randn(1, x.shape[2] * FHOP)
        it = dict(id=i, cat=cat, text=text, files={})
        yref = None
        for v in VARIANTS:
            if v == 'yati':
                asr_y, F0_y, N_y = text_side(ty, ids_t)
                xy = trunk(vy, asr_y, F0_y, N_y)
                nzy = torch.randn(1, xy.shape[2] * FHOP)
                g = None
                # yati: same FIX H recipe on her head
                hd = vy.head
                B, _, n = xy.shape
                T = n * hd.up
                z = hd.norm0(hd.inp(xy).transpose(1, 2)).transpose(1, 2)
                for b in hd.blocks:
                    z = b(z)
                hn = hd.norm1(z.transpose(1, 2))
                o = hd.out(hn).view(B, T, hd.C * hd.G + 1); o = torch.cat([o, o[:, -1:]], 1)
                gg = (o[..., :-1].reshape(B, T + 1, hd.C, hd.G) @ hd.M.T).permute(0, 2, 3, 1)
                gh, gn = torch.exp(gg[:, 0].clamp(max=6)), torch.exp(gg[:, 3].clamp(max=6))
                uu = torch.complex(gg[:, 1], gg[:, 2]); uu = uu / uu.abs().clamp_min(1e-4)
                ee, cyc, uv = sine_bank(F0_y, hd.K, None)
                Ey = hd.stft(ee); Nz0 = hd.stft(nzy)
                Nm = [hd.stft(nzy * uv * fn(2 * torch.pi * m * cyc)) for m in range(1, hd.mod + 1) for fn in (torch.cos, torch.sin)]
                f0raw = torch.cat([F0_y.repeat_interleave(hd.up, 1), F0_y[:, -1:]], 1)
                f0f = f0raw.clamp_min(60)
                fr = torch.linspace(0, SR / 2, hd.nb)[None, :, None]
                ratio = fr / f0f[:, None]
                shift = 0.5 * torch.tanh(o[..., -1])[:, None]
                Nzy = Nz0 + sum(gg[:, 4 + j] * Nm[j] for j in range(len(Nm)))
                voi = (f0raw > 10).float()
                bm = (fr[0, :, 0] >= 800).float()[None, :, None]
                fq = (ratio - ratio.round()).abs()
                wp = 1 - (torch.exp(-((fq / 0.22) ** 2)) * bm * voi[:, None, :]).clamp(max=1)
                N2 = Nzy * wp
                P0_ = ((Nzy.abs() ** 2) * bm).sum(1, keepdim=True)
                P1_ = ((N2.abs() ** 2) * bm).sum(1, keepdim=True)
                sc = torch.sqrt(P0_ / P1_.clamp_min(1e-12)).clamp(max=4.0)
                sc = nn.functional.avg_pool1d(sc, 5, 1, 2, count_include_pad=False)
                Nzy = N2 * sc
                Hh = gh * uu * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
                yy = torch.istft(Ey * Hh + Nzy * gn, hd.n_fft, hd.hop, window=hd.window, length=n * FHOP)
                gb = torch.exp((hd.out_b(hn).view(B, n * 4, hd.Gb) @ hd.Mb.T).clamp(max=6)).transpose(1, 2)
                gb = torch.cat([gb, gb[..., -1:]], -1)
                vb = torch.cat([F0_y.repeat_interleave(4, 1), F0_y[:, -1:]], 1) > 10
                gb = gb * (~vb).float()[:, None, :]
                Nb = torch.stft(nzy.flip(-1), 150, 75, window=hd.wb, return_complex=True)
                yy = yy + torch.istft(Nb * gb, 150, 75, window=hd.wb, length=n * FHOP)
                y = yy.squeeze().numpy()
            else:
                kw = {}
                if v == 'maskAll':
                    kw = dict(band_lo=150.0)
                elif v == 'maskWide':
                    kw = dict(band_lo=150.0, sigma=0.35)
                elif v == 'gainSmooth':
                    kw = dict(gain_smooth=5)
                elif v == 'combo':
                    kw = dict(band_lo=150.0, gain_smooth=5)
                elif v == 'lowOff':
                    kw = dict(low_off=True)
                y = head_parts(vd.head, x, F0, nz, **kw).squeeze().numpy()
            if v == 'fixH':
                yref = y.copy()
            if yref is not None:
                y = y * (rms(yref) / (rms(y) + 1e-9))
            fn = f'{i}_{v}.wav'
            sf.write(os.path.join(OUT, fn), np.clip(y, -1, 1), SR)
            it['files'][v] = f'{OUT}/{fn}'
        items.append(it)
        print(i, cat, 'done', flush=True)

json.dump(dict(items=items), open(os.path.join(OUT, 'items.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('round 2 written to', OUT)
