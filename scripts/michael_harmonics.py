"""Why does Michael sound watery with every head setting? Measure the HARMONIC path itself:
frame-to-frame flicker of the harmonic gains (gh), phase rotation speed per frame, F0 jitter, and the
amplitude-modulation spectrum (30-90 Hz sidebands = watery). Compare Michael with Yati v0.4."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn as nn

from ms_g2p import G2P
from ms_norm import normalize
from student import TextStudent
from vhead import FHOP, SR, VDecoder, sine_bank
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations


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
    vd = VDecoder(head='v2', dim=160, blocks=4).eval(); strip(vd)
    vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=True)
    vd.head.M.copy_(ck['M']); vd.head.Mb.copy_(ck['Mb'])
    return ts, vd


def load_yati():
    ts = TextStudent('s+mlp').eval()
    ts.load_state_dict(torch.load('runs/k16_final/textA_c4_last.pt', map_location='cpu')['model'])
    vd = VDecoder(head='v2').eval(); vd.load_ckpt('runs/k16_final/dec_c4_last.pt')
    return ts, vd


def text_side(ts, ids):
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
    F0, N = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return asr @ aln, F0, N


def trunk(vd, asr, F0c, N):
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


def head_gains(h, x):
    B, _, n = x.shape
    T = n * h.up
    z = h.norm0(h.inp(x).transpose(1, 2)).transpose(1, 2)
    for b in h.blocks:
        z = b(z)
    hn = h.norm1(z.transpose(1, 2))
    o = h.out(hn).view(B, T, h.C * h.G + 1)
    o = torch.cat([o, o[:, -1:]], 1)
    g = (o[..., :-1].reshape(B, T + 1, h.C, h.G) @ h.M.T).permute(0, 2, 3, 1)
    return g, o


g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
text = 'Selamat pagi, ini ujian suara untuk mengukur kualiti.'
ids = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)

for name, loader in (('michael', load_michael), ('yati', load_yati)):
    ts, vd = loader()
    with torch.no_grad():
        asr, F0, N = text_side(ts, ids)
        x = trunk(vd, asr, F0, N)
        g, o = head_gains(vd.head, x)
        voiced = (torch.cat([F0.repeat_interleave(vd.head.up, 1), F0[:, -1:]], 1) > 10)[0].numpy()
        # harmonic gain flicker: std of frame-to-frame log-gain change (voiced frames, 1-4 kHz bins)
        gh = g[0, 0]                                                    # [nb, T+1]
        dgh = (gh[:, 1:] - gh[:, :-1])
        vcol = voiced[1:] & voiced[:-1]
        fr = torch.linspace(0, SR / 2, vd.head.nb)
        band = ((fr > 1000) & (fr < 4000)).numpy()
        flick = float(dgh[band][:, vcol].std())
        # noise gain flicker for comparison
        gn = g[0, 3]
        dgn = (gn[:, 1:] - gn[:, :-1])
        nflick = float(dgn[band][:, vcol].std())
        # phase rotation per frame: angle of u (units of 2*pi) between frames
        u = torch.complex(g[0, 1], g[0, 2])
        u = u / u.abs().clamp_min(1e-4)
        du = (u[:, 1:] * u[:, :-1].conj()).angle().abs()
        pband = band
        phase_step = float(du[pband][:, vcol].median())
        # F0 jitter
        f0 = F0[0].numpy()
        vv = f0 > 10
        jit = float(np.abs(np.diff(f0[vv])).mean() / np.median(f0[vv]) * 100)
        print(f'{name:8s} gh flicker {flick:.3f} | gn flicker {nflick:.3f} | phase step/frame {phase_step:.3f} cyc | F0 jitter {jit:.2f}%/frame | voiced frames {int(vv.sum())}')

        # amplitude-modulation spectrum of the harmonic-only output (30-90 Hz band energy)
        uu = u
        e, cyc, uv = sine_bank(F0, vd.head.K, None)
        E = vd.head.stft(e)
        shift = 0.5 * torch.tanh(o[..., -1])[:, None]
        ratio = torch.linspace(0, SR / 2, vd.head.nb)[None, :, None] / torch.cat([F0.repeat_interleave(vd.head.up, 1), F0[:, -1:]], 1).clamp_min(60)[:, None]
        ghT = torch.exp(g[:, 0].clamp(max=6))
        yh = torch.istft(E * (ghT * uu * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)), vd.head.n_fft, vd.head.hop, window=vd.head.window, length=x.shape[2] * FHOP).squeeze().numpy()
        import librosa
        env = np.abs(librosa.stft(yh, n_fft=256, hop_length=64))[:64].sum(0)
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        fs = SR / 64
        fq = np.fft.rfftfreq(len(env), 1 / fs)
        m = (fq > 30) & (fq < 90)
        print(f'         AM sidebands 30-90 Hz (harmonics only): {spec[m].mean() / (spec.mean() + 1e-9):.2f}x average')
