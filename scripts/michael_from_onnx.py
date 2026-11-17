"""Rebuild Michael's torch model from the dequantized ONNX weights, then re-export with FIX H.

The compiled graph keeps the torch parameter names, so: dequantize every weight, collect a
state_dict, load it into TextStudent('s+mlp') + VDecoder(head='v2'), and sanity-check the
reconstruction against the ONNX (durations and F0 curves must match; waveform is compared on
deterministic parts). usage: python scripts/michael_from_onnx.py [--export]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.stubs'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paradee_ms'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnx
from onnx import numpy_helper
import torch

m = onnx.load('runs/michael_v02.onnx')
g = m.graph

# 1. run the dequantize chain manually: build a tensor value map
const = {i.name: numpy_helper.to_array(i) for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}

# follow every DequantizeLinear / Cast-from-fp16 node
vals = dict(const)
for _ in range(3):
    for n in g.node:
        if n.op_type == 'DequantizeLinear':
            try:
                q, s, z = vals[n.input[0]], vals[n.input[1]], vals[n.input[2]]
                s = s.astype(np.float32); z = z.astype(np.float32)
                ax = next(a.i for a in n.attribute if a.name == 'axis')
                shape = [1] * q.ndim; shape[ax] = -1
                vals[n.output[0]] = (q.astype(np.float32) - z.reshape(shape)) * s.reshape(shape)
            except Exception:
                pass
        elif n.op_type == 'Cast' and n.input[0] in vals:
            to = next(a.i for a in n.attribute if a.name == 'to')
            if to == 1 and vals[n.input[0]].dtype != np.float32:
                vals[n.output[0]] = vals[n.input[0]].astype(np.float32)

names = [k for k in vals if k.endswith('.weight') or k.endswith('.bias') or k.endswith('.style') or
         k.endswith('.weight_orig') or 'running_' in k or k.endswith('fwd') or k.endswith('inv') or k.endswith('w2')]
print(len(names), 'tensor names collected')

# 2. load into the torch modules
from student import TextStudent
from vhead import VDecoder, SR, FHOP  # noqa: F401

sd = {n: torch.from_numpy(np.array(vals[n], np.float32, copy=True)) for n in names}
ts = TextStudent('s+mlp').eval()
vd = VDecoder(head='v2').eval()
for label, mod, keys in (('ts', ts, [k for k in sd if k.startswith('ts.')]),
                         ('vd', vd, [k for k in sd if k.startswith('vd.')])):
    sub = {k[len(label) + 1:]: sd[k] for k in keys}
    miss, unexp = mod.load_state_dict(sub, strict=False)
    print(label, 'missing:', list(miss)[:8], '| unexpected:', list(unexp)[:8])
    if 'style' in sub:
        print(' ', label, 'style loaded, shape', sub['style'].shape)

# 3. zero-noise parity: patch a copy of the ONNX to replace RandomNormalLike with a zeros ConstantOfShape,
#    so ONNX out = harmonics only; compare with the torch head called with noise=0
import onnxruntime as ort
import soundfile as sf

m2 = onnx.load('runs/michael_v02.onnx')
g2 = m2.graph
rn = next(n for n in g2.node if n.op_type == 'RandomNormalLike')
shape_t = next(numpy_helper.to_array(i) for i in g2.initializer if i.name == rn.input[0])
g2.node.remove(rn)
g2.initializer.extend([numpy_helper.from_array(np.zeros(int(shape_t[0]), np.float32), 'zero_noise')])
nz_node = onnx.helper.make_node('Expand', ['zero_noise', rn.input[0]], [rn.output[0]])
g2.node.insert(0, nz_node)
onnx.checker.check_model(m2)
p_zero = 'runs/michael_v02_zeronoise.onnx'
onnx.save(m2, p_zero)
sess = ort.InferenceSession(p_zero, providers=['CPUExecutionProvider'])

from ms_g2p import G2P
from ms_norm import normalize

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
text = 'Selamat pagi, ini ujian suara Michael versi baru.'
ids = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)
y_onnx = sess.run(None, {'input_ids': ids.numpy(), 'speed': np.ones(1, np.float32)})[0][0]
sf.write('runs/michael_zeronoise_onnx.wav', y_onnx, SR)

# torch side: replicate Paradee2.forward up to head, then call the ORIGINAL torch head with noise=0
import torch.nn as nn  # noqa: E402
t = ts
p = t.predictor
s = t.style
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
n_ = torch.clamp(torch.round(dur), min=1).long()
end = torch.cumsum(n_, 0)
start = end - n_
f = torch.arange(int(end[-1]))[None]
aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
F0, N = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
with torch.no_grad():
    xh = vd.encode(torch.cat([asr @ aln, vd.F0_conv(F0.unsqueeze(1)), vd.N_conv(N.unsqueeze(1))], 1), vd.style)
    asr_res = vd.asr_res(asr @ aln)
    res = True
    for block in vd.decode:
        if res:
            xh = torch.cat([xh, asr_res, vd.F0_conv(F0.unsqueeze(1)), vd.N_conv(N.unsqueeze(1))], 1)
        xh = block(xh, vd.style)
        if block.upsample_type != 'none':
            res = False
    from vhead import FHOP
    y_torch = vd(xh.shape and xh and None or None) if False else None
    # call the head directly with zero noise
    noise = torch.zeros(1, xh.shape[2] * FHOP)
    # replicate SourceFilterHead2.forward through vd.head
    import inspect
    from vhead import SourceFilterHead2
    hd = vd.head
    B, _, nf = xh.shape
    Tt = nf * hd.up
    z = hd.norm0(hd.inp(xh).transpose(1, 2)).transpose(1, 2)
    for b in hd.blocks:
        z = b(z)
    hn = hd.norm1(z.transpose(1, 2))
    o = hd.out(hn).view(1, Tt, hd.C * hd.G + 1)
    o = torch.cat([o, o[:, -1:]], 1)
    gg = (o[..., :-1].reshape(1, Tt + 1, hd.C, hd.G) @ hd.M.T).permute(0, 2, 3, 1)
    gh, gn = torch.exp(gg[:, 0].clamp(max=6)), torch.exp(gg[:, 3].clamp(max=6))
    u = torch.complex(gg[:, 1], gg[:, 2])
    u = u / u.abs().clamp_min(1e-4)
    ee, cyc, uv = None, None, None
    from vhead import sine_bank
    ee, cyc, uv = sine_bank(F0, hd.K, None)
    E = hd.stft(ee)
    Nz0 = hd.stft(noise)
    Nm = [hd.stft(noise * uv * fn(2 * torch.pi * mm * cyc)) for mm in range(1, hd.mod + 1) for fn in (torch.cos, torch.sin)]
    f0raw = torch.cat([F0.repeat_interleave(hd.up, 1), F0[:, -1:]], 1)
    f0f = f0raw.clamp_min(60)
    fr = torch.linspace(0, SR / 2, hd.nb)[None, :, None]
    ratio = fr / f0f[:, None]
    shift = 0.5 * torch.tanh(o[..., -1])[:, None]
    Nz = Nz0 + sum(gg[:, 4 + j] * Nm[j] for j in range(len(Nm)))
    Hh = gh * u * torch.polar(torch.ones_like(ratio), 2 * torch.pi * ratio * shift)
    yh = torch.istft(E * Hh, hd.n_fft, hd.hop, window=hd.window, length=nf * FHOP)
    yn = torch.istft(Nz * gn, hd.n_fft, hd.hop, window=hd.window, length=nf * FHOP)
    if hd.burst:
        gb = torch.exp((hd.out_b(hn).view(1, nf * 4, hd.Gb) @ hd.Mb.T).clamp(max=6)).transpose(1, 2)
        gb = torch.cat([gb, gb[..., -1:]], -1)
        Nb = torch.stft(noise.flip(-1), 150, 75, window=hd.wb, return_complex=True)
        yn = yn + torch.istft(Nb * gb, 150, 75, window=hd.wb, length=nf * FHOP)
    y_t = (yh + yn).squeeze().numpy()
sf.write('runs/michael_zeronoise_torch.wav', y_t, SR)
L = min(len(y_onnx), len(y_t))
d01 = np.abs(y_onnx[:L] - y_t[:L])
print(f'zero-noise parity: max diff {d01.max():.4f}, rms diff {np.sqrt((d01 ** 2).mean()):.5f} | lengths {len(y_onnx)} vs {len(y_t)} | signal rms {np.sqrt((y_t ** 2).mean()):.4f}')

