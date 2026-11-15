"""A1 follow-up: can repairing the text side's predicted F0 contour fix the damp sound, no training?

The owner's damp_where verdict: version 3 (v0.2 from text) is damp, version 2 (decoder + measured
inputs) is fine -> the damp lives in the predicted F0/N. f0n_diag2 found 17% voiced-frame dropouts
(F0 < 50 Hz -> the decoder makes unvoiced noise) and a 227 Hz span vs the clips' 69 Hz.

Variants (pure functions of the predicted contour, no measured-clip info):
  A v0.2 as shipped          - predicted F0/N unchanged
  B gaps filled              - dips below 50 Hz that sit between voiced frames are interpolated
  C B + smoothing            - 7-frame median filter on F0, clamp 70..350 Hz, 5-frame smooth on N
  D C + span cap             - contour rescaled so its 10-90th percentile span is at most 80 Hz

Each sentence is rendered 4x with the same noise seed, so only the contour differs. Writes
runs/listen/f0_fix/*.wav and runs/listen/f0_fix.html (radio: least damp).

    PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe scripts/f0_fix.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent, alignment  # noqa: E402
from vhead import VDecoder  # noqa: E402

N_PICK = 8
SHARD = os.path.join(ROOT, 'runs/k12_rows/rows/shard_0000.pt')
DEC = os.path.join(ROOT, 'runs/k12/ckpt/dec_last.pt')
TEXT = os.path.join(ROOT, 'runs/k12/ckpt/textA_last.pt')
OUT = os.path.join(ROOT, 'runs/listen/f0_fix')
SR = 24000
torch.set_num_threads(4)

ts = TextStudent('s+mlp').eval()
ts.load_state_dict(torch.load(TEXT, map_location='cpu')['model'])
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(DEC)
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
rows = torch.load(SHARD, map_location='cpu', weights_only=False)
held = rows[:60]
pick = [held[i] for i in np.linspace(0, len(held) - 1, N_PICK).round().astype(int)]
os.makedirs(OUT, exist_ok=True)


def text_side_all(ids):
    """asr features, alignment, and the predicted F0/N (same maths as the shipped text side)."""
    t, p, s = ts, ts.predictor, ts.style
    d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
    x, T = d_en, ids.shape[1]
    sx = s[:, :, None].expand(-1, -1, T)
    x = torch.cat([x, sx], 1)
    for b in p.text_encoder.lstms:
        x = (b(x.transpose(-1, -2))[0].transpose(-1, -2) if isinstance(b, nn.LSTM)
             else torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1))
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
    F0, Nn = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    return asr @ aln, F0[0].numpy(), Nn[0].numpy(), n.numpy()


def fill_gaps(f0, max_gap=15):
    """Frames below 50 Hz that sit between voiced stretches (within max_gap frames) get interpolated."""
    out = f0.copy()
    voiced = f0 >= 50
    n = len(out)
    for i in np.where(~voiced)[0]:
        left = voiced[:i][::-1]
        li = np.argmax(left) if left.any() else -1
        right = voiced[i + 1:]
        ri = np.argmax(right) if right.any() else -1
        if li >= 0 and ri >= 0 and li + ri + 1 <= max_gap:
            a, b = out[i - li - 1], out[i + ri + 1]
            out[i] = a + (b - a) * (li + 1) / (li + ri + 2)
    return out


def repair(f0, n_contour, level):
    """level 0 shipped; 1 fill; 2 fill+smooth; 3 fill+smooth+span cap."""
    if level >= 1:
        f0 = fill_gaps(f0)
    if level >= 2:
        k = 7
        pad = k // 2
        m = np.pad(f0, pad, mode='edge')
        f0 = np.array([np.median(m[i:i + k]) for i in range(len(f0))])
        f0 = np.clip(f0, 70, 350)
        w = 5
        pad = w // 2
        m = np.pad(n_contour, pad, mode='edge')
        n_contour = np.convolve(m, np.ones(w) / w, 'valid')[:len(n_contour)]
    if level >= 3:
        v = f0 >= 50
        if v.sum() > 4:
            lo, hi = np.percentile(f0[v], 10), np.percentile(f0[v], 90)
            med = np.median(f0[v])
            if hi - lo > 80:
                semis = 12 * np.log2(np.clip(f0, 1, None) / med)
                cap = 12 * np.log2((med * 2 ** (40 / 12)) / med)     # 80 Hz span -> +-40 semitone-ish? keep simple: clip to +-4 st
                semis = np.clip(semis, -4, 4)
                f0 = np.where(v, med * 2 ** (semis / 12), f0)
    return f0, n_contour


VARIANTS = ['A: v0.2 as shipped', 'B: pitch gaps filled', 'C: B + smoothing', 'D: C + span cap']
items = []
with torch.no_grad():
    for r in pick:
        iid = r['id']
        ids = torch.tensor([g2p.tokens(g2p(normalize(r['text'])))], dtype=torch.long)
        asr, F0p, Np, dur = text_side_all(ids)
        Fr = int(dur.sum())
        files = {}
        for level in range(4):
            f0, nc = repair(F0p.copy(), Np.copy(), level)
            f0 = np.clip(f0, 0, None)
            torch.manual_seed(0)
            y = vd(asr, torch.from_numpy(f0).float()[None], torch.from_numpy(nc).float()[None]).squeeze().numpy()
            name = f'f0_fix/{iid}_{chr(65 + level)}.wav'
            sf.write(os.path.join(ROOT, 'runs/listen', name), np.clip(y, -1, 1).astype(np.float32), SR)
            files[chr(65 + level)] = name
            print(iid, chr(65 + level), round(len(y) / SR, 2), 's', flush=True)
        items.append(dict(id=iid, text=r['text'], files=files))

html = '''<!doctype html><html><head><meta charset="utf-8"><title>Fixing the damp: text-side pitch repair</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#0b6e4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#5fd3a8}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1080px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:6px 0}.row{border-top:1px solid var(--line);padding:12px 0}
.t{font-weight:600}.g{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}.g div{flex:1;min-width:230px}audio{width:100%;height:32px}
label{font-size:13px;color:var(--mut);display:block;margin-bottom:2px;cursor:pointer}input{accent-color:var(--acc)}
#sum{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);font-weight:600}
textarea{width:100%;height:54px;background:var(--bg);color:var(--fg);border:1px solid var(--line);font:12px monospace}
</style></head><body><h1>Fixing the damp sound: repairing the predicted pitch</h1>
<p>Same sentence, four renders of the same model - only the pitch contour fed to the decoder differs, the noise is identical. Headphones. For each sentence pick the ONE that sounds least damp, wet or phasey (or "none is OK" if none is acceptable). It is fine if several sound the same; that is a useful answer too.</p>
<div id="sum"></div><div id="l"></div><p>Result to paste back:</p><textarea id="res" readonly></textarea>
<script>
const D=__DATA__,N=__NAMES__,K='f0FixPicks';let P={};try{P=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const l=document.getElementById('l');
D.forEach((it,i)=>{const r=document.createElement('div');r.className='row';
 r.innerHTML='<div class="t">'+(i+1)+'. '+it.text+'</div><div class="g">'+N.map((n,j)=>'<div><label><input type="radio" name="p'+i+'" value="'+j+'"'+(P[it.id]===j?' checked':'')+'> '+n+'</label><audio controls preload="none" src="'+it.files['ABCD'[j]]+'"></audio></div>').join('')+'</div><label><input type="radio" name="p'+i+'" value="none"'+(P[it.id]==='none'?' checked':'')+'> none is OK</label>';
 l.appendChild(r);
 r.querySelectorAll('input').forEach(x=>{x.onchange=()=>{P[it.id]=x.value==='none'?'none':+x.value;
  try{localStorage.setItem(K,JSON.stringify(P))}catch(e){}show()}})});
function show(){const n={};Object.values(P).forEach(v=>n[v]=(n[v]||0)+1);
 const s=N.map((x,j)=>'ABCD'[j]+' '+(n[j]||0)).join(' | ')+' | none '+(n.none||0)+' | rated '+Object.keys(P).length+'/'+D.length;
 document.getElementById('sum').textContent=s;document.getElementById('res').value=s+'\\n'+JSON.stringify(P)}
show();
</script></body></html>'''
html = html.replace('__DATA__', json.dumps(items, ensure_ascii=False)).replace('__NAMES__', json.dumps(VARIANTS))
out = os.path.join(ROOT, 'runs/listen/f0_fix.html')
open(out, 'w', encoding='utf-8').write(html)
print('wrote', out, len(items), 'sentences x', len(VARIANTS), 'variants')
