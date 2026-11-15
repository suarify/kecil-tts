"""A1: where does the damp sound come from? Build runs/listen/damp_where.html.

Eight held-out rows (the first 60 rows of the kmb-12 shard are the held set), four versions each, open labels:
  1 the Chatterbox clip the model learned from   (runs/k12/cbwav/<id>.wav)
  2 the v0.2 decoder fed that clip's measured inputs (t_en expanded by pred_dur, measured F0/N; seed 0)
  3 v0.2 as shipped, from text                    (the int8 ONNX, python package)
  4 Kokoro af_bella reading the same Malay phonemes (the row's ps, local CPU KModel)

The owner ticks EVERY version that sounds damp. Result JSON saved in the textarea.

    PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe scripts/listen_damp_where.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from student import alignment  # noqa: E402
from vhead import VDecoder  # noqa: E402

N_PICK = 8
SHARD = os.path.join(ROOT, 'runs/k12_rows/rows/shard_0000.pt')
DEC = os.path.join(ROOT, 'runs/k12/ckpt/dec_last.pt')
ONNX = os.path.join(ROOT, 'runs/k12/ms_A_int8_gpu.onnx')
CBWAV = os.path.join(ROOT, 'runs/k12/cbwav')
OUT = os.path.join(ROOT, 'runs/listen/damp_where')
SR = 24000
torch.set_num_threads(4)

NAMES = ['1: Chatterbox clip (the training target)',
         '2: v0.2 decoder, fed this clip\'s measured inputs',
         '3: v0.2 as shipped, from text',
         '4: Kokoro af_bella reading the Malay phonemes']

# ---------------- load rows; the first 60 are the held set ----------------
rows = torch.load(SHARD, map_location='cpu', weights_only=False)
held = rows[:60]
pick = [held[i] for i in np.linspace(0, len(held) - 1, N_PICK).round().astype(int)]
os.makedirs(OUT, exist_ok=True)

# ---------------- system 2: the decoder with measured inputs ----------------
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(DEC)
dec_audios = {}
with torch.no_grad():
    for r in pick:
        aln = alignment(r['pred_dur'])
        Fr = aln.shape[1]
        torch.manual_seed(0)
        y = vd((r['t_en'].float() @ aln)[None], r['F0'][None, :2 * Fr], r['N'][None, :2 * Fr]).squeeze().numpy()
        dec_audios[r['id']] = y.astype(np.float32)
        print('decoder', r['id'], round(len(y) / SR, 2), 's', flush=True)
del vd

# ---------------- system 3: v0.2 from text (shipped ONNX) ----------------
from suaradee import Suaradee  # noqa: E402

tts = Suaradee(ONNX)
v02_audios = {r['id']: tts(r['text']) for r in pick}
print('v0.2 from text done', flush=True)

# ---------------- system 4: Kokoro af_bella on the same phonemes (local CPU) ----------------
import types  # noqa: E402


def _stub(name):
    m = types.ModuleType(name)

    def _ga(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return type(attr, (), {})
    m.__getattr__ = _ga
    return m


sys.modules.setdefault('misaki', _stub('misaki'))
for sub in ('en', 'espeak'):
    sys.modules.setdefault('misaki.' + sub, _stub('misaki.' + sub))
    setattr(sys.modules['misaki'], sub, sys.modules['misaki.' + sub])
from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

km = KModel(repo_id='hexgrad/Kokoro-82M').eval()
bella = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_bella.pt'), weights_only=True)
kokoro_audios = {}
with torch.no_grad():
    for r in pick:
        audio = km(r['ps'][:510], bella[len(r['ps'][:510]) - 1], 1.0).float().numpy()
        kokoro_audios[r['id']] = audio.astype(np.float32)
        print('kokoro', r['id'], round(len(audio) / SR, 2), 's', flush=True)

# ---------------- write files + page data ----------------
items = []
for n, r in enumerate(pick):
    iid = r['id']
    files = [f'damp_where/{iid}_1_chatterbox.wav', f'damp_where/{iid}_2_decoder.wav',
             f'damp_where/{iid}_3_v02.wav', f'damp_where/{iid}_4_kokoro.wav']
    sf.write(os.path.join(ROOT, 'runs', 'listen', files[0]), sf.read(os.path.join(CBWAV, iid + '.wav'), dtype='float32')[0], SR)
    sf.write(os.path.join(ROOT, 'runs', 'listen', files[1]), dec_audios[iid], SR)
    sf.write(os.path.join(ROOT, 'runs', 'listen', files[2]), v02_audios[iid], SR)
    sf.write(os.path.join(ROOT, 'runs', 'listen', files[3]), kokoro_audios[iid], SR)
    items.append(dict(id=iid, text=r['text'], files=files))
    print('wrote', iid, flush=True)

html = '''<!doctype html><html><head><meta charset="utf-8"><title>Where is the damp sound?</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#0b6e4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#5fd3a8}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1080px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:6px 0}.row{border-top:1px solid var(--line);padding:12px 0}
.t{font-weight:600}.g{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}.g div{flex:1;min-width:240px}audio{width:100%;height:32px}
label{font-size:13px;color:var(--mut);display:block;margin-bottom:2px;cursor:pointer}input{accent-color:var(--acc)}
#sum{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);font-weight:600}
textarea{width:100%;height:54px;background:var(--bg);color:var(--fg);border:1px solid var(--line);font:12px monospace}
</style></head><body><h1>Where is the damp sound?</h1>
<p>Same sentence, four versions. Use headphones. Tick EVERY version that sounds damp, muffled, wet or phasey - not the one you like best. Version 1 is the clip the model was trained on; 2 is our decoder fed that clip's own measured timing, pitch and loudness; 3 is v0.2 as you ship it; 4 is stock Kokoro Bella reading the Malay sounds.</p>
<div id="sum"></div><div id="l"></div><p>Result to paste back:</p><textarea id="res" readonly></textarea>
<script>
const D=__DATA__,N=__NAMES__,K='dampWherePicks';let P={};try{P=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const l=document.getElementById('l');
D.forEach((it,i)=>{const r=document.createElement('div');r.className='row';
 r.innerHTML='<div class="t">'+(i+1)+'. '+it.text+'</div><div class="g">'+N.map((n,j)=>'<div><label><input type="checkbox" name="p'+i+'" value="'+j+'"> '+n+'</label><audio controls preload="none" src="'+it.files[j]+'"></audio></div>').join('')+'</div>';
 l.appendChild(r);
 r.querySelectorAll('input').forEach(x=>{const cur=P[it.id]||[];if(cur.includes(+x.value))x.checked=true;
  x.onchange=()=>{const s=new Set(P[it.id]||[]);x.checked?s.add(+x.value):s.delete(+x.value);P[it.id]=[...s].sort((a,b)=>a-b);
   try{localStorage.setItem(K,JSON.stringify(P))}catch(e){}show()}})});
function show(){const n={};Object.values(P).forEach(v=>v.forEach(j=>n[j]=(n[j]||0)+1));
 const s=N.map((x,j)=>j+1+' '+(n[j]||0)).join(' | ')+' | rated '+Object.keys(P).length+'/'+D.length;
 document.getElementById('sum').textContent=s;document.getElementById('res').value=s+'\\n'+JSON.stringify(P)}
show();
</script></body></html>'''
html = html.replace('__DATA__', json.dumps(items, ensure_ascii=False)).replace('__NAMES__', json.dumps(NAMES))
out = os.path.join(ROOT, 'runs/listen/damp_where.html')
open(out, 'w', encoding='utf-8').write(html)
print('wrote', out, len(items), 'sentences x 4 versions')
