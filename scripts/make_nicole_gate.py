"""Nicole gate page: old Nicole v0.1 / new Nicole v0.5 / Yati v0.5, blind loudness-matched."""
import csv
import html
import os
import random
import shutil

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
WAVOUT = 'runs/listen/nicole_v05_gate'
shutil.rmtree(WAVOUT, ignore_errors=True)
os.makedirs(WAVOUT, exist_ok=True)

# regenerate old Nicole v0.1 wavs on the 80-set if missing
import subprocess, sys
old_dir = 'runs/sota/nicole_v010'
if not os.path.isdir(old_dir) or len(os.listdir(old_dir)) < 80:
    print('generating old Nicole wavs...')
    subprocess.run([sys.executable, 'scripts/render_sents.py'], capture_output=True) if os.path.exists('scripts/render_sents.py') else None
    # render directly
    sys.path.insert(0, 'g2p')
    import onnxruntime as ort
    from ms_g2p import G2P
    from ms_norm import normalize
    g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
    sess = ort.InferenceSession('runs/nicole_v010_repo/onnx/model_quantized.onnx', providers=['CPUExecutionProvider'])
    os.makedirs(old_dir, exist_ok=True)
    for r in csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'):
        ids = np.array([g2p.tokens(g2p(normalize(r['text'])))], dtype=np.int64)
        y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
        sf.write(f"{old_dir}/{r['id']}.wav", y, 24000)
    print('old Nicole rendered')

LABELS = ['Yati v0.5', 'Nicole v0.1 (old)', 'Nicole v0.5 (new)']
FOLDERS = ['runs/sota/p3_dec_gan', old_dir, 'runs/sota/nicole_v05_gan']
rows = list(csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'))
rng = random.Random(909)
by = {}
for r in rows:
    by.setdefault(r['category'], []).append(r)
sel = rng.sample(by['general'], 6) + rng.sample(by['codeswitch'], 3) + rng.sample(by['numbers'], 3)
rng.shuffle(sel)

pairs = []
for r in sel:
    id_ = r['id']
    wavs = []
    ok = True
    for f in FOLDERS:
        p = f'{f}/{id_}.wav'
        if not os.path.exists(p):
            ok = False; break
        wavs.append(sf.read(p, dtype='float32')[0])
    if not ok:
        print('MISSING', id_); continue
    rms = [float(np.sqrt(np.mean(w.astype(np.float64)**2))) for w in wavs]
    target = min(rms)
    wavs = [w * (target / max(x, 1e-9)) for w, x in zip(wavs, rms)]
    n = min(len(w) for w in wavs)
    order = list(range(3)); rng.shuffle(order)
    for j, src in enumerate(order):
        sf.write(f'{WAVOUT}/{id_}_{j}.wav', wavs[src][:n], 24000)
    pairs.append(dict(id=id_, text=r['text'], cat=r['category'], src=order,
                      files=[f'nicole_v05_gate/{id_}_{j}.wav' for j in range(3)]))

trs = []
for i, p in enumerate(pairs):
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{f}">&#9654; {x}</button>'
                   for c, f, x in zip(('#4aa3df','#d98c4a','#5aa86c'), p['files'], 'ABC'))
    opts = ''.join(f'<label><input type="radio" name="pick{i}" value="{x}"> {x}</label> ' for x in 'ABC')
    wets = ''.join(f'<label><input type="checkbox" name="wet{i}" value="{x}"> {x} watery</label> ' for x in 'ABC')
    trs.append(f'<tr><td class="tx"><span class="cat">{html.escape(p["cat"])}</span> {html.escape(p["text"])}</td>'
               f'<td class="aud">{btns}</td><td>{opts}<br>{wets}</td><td class="rev" id="rev{i}">?</td></tr>')
labels_js = '{' + ','.join(f'"{i}":[{",".join(chr(39)+LABELS[s]+chr(39) for s in p["src"])}]' for i,p in enumerate(pairs)) + '}'
page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Nicole v0.5 gate</title>
<style>body{{font:15px/1.5 'Plus Jakarta Sans',system-ui,sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1350px;margin:0 auto}}h1{{font:800 23px Outfit,sans-serif}}h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf}}table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 7px;border-top:1px solid #22384a;font-size:13px;vertical-align:middle}}
.tx{{width:38%}}.aud{{white-space:nowrap}}.rev{{font-size:12px;color:#8fd3a0;width:150px}}
.cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 11px;font:600 12px Outfit,sans-serif;cursor:pointer;margin:2px}}
button.p.on{{background:var(--c);color:#0d1b26}}
#result{{background:#12242f;border:1px solid #2c4a5e;border-radius:10px;padding:16px;margin-top:18px}}#result b{{color:#5fd6d6}}</style></head>
<body><main><h1>Kecil-TTS <span>Nicole v0.5 gate</span> &mdash; blind 3-way</h1>
<p class="note">Loudness-matched, shuffled. Pick the <b>best</b>, tick <b>watery</b>, then Reveal. Pass = New Nicole picked &ge; 8/{len(pairs)} (or old Nicole if you prefer it) and watery &le; 2/{len(pairs)}.</p>
<table>{''.join(trs)}</table><p><button class="p" style="--c:#5fd6d6" onclick="reveal()">Reveal</button></p><div id="result" hidden></div></main>
<script>document.addEventListener('click',e=>{{const b=e.target.closest('button.p');if(!b||b.onclick)return;
document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));b.classList.add('on');
const a=new Audio(b.dataset.src);a.onended=()=>b.classList.remove('on');a.play();}});
const L={labels_js};const NT={len(pairs)};
function reveal(){{let best=0,wet=0,done=0;
for(let i=0;i<NT;i++){{const r=document.querySelector(`input[name="pick${{i}}"]:checked`);const lab=L[i];
document.getElementById('rev'+i).textContent='A='+lab[0]+' B='+lab[1]+' C='+lab[2];
if(!r)continue;done++;if(lab['ABC'.indexOf(r.value)].indexOf('new')>=0)best++;
document.querySelectorAll(`input[name="wet${{i}}"]:checked`).forEach(x=>{{if(lab['ABC'.indexOf(x.value)].indexOf('new')>=0)wet++;}});
}}
const d=document.getElementById('result');d.hidden=false;
d.innerHTML=`NEW Nicole best <b>${{best}}/${{done}}</b> | watery <b>${{wet}}</b> | (compare with old Nicole / Yati in the revealed labels)`;}}</script></body></html>'''
open('runs/listen/nicole_v05_gate.html', 'w', encoding='utf-8').write(page)
print('wrote page', len(pairs), 'pairs')
