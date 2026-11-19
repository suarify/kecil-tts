"""Gate 3: blind 3-way page {v0.4.1, P2, P3}, 12 sentences, loudness-matched, shuffled.

usage: python scripts/make_gate3_page.py --p3 FOLDER --tag TAG
P2 wavs come from runs/sota/p2 (regenerate if missing). v0.4.1 from runs/sota/yati_v041.
"""
import argparse
import csv
import html
import os
import random
import shutil

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument('--p3', required=True, help='folder of 80 P3 wavs, e.g. runs/sota/p3_p3_export')
ap.add_argument('--tag', required=True, help='P3 tag for the label')
ap.add_argument('--base', default='runs/sota/yati_v041')
ap.add_argument('--p2', default='runs/sota/p2')
ap.add_argument('--out', default='runs/listen/sota_p3.html')
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--seed', type=int, default=23)
A = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
WAVOUT = os.path.splitext(A.out)[0].replace('\\', '/')
shutil.rmtree(WAVOUT, ignore_errors=True)
os.makedirs(WAVOUT, exist_ok=True)

rows = list(csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'))
rng = random.Random(A.seed)
by = {}
for r in rows:
    by.setdefault(r['category'], []).append(r)
sel = rng.sample(by['general'], 6) + rng.sample(by['codeswitch'], 3) + rng.sample(by['numbers'], 3)
rng.shuffle(sel)

LABELS = ['v0.4.1', 'P2', A.tag]
FOLDERS = [A.base, A.p2, A.p3]
pairs = []
for r in sel:
    id_ = r['id']
    wavs = []
    ok = True
    for f in FOLDERS:
        p = f'{f}/{id_}.wav'
        if not os.path.exists(p):
            ok = False
            break
        wavs.append(sf.read(p, dtype='float32')[0])
    if not ok:
        print('MISSING', id_)
        continue
    rms = [float(np.sqrt(np.mean(w.astype(np.float64) ** 2))) for w in wavs]
    target = min(rms)
    wavs = [w * (target / max(x, 1e-9)) for w, x in zip(wavs, rms)]
    n = min(len(w) for w in wavs)
    order = list(range(3)); rng.shuffle(order)
    for j, src in enumerate(order):
        sf.write(f'{WAVOUT}/{id_}_{j}.wav', wavs[src][:n], 24000)
    pairs.append(dict(id=id_, text=r['text'], cat=r['category'],
                      src=order, files=[f'{os.path.basename(WAVOUT)}/{id_}_{j}.wav' for j in range(3)]))

assert len(pairs) == A.n, f'only {len(pairs)} pairs'
trs = []
for i, p in enumerate(pairs):
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{f}">&#9654; {x}</button>'
                   for c, f, x in zip(('#4aa3df', '#d98c4a', '#5aa86c'), p['files'], 'ABC'))
    opts = ''.join(f'<label><input type="radio" name="pick{i}" value="{x}"> {x} best</label> '
                   for x in 'ABC')
    wets = ''.join(f'<label><input type="checkbox" name="wet{i}" value="{x}"> {x} watery</label> '
                   for x in 'ABC')
    trs.append(f'<tr id="r{i}"><td class="tx"><span class="cat">{html.escape(p["cat"])}</span> '
               f'{html.escape(p["text"])}</td><td class="aud">{btns}</td><td>{opts}<br>{wets}</td>'
               f'<td class="rev" id="rev{i}">?</td></tr>')

labels_js = '{' + ','.join(f'"{i}":[{",".join(chr(39)+LABELS[s]+chr(39) for s in p["src"])}]'
                          for i, p in enumerate(pairs)) + '}'
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Gate 3 - blind 3-way</title>
<style>
body{{font:15px/1.5 'Plus Jakarta Sans',system-ui,sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1350px;margin:0 auto}} h1{{font:800 23px Outfit,sans-serif;margin:0}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}} table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 7px;border-top:1px solid #22384a;font-size:13px;vertical-align:middle}}
.tx{{width:38%}} .aud{{white-space:nowrap}} .rev{{font-size:12px;color:#8fd3a0;width:150px}}
.cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 11px;font:600 12px Outfit,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}} button.p.on{{background:var(--c);color:#0d1b26}}
#result{{background:#12242f;border:1px solid #2c4a5e;border-radius:10px;padding:16px;margin-top:18px;font-size:15px}}
#result b{{color:#5fd6d6}}
</style></head><body><main>
<h1>Kecil-TTS <span>Gate 3</span> &mdash; blind 3-way ({len(pairs)} sentences)</h1>
<p class="note">Loudness-matched, order shuffled per row. Play A/B/C, pick the <b>best</b> one and tick <b>watery</b> for any with a watery/phasey texture. Labels reveal after pressing Reveal. <b>Pass = {A.tag} picked &ge; 8/{len(pairs)} AND watery &le; 2/{len(pairs)}.</b></p>
<table>{''.join(trs)}</table>
<p><button class="p" style="--c:#5fd6d6" onclick="reveal()">Reveal labels &amp; score</button></p>
<div id="result" hidden></div>
</main><script>
document.addEventListener('click',e=>{{const b=e.target.closest('button.p');if(!b||b.onclick)return;
 document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 const a=new Audio(b.dataset.src);a.onended=()=>b.classList.remove('on');a.play();}});
const L={labels_js}; const TAG='{A.tag}'; const NT={len(pairs)};
function reveal(){{let best=0,wet=0,done=0;
 for(let i=0;i<NT;i++){{const r=document.querySelector(`input[name="pick${{i}}"]:checked`);
  const lab=L[i];let s='';for(let j=0;j<3;j++)s+='ABC'[j]+'='+lab[j]+' ';
  document.getElementById('rev'+i).textContent=s+(r?' | '+r.value:'');
  if(!r)continue;done++;
  if(lab['ABC'.indexOf(r.value)]===TAG)best++;
  document.querySelectorAll(`input[name="wet${{i}}"]:checked`).forEach(x=>{{
   if(lab['ABC'.indexOf(x.value)]===TAG)wet++;}});
 }}
 const d=document.getElementById('result');d.hidden=false;
 d.innerHTML=`{A.tag} best <b>${{best}}/${{done}}</b> (need &ge;8/${{NT}}) &nbsp;|&nbsp; {A.tag} watery <b>${{wet}}</b> (need &le;2/${{NT}})`
  +`<br>Pass = <b>${{best>=8&&wet<=2&&done===NT?'YES':'NO'}}</b>`;
}}
</script></body></html>
'''
open(A.out, 'w', encoding='utf-8').write(page)
print('wrote', A.out, 'with', len(pairs), 'pairs ->', WAVOUT)
