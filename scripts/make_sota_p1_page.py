"""Build the blind, loudness-matched Phase 1 owner page: 12 sentences x {v0.4.1, P1}.

usage: python scripts/make_sota_p1_page.py --baseline runs/sota/yati_v041 --p1 runs/sota/P1_TAG
Each wav folder must contain <id>.wav files. The page shuffles A/B per row, matches RMS,
lets the owner pick the cleaner one and tick 'watery', then reveals the labels and the tally.
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
ap.add_argument('--baseline', default='runs/sota/yati_v041')
ap.add_argument('--p1', required=True)
ap.add_argument('--out', default='runs/listen/sota_p1.html')
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--seed', type=int, default=11)
A = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
WAVOUT = 'runs/listen/sota_p1'
shutil.rmtree(WAVOUT, ignore_errors=True)
os.makedirs(WAVOUT, exist_ok=True)

rows = list(csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'))
rng = random.Random(A.seed)
# spread across categories: general, codeswitch, numbers (eval set is 40/20/20)
by = {}
for r in rows:
    by.setdefault(r['category'], []).append(r)
sel = []
quota = {'general': 6, 'codeswitch': 3, 'numbers': 3}
for cat, k in quota.items():
    sel += rng.sample(by[cat], k)
rng.shuffle(sel)
sel = sel[:A.n]


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


pairs = []
for r in sel:
    id_ = r['id']
    fa, fb = f'{A.baseline}/{id_}.wav', f'{A.p1}/{id_}.wav'
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print('MISSING', fa, fb)
        continue
    a, sr = sf.read(fa, dtype='float32')
    b, _ = sf.read(fb, dtype='float32')
    ra, rb = rms(a), rms(b)
    target = min(ra, rb)              # avoid clipping when scaling up
    a = a * (target / max(ra, 1e-9))
    b = b * (target / max(rb, 1e-9))
    # blind: which file is A (0 = baseline, 1 = P1)
    flip = rng.random() < 0.5
    wa, wb = (b, a) if flip else (a, b)
    la, lb = ('P1', 'v0.4.1') if flip else ('v0.4.1', 'P1')
    n = min(len(wa), len(wb))
    wa, wb = wa[:n], wb[:n]
    sf.write(f'{WAVOUT}/{id_}_A.wav', wa, sr)
    sf.write(f'{WAVOUT}/{id_}_B.wav', wb, sr)
    pairs.append(dict(id=id_, text=r['text'], cat=r['category'], A=la, B=lb,
                      a=f'{WAVOUT}/{id_}_A.wav'.replace('runs/listen/', ''),
                      b=f'{WAVOUT}/{id_}_B.wav'.replace('runs/listen/', '')))

assert len(pairs) == A.n, f'only {len(pairs)} pairs'

labels_js = '{' + ','.join(f'"{i}":["{p["A"]}","{p["B"]}"]' for i, p in enumerate(pairs)) + '}'

trs = []
for i, p in enumerate(pairs):
    trs.append(
        f'<tr id="r{i}"><td class="tx"><span class="cat">{html.escape(p["cat"])}</span> '
        f'{html.escape(p["text"])}</td>'
        f'<td class="aud"><button class="p" style="--c:#4aa3df" data-src="{p["a"]}">&#9654; A</button>'
        f'<button class="p" style="--c:#d98c4a" data-src="{p["b"]}">&#9654; B</button></td>'
        f'<td><label><input type="radio" name="pick{i}" value="A"> A cleaner</label> '
        f'<label><input type="radio" name="pick{i}" value="B"> B cleaner</label><br>'
        f'<label><input type="checkbox" name="watery{i}" value="A"> A watery</label> '
        f'<label><input type="checkbox" name="watery{i}" value="B"> B watery</label></td>'
        f'<td class="rev" id="rev{i}">?</td></tr>')

page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SOTA Phase 1 - blind A/B</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1250px;margin:0 auto}} h1{{font-family:Sora,Inter,sans-serif;font-size:23px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}} td{{padding:9px 7px;border-top:1px solid #22384a;font-size:13px;vertical-align:middle}}
.tx{{width:40%}} .aud{{white-space:nowrap}} .rev{{font-size:12px;color:#8fd3a0;width:90px}}
.cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 11px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}} button.p.on{{background:var(--c);color:#0d1b26}}
#result{{background:#12242f;border:1px solid #2c4a5e;border-radius:10px;padding:16px;margin-top:18px;font-size:15px}}
#result b{{color:#5fd6d6}}
</style></head><body><main>
<h1>Kecil-TTS <span>SOTA Phase 1</span> &mdash; blind A/B ({len(pairs)} sentences)</h1>
<p class="note">Loudness-matched (equal RMS), A/B shuffled per row. For each sentence play A and B, pick the
<b>cleaner</b> one, and tick <b>watery</b> for any sample with a watery/phasey texture. Labels are hidden until you
press <b>Reveal</b>. <b>Pass = P1 picked &ge; 8/{len(pairs)}.</b></p>
<table>{''.join(trs)}</table>
<p><button class="p" style="--c:#5fd6d6" onclick="reveal()">Reveal labels &amp; score</button></p>
<div id="result" hidden></div>
</main>
<script>
document.addEventListener('click',e=>{{const b=e.target.closest('button.p');if(!b||b.onclick)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));b.classList.add('on');
  const a=new Audio(b.dataset.src);a.onended=()=>b.classList.remove('on');a.play();}});
const LABELS={labels_js};
const NT={len(pairs)};
function reveal(){{
  let p1=0, wateryV=0, wateryP=0, done=0;
  for(let i=0;i<NT;i++){{
    const r=document.querySelector(`input[name="pick${{i}}"]:checked`);
    const lab=LABELS[i];
    document.getElementById('rev'+i).textContent='A='+lab[0]+' B='+lab[1]+(r?' | '+r.value:'');
    if(!r)continue; done++;
    const chosenIsP1=(r.value==='A')?lab[0]==='P1':lab[1]==='P1';
    if(chosenIsP1)p1++;
    const w=document.querySelectorAll(`input[name="watery${{i}}"]:checked`);
    w.forEach(x=>{{if((x.value==='A'&&lab[0]==='P1')||(x.value==='B'&&lab[1]==='P1'))wateryP++;
                   else wateryV++;}});
  }}
  const d=document.getElementById('result');d.hidden=false;
  d.innerHTML=`P1 chosen <b>${{p1}}/${{done}}</b> (need &ge;8/${{NT}}) &nbsp;|&nbsp; watery ticks: P1 <b>${{wateryP}}</b>, v0.4.1 <b>${{wateryV}}</b>`
    +`<br>Pass = <b>${{p1>=8&&done===NT?'YES':'NO'}}</b>`;
}}
</script></body></html>
'''
open(A.out, 'w', encoding='utf-8').write(page)
print(f'wrote {A.out} with {len(pairs)} pairs -> {WAVOUT}')
