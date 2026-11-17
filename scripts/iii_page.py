"""Build runs/listen/iii.html from iii/items.json."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/iii/items.json'), encoding='utf-8'))

COLS = [('base', 'published (hang7)', '#e05c5c'),
        ('smooth3', 'previous pick: smooth both gains', '#d69a2d'),
        ('smoothGn', 'smooth NOISE gains only', '#5aa86c'),
        ('smooth121', 'light [1,2,1] smoothing', '#2fb8b8'),
        ('vowel', 'more harmonic solidity + noise smooth', '#4a90d9'),
        ('softGn', 'soft bursts + noise smooth', '#8a6cc8')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    btns = ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

m = d['metrics']
mrows = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td></tr>' for k, lbl, _ in COLS)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS Michael: the III vowels + smoothing without distortion</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:36%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; the "III" vowels and smoothing without distortion</h1>
<p class="note">The front end reads <b>III</b> as three stressed <b>&#712;I</b> vowels, so it is a long stressed vowel
&mdash; exactly where the noise floor shows as watery/robotic. Your previous pick (smooth both gains) helped but
smears the harmonic transients (the mild distortion). These variants smooth <b>only the noise gains</b>, or smooth
more lightly, so the harmonics &mdash; and their consonants &mdash; stay untouched.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over the three sentences</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flatness 2-6k</th><th>&gt;4 kHz</th></tr>{mrows}</table>
<p class="note">Pick the one where III is least watery AND the rest of the sentence has no smeared/distorted edge.
If the III vowels still fail in every variant, the next step is a vowel-specific test (noise fully removed only on
long vowels) or accepting that this syllable exposes the training data.</p>
</main>
<script>
let cur=null;
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  if(cur&&cur!==b){{cur.pause&&cur.pause();cur.classList.remove('on');}}
  if(cur===b&&cur.pause===undefined)return;
  cur=b; b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play(); cur=a;
}});
</script></body></html>"""
open(os.path.join(ROOT, 'runs/listen/iii.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/iii')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/iii'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/iii.html'), os.path.join(ROOT, 'runs/local_v041/iii.html'))
print('page written, rows:', len(d['items']))
