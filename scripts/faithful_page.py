"""Build runs/listen/faithful.html: teacher clip vs student faithful / processed variants."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
d = json.load(open('runs/listen/faithful/items.json', encoding='utf-8'))

TEACH = {'iii': 'vc10/b/c0571.wav'}
COLS = [('faithful', 'student, faithful (as trained)', '#d69a2d'),
        ('faithfulAir', 'faithful + air', '#4a90d9'),
        ('fixH', 'our v0.3.0 recipe (clamp 3-8)', '#e05c5c'),
        ('fixHNoClamp', 'our recipe, clamp OFF', '#5aa86c'),
        ('fixHFree', 'our recipe, bursts ungated', '#8a6cc8')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    t = TEACH.get(it['id'])
    btns = (f'<button class="p" style="--c:#2fb8b8" data-src="{t}">&#9654;<span class="l"> TEACHER clip</span></button>' if t else '')
    btns += ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS if k in rel)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

m = d['metrics']
mrows = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td></tr>'
                for k, lbl, _ in COLS)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Michael student: is our processing the problem?</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1200px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:30%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; is our export processing the problem?</h1>
<p class="note"><b>faithful</b> is exactly what the training produced: no gap mask, no burst gate, no gain smoothing,
no air shelf, no duration clamp. The other columns add pieces of our v0.3.0 recipe one at a time. The TEACHER clip is
the converted Chatterbox audio the student was trained on.</p>
<p class="note">Watch the durations: <b>faithful</b> is 4.58 s on sentence 1 while <b>our recipe</b> is 5.85 s &mdash;
the clamp is what made the voice slower.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th></tr>{mrows}</table>
<p class="note">Tell me which student column sounds least phasey/watery and whether the speed is right, and I bake
exactly that recipe.</p>
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
open('runs/listen/faithful.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/faithful'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree('runs/listen/faithful', dst)
# teacher clips are referenced through the existing vc10 folder on the Space copy
shutil.copy('runs/listen/faithful.html', 'runs/local_v041/faithful.html')
print('page written')
