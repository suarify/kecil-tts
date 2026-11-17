"""Build runs/listen/clarity.html: teacher vs faithful student with clarity post-processing options."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
d = json.load(open('runs/listen/clarity/items.json', encoding='utf-8'))

TEACH = {'iii': 'vc10/b/c0571.wav', 'call': 'vc15/old/c0184.wav'}
COLS = [('faithful', 'faithful (what training produced)', '#d69a2d'),
        ('air3', '+3 dB air', '#4a90d9'),
        ('air5', '+5 dB air', '#2fb8b8'),
        ('noise60', 'noise floor x0.6', '#8a6cc8'),
        ('air3noise60', 'air3 + noise x0.6', '#5aa86c'),
        ('air5noise60', 'air5 + noise x0.6', '#b0752a'),
        ('harmBoost', 'harmonics +3 dB above 1.5k', '#9a5fb8')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    t = TEACH.get(it['id'])
    btns = (f'<button class="p" style="--c:#00b3b3" data-src="{t}">&#9654;<span class="l"> TEACHER</span></button>' if t else '')
    btns += ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS if k in rel)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

m = d['metrics']
mrows = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td><td>{m[k][3]:.3f}</td></tr>'
                for k, lbl, _ in COLS)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Michael: can post-processing remove the murk?</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1250px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:28%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; can post-processing remove the murk?</h1>
<p class="note">All columns are the same <b>faithful</b> student render (native speed, no clamp, no burst gate).
Only post-processing changes: air shelf (top octave) and noise-floor scaling (keeps the harmonics dominant).
No training, so this is free.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
<p class="note">Pick the clearest one. If none approaches the TEACHER, the remaining murk is the 8 M model itself and
only a longer/bigger training run can move it.</p>
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
open('runs/listen/clarity.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/clarity'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree('runs/listen/clarity', dst)
shutil.copy('runs/listen/clarity.html', 'runs/local_v041/clarity.html')
print('page written')
