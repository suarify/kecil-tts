"""Build runs/listen/v03.html from v03/items.json."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/v03/items.json'), encoding='utf-8'))

COLS = [('v022', 'v0.2.2 published (int8 rebuild, clamp 3-8)', '#e05c5c'),
        ('off', 'v0.3.0 true checkpoints, no clamp', '#d69a2d'),
        ('min2', 'v0.3.0, clamp 2-8', '#4a90d9'),
        ('min3', 'v0.3.0, clamp 3-8', '#5aa86c')]

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
<title>Kecil-TTS Michael: v0.2.2 vs the retrained v0.3.0</title>
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
<h1>Kecil-TTS <span>Michael</span> &mdash; v0.2.2 vs the retrained v0.3.0</h1>
<p class="note">v0.3.0 is exported from the <b>true training checkpoints</b> (decoder +4,000 steps, text side +4,500)
instead of the int8 ONNX rebuild, and the duration clamp is now optional: the real text model's durations are healthy
(max 7 frames, no 400 ms holds), so the clamp only needs to stop one-frame phonemes.</p>
<p class="note">&#9654; <b>no clamp</b>: trust the model's timing. &#9654; <b>clamp 2-8</b>: only lift 25 ms
phonemes to 50 ms. &#9654; <b>clamp 3-8</b>: the current setting. All have the full synthesis recipe
(gap-masked noise, 22 ms burst hangover, noise smoothing, +3 dB air).</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flatness 2-6k</th><th>&gt;4 kHz</th></tr>{mrows}</table>
<p class="note">Pick one and I publish it as v0.3.0.</p>
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
open(os.path.join(ROOT, 'runs/listen/v03.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/v03')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/v03'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/v03.html'), os.path.join(ROOT, 'runs/local_v041/v03.html'))
print('page written, rows:', len(d['items']))
