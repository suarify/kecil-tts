"""Build runs/listen/michael2.html from michael_ablate2/items.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/michael_ablate2/items.json'), encoding='utf-8'))

COLS = [('fixH', 'v0.2.1 (current)', '#5aa86c'),
        ('maskAll', 'gap mask whole band (150 Hz+)', '#2fb8b8'),
        ('maskWide', 'wider mask + whole band', '#d69a2d'),
        ('gainSmooth', 'harmonic gains smoothed', '#8a6cc8'),
        ('combo', 'whole band + smoothed gains', '#4a90d9'),
        ('lowOff', 'no noise below 800 Hz', '#b0752a'),
        ('yati', 'Yati v0.4.1 (reference voice)', '#e05c5c')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    btns = ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS Michael round 2: low band + mask width</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:30%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> round 2 &mdash; the low band and the mask width</h1>
<p class="note">Every round-1 setting sounded watery, so the head's noise/phase switches are not the cause. The next
suspects are the band FIX H never touched: a male voice sits at ~120&nbsp;Hz, so the <b>0&ndash;800&nbsp;Hz</b> band has
wide gaps between few harmonics where noise can smear, and the mask (0.22&nbsp;&times;&nbsp;f0) may simply be too
narrow for his spacing.</p>
<p class="note">&#9654; <b>maskAll</b>: same mask, applied from 150&nbsp;Hz instead of 800. &#9654; <b>maskWide</b>:
0.35&nbsp;&times;&nbsp;f0 mask, whole band. &#9654; <b>gainSmooth</b>: harmonic gains averaged over 5 neighbouring
bins (kills comb notches). &#9654; <b>combo</b>: both. &#9654; <b>lowOff</b>: no noise gains below 800&nbsp;Hz.
&#9654; <b>Yati</b> is the same sentence through the voice you accepted &mdash; use it as the wateriness reference.</p>
<table id="rows">{''.join(rows)}</table>
<p class="note">If even <b>maskAll</b> and <b>combo</b> stay watery next to the Yati reference, the artefact is baked
into Michael's training targets (his clips were voice-converted twice) and the only real fix is retraining him on
cleaner targets.</p>
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
open(os.path.join(ROOT, 'runs/listen/michael2.html'), 'w', encoding='utf-8').write(html)
os.makedirs(os.path.join(ROOT, 'runs/local_v041'), exist_ok=True)
import shutil
shutil.copytree(os.path.join(ROOT, 'runs/listen/michael_ablate2'), os.path.join(ROOT, 'runs/local_v041/michael_ablate2'), dirs_exist_ok=True)
shutil.copy(os.path.join(ROOT, 'runs/listen/michael2.html'), os.path.join(ROOT, 'runs/local_v041/michael2.html'))
print('page written, rows:', len(d['items']))
