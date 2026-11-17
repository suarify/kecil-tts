"""Build runs/listen/grid.html from grid/items.json."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/grid/items.json'), encoding='utf-8'))

COLS = [
    ('ref', 'clamp 2-8 + air3 @6.5k (current best)', '#e05c5c'),
    ('air2', 'air +2 dB', '#d69a2d'),
    ('air4', 'air +4 dB', '#5aa86c'),
    ('air5', 'air +5 dB', '#2fb8b8'),
    ('fc6k', 'air3 but shelf at 6.0 kHz', '#4a90d9'),
    ('fc7k', 'air3 but shelf at 7.0 kHz', '#8a6cc8'),
    ('clamp27', 'clamp 2-7 frames', '#b0752a'),
    ('clamp29', 'clamp 2-9 frames', '#9a5fb8'),
    ('clamp38', 'clamp 3-8 frames', '#3f9b6d'),
    ('trem', 'ref + slow amplitude sway', '#c8553d'),
]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    btns = ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

m = d['metrics']
mrows = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td><td>{m[k][3]:.3f}</td></tr>'
                for k, lbl, _ in COLS)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS Michael: grid around clamp + air</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1240px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:24%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; tuning around <b>clamp + air</b></h1>
<p class="note">Current best = durations clamped to 50&ndash;200 ms + 3 dB high shelf above 6.5 kHz. This grid nudges
each part: how much air, where the shelf starts, how tightly to clamp the timing, and one micro-variation of the
noise floor. Note <b>clamp 3&ndash;8</b> already measures cleaner (periodicity 0.635 vs 0.608) &mdash; it stops any
phoneme from being shorter than 75 ms, which removes the remaining staccato.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over the three sentences</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
<p class="note">Two answers help most: the best <b>air amount</b> (2/3/4/5 dB) and the best <b>clamp range</b>
(2-7 / 2-8 / 2-9 / 3-8). Then I bake that exact pair into the export.</p>
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
open(os.path.join(ROOT, 'runs/listen/grid.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/grid')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/grid'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/grid.html'), os.path.join(ROOT, 'runs/local_v041/grid.html'))
print('page written, rows:', len(d['items']))
