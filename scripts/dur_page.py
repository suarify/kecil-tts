"""Build runs/listen/dur.html from dur/items.json (duration-based de-robotising)."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/dur/items.json'), encoding='utf-8'))

COLS = [('base', 'published (hang7 + noise-smooth)', '#e05c5c'),
        ('clamp', 'durations clamped to 50-200 ms', '#5aa86c'),
        ('med3', 'median-filtered durations', '#2fb8b8'),
        ('clampair', 'clamp + 3 dB air', '#4a90d9'),
        ('trem', 'clamp + slow amplitude sway', '#d69a2d'),
        ('vib03', 'clamp + gentle 0.3% vibrato', '#8a6cc8')]

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
<title>Kecil-TTS Michael: fix the timing, not the spectrum</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:34%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
code{{background:#12242f;border-radius:4px;padding:1px 5px;font-size:12px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; fix the timing, not the spectrum</h1>
<p class="note">Reason for this round: every spectral lever failed on <b>III</b> and <b>atau</b>, and vibrato made them
worse. Looking at the model's own timing explains why: for your sentence the word <b>atau</b> got durations
<code>16, 6, 2, 3</code> frames &mdash; one vowel held for <b>400 ms</b> &mdash; while in another sentence the same
word got <code>1, 4, 2, 3</code> (one phoneme only <b>25 ms</b>). 15&ndash;17% of all phonemes get a single frame.
A 400 ms vowel with frozen timbre can only sound robotic/damp/watery, and a fast vibrato lands between the 25 ms
frames that carry the phase, which smears instead of helping.</p>
<p class="note">&#9654; <b>clamp</b>: clamp every phoneme to 50&ndash;200 ms (kills the 1-frame rush and the 400 ms
hold). &#9654; <b>med3</b>: median-filter the duration curve, then clamp. &#9654; <b>clampair</b>: clamp + 3 dB air.
&#9654; <b>trem</b>: clamp + a slow &plusmn;18% amplitude sway on the noise floor (life without pitch change).
&#9654; <b>vib03</b>: clamp + a gentler vibrato (0.3% at 3 Hz) to see whether slower is acceptable.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over the three sentences</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flatness 2-6k</th><th>&gt;4 kHz</th></tr>{mrows}</table>
<p class="note">Key question: does <b>clamp</b> make III and atau sound less robotic/damp without breaking normal
words? If yes, this is a clean bakeable fix inside the ONNX (a Clip on the duration tensor) &mdash; it changes no
weights and no spectral settings.</p>
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
open(os.path.join(ROOT, 'runs/listen/dur.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/dur')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/dur'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/dur.html'), os.path.join(ROOT, 'runs/local_v041/dur.html'))
print('page written, rows:', len(d['items']))
