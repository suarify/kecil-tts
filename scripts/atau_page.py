"""Build runs/listen/atau.html from atau/items.json."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/atau/items.json'), encoding='utf-8'))

COLS = [('smoothGn', 'noise-smooth (current best)', '#5aa86c'),
        ('air3', '+3 dB air above 6.5 kHz', '#2fb8b8'),
        ('air5', '+5 dB air above 6.5 kHz', '#4a90d9'),
        ('breath', 'more breath (55% gap noise moved)', '#d69a2d'),
        ('air3breath', 'air + breath', '#8a6cc8'),
        ('vib', 'tiny vibrato (0.4%)', '#b0752a'),
        ('air3vib', 'air + vibrato', '#e05c5c')]

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
<title>Kecil-TTS Michael: the 'atau' word (robotic + damp)</title>
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
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; the "atau" word: robotic and damp</h1>
<p class="note">"atau" is a diphthong (a + au) right after the stressed III vowels, so the pitch has just settled and
the top of the spectrum is thinnest there. Three separate levers on the page:</p>
<p class="note">&#9654; <b>air</b> restores the missing top octave above 6.5 kHz (the "damp"). &#9654; <b>breath</b>
keeps more of the gap noise, which reads as natural aspiration instead of a dry buzz. &#9654; <b>vibrato</b> adds a
0.4% pitch sway at 4.7 Hz, which is what a real voice does and what a flat neural curve does not (the "robotic").
Loudness-matched.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over the three sentences</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flatness 2-6k</th><th>&gt;4 kHz</th></tr>{mrows}</table>
<p class="note">Listen for "atau" specifically in sentence 1 and 2 &mdash; tell me which lever fixes it, and whether
the vibrato sounds natural or seasick.</p>
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
open(os.path.join(ROOT, 'runs/listen/atau.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/atau')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/atau'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/atau.html'), os.path.join(ROOT, 'runs/local_v041/atau.html'))
print('page written, rows:', len(d['items']))
