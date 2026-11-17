"""Build runs/listen/michael3.html from michael_ablate3/items.json (choppiness round)."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/michael_ablate3/items.json'), encoding='utf-8'))

COLS = [('fixH21', 'published v0.2.1 (21-frame hangover)', '#e05c5c'),
        ('fixH0', 'gated, no hangover', '#d69a2d'),
        ('hang7', 'gated, 7-frame hangover (gentle)', '#5aa86c'),
        ('burstSoft', 'burst kept, 30% in voiced', '#2fb8b8'),
        ('keep25', 'quarter gap-noise, gentle gate', '#4a90d9'),
        ('harmFill', '40% gap-noise, gentle gate', '#8a6cc8'),
        ('harmOnly', 'harmonics only (no noise)', '#b0752a')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    btns = ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS Michael round 3: chopped characters vs watery noise</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1180px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:26%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> round 3 &mdash; stop the chopped characters, keep the clean core</h1>
<p class="note">Owner: v0.2.1 is <b>choppy, some characters chopped</b>, and harmonics-only is <b>cleaner</b>.
That means the burst gate (my 21-frame hangover removes bursts around every voiced stretch) is eating consonants,
and the gap noise is what stays watery. So: keep the harmonic core, restore consonants, tune the noise level.</p>
<p class="note">&#9654; <b>fixH21</b> is the published file (reference for the choppiness). &#9654; <b>hang7</b>:
consonants get their bursts back 14 ms sooner. &#9654; <b>burstSoft</b>: no gating at all, bursts just 30% quieter
inside voiced sound. &#9654; <b>keep25 / harmFill</b>: a quarter / 40% of the gap noise with a gentle gate &mdash;
the harmonics carry the rest. Loudness-matched to the published file.</p>
<table id="rows">{''.join(rows)}</table>
<p class="note">Pick the button that (a) no longer chops characters and (b) keeps the watery down. Two answers are
fine: "cleanest prosody" + "least watery".</p>
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
open(os.path.join(ROOT, 'runs/listen/michael3.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/michael_ablate3')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/michael_ablate3'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/michael3.html'), os.path.join(ROOT, 'runs/local_v041/michael3.html'))
print('page written, rows:', len(d['items']))
