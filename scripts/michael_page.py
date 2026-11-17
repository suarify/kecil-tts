"""Build runs/listen/michael.html from michael_ablate/items.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/michael_ablate/items.json'), encoding='utf-8'))

COLS = [('old', 'v0.2 (old export)', '#e05c5c'),
        ('fixH', 'v0.2.1 (FIX H, current)', '#5aa86c'),
        ('noPhase', 'FIX H, learned phase OFF', '#8a6cc8'),
        ('noShift', 'FIX H, pulse shift OFF', '#d69a2d'),
        ('keepHalf', 'FIX H + half noise -> harmonics', '#4a90d9'),
        ('noBurst', 'FIX H, burst path removed', '#2fb8b8'),
        ('noMod', 'FIX H, pulsed noise removed', '#b0752a')]

rows = []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    btns = ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')

m = d['metrics']
mrows = []
for k, lbl, _ in COLS:
    p, fl, o4 = m[k]
    mrows.append(f'<tr><td>{lbl}</td><td>{p:.3f}</td><td>{fl:.3f}</td><td>{o4:.3f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS Michael: where does the watery / phasey sound come from</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1120px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:34%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
#rows.blind button.p{{color:#0d1b26;background:#2c4a5e}}
#rows.blind button.p.on{{background:#5fd6d6}}
#shuffle{{background:#2fb8b8;color:#0d1b26;border:0;border-radius:99px;padding:9px 16px;font:700 13px Inter,sans-serif;cursor:pointer}}
table.m td{{padding:6px 14px;font-size:13.5px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael</span> &mdash; hunting the remaining watery / phasey sound</h1>
<p class="note">The owner hears watery + phase artifacts in v0.2.1 (FIX H). This round isolates the head's knobs on
Michael's own weights: the <b>learned phase</b> (u), the <b>pulse shift</b> (a delay of the whole pulse per frame),
and how much voiced noise energy stays in the gaps vs moves into the harmonics. All loudness-matched,
same noise seed per sentence.</p>
<p class="note">&#9654; <b>noPhase</b>: learned phase replaced by a flat phase (keeps the pulse shift).
&#9654; <b>noShift</b>: pulse shift removed (keeps the learned phase). &#9654; <b>keepHalf</b>: half the voiced noise
energy moved into the harmonics (solidity). &#9654; <b>noBurst</b>: the 3 ms burst path removed entirely.
&#9654; <b>noMod</b>: pulsed-noise sources dropped.</p>
<p><button id="shuffle">Start blind test</button> <span id="hint" style="color:#9db1bf;font-size:13px"></span></p>
<table id="rows">{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(d['items'])} sentences</h2>
<table class="m"><tr><th>version</th><th>periodicity</th><th>flatness 2-6 kHz</th><th>energy &gt;4 kHz</th></tr>{''.join(mrows)}</table>
<p class="note" style="margin-top:18px">Tell me the two best buttons (they can be different variants per problem:
e.g. one for the watery part, one for the phasey part) and I'll bake the winner into the next Michael export.</p>
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
let blind=false;
const shuffle=document.getElementById('shuffle'),tbl=document.getElementById('rows'),hint=document.getElementById('hint');
const NAMES={json.dumps([lbl for _, lbl, _ in COLS])};
shuffle.onclick=()=>{{
  blind=!blind; tbl.classList.toggle('blind',blind); document.querySelectorAll('#rows td.bs').forEach(td=>{{
    const btns=[...td.children];
    if(blind){{const names=[...btns].map(b=>b.querySelector('.l').textContent);
      for(let i=names.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[names[i],names[j]]=[names[j],names[i]];}}
      btns.forEach((b,i)=>b.querySelector('.l').textContent=names[i]);}}
    else btns.forEach((b,i)=>b.querySelector('.l').textContent=' '+NAMES[i]);
  }});
  hint.textContent=blind?'Buttons are shuffled - pick the cleanest one, then press again to reveal.':'';
  shuffle.textContent=blind?'Reveal + stop blind test':'Start blind test';
}};
</script></body></html>"""
open(os.path.join(ROOT, 'runs/listen/michael.html'), 'w', encoding='utf-8').write(html)
print('page written, rows:', len(d['items']))
