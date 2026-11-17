"""Build runs/listen/ablate.html from ablate/items.json (round 2: gap-masked noise fixes)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/ablate/items.json'), encoding='utf-8'))

COLS = [('target', 'Chatterbox original', '#2fb8b8'),
        ('v04', 'v0.4 as trained', '#e05c5c'),
        ('harm', 'Harmonics only (round-1 pick)', '#8a6cc8'),
        ('fixC', 'FIX C: noise into gaps', '#d69a2d'),
        ('fixC2', 'FIX C2: gaps + half noise', '#5aa86c'),
        ('fixD', 'FIX D: gaps, half noise, no pulse-noise', '#4a90d9')]

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
t = m['target']
mrows.append(f'<tr class="ref"><td>Chatterbox original (goal)</td><td>{t[0]:.3f}</td><td>{t[1]:.3f}</td><td>{t[2]:.3f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS ablation 2: noise into the gaps between harmonics</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1080px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:40%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
#rows.blind button.p{{color:#0d1b26;background:#2c4a5e}}
#rows.blind button.p.on{{background:#5fd6d6}}
#shuffle{{background:#2fb8b8;color:#0d1b26;border:0;border-radius:99px;padding:9px 16px;font:700 13px Inter,sans-serif;cursor:pointer}}
table.m td{{padding:6px 14px;font-size:13.5px}} table.m tr.ref td{{color:#5fd6d6;border-top:2px solid #2fb8b8}}
</style></head><body><main>
<h1>Kecil-TTS <span>ablation 2</span> &mdash; keep the noise energy, but move it out of the harmonics</h1>
<p class="note">Round 1 said: the harmonic part with its learned phase is the clean part, and the watery/phaser
texture comes from the <b>noise part smearing across the harmonic partials</b> (your pick "harmonics only" was
muffled only because the noise &mdash; which also carries the brightness fill &mdash; was removed entirely).</p>
<p class="note">So this round keeps the learned phase AND the full noise energy, but in voiced frames the noise
spectrum is masked <b>into the gaps between the harmonics</b> (&gt;800&nbsp;Hz), the way real aspiration sits.
&nbsp;&#9654; <b>FIX C</b>: all noise energy, gap-masked. &#9654; <b>FIX C2</b>: gap-masked + only half the noise
energy (the other half moved into the harmonics, energy-compensated). &#9654; <b>FIX D</b>: C2 + the
pulsed-noise sources dropped (plain noise only). All loudness-matched, same noise seed per sentence.</p>
<p><button id="shuffle">Start blind test</button> <span id="hint" style="color:#9db1bf;font-size:13px"></span></p>
<table id="rows">{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(d['items'])} sentences</h2>
<table class="m"><tr><th>version</th><th>periodicity</th><th>flatness 2-6 kHz</th><th>energy &gt;4 kHz</th></tr>{''.join(mrows)}</table>
<p class="note" style="margin-top:18px"><b>Note on Michael:</b> same decoder head &mdash; the winning setting
transfers to his next export (needs his training checkpoint from the other workspace, or a re-export).</p>
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
const NAMES=['Chatterbox original','v0.4 as trained','Harmonics only (round-1 pick)','FIX C: noise into gaps','FIX C2: gaps + half noise','FIX D: gaps, half noise, no pulse-noise'];
shuffle.onclick=()=>{{
  blind=!blind; tbl.classList.toggle('blind',blind); document.querySelectorAll('#rows td.bs').forEach(td=>{{
    const btns=[...td.children];
    if(blind){{const names=[...btns].map(b=>b.querySelector('.l').textContent);
      for(let i=names.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[names[i],names[j]]=[names[j],names[i]];}}
      btns.forEach((b,i)=>b.querySelector('.l').textContent=names[i]);}}
    else btns.forEach((b,i)=>b.querySelector('.l').textContent=' '+NAMES[i]);
  }});
  hint.textContent=blind?'Buttons are shuffled - pick the one closest to the Chatterbox original, then press again to reveal.':'';
  shuffle.textContent=blind?'Reveal + stop blind test':'Start blind test';
}};
</script></body></html>"""
open(os.path.join(ROOT, 'runs/listen/ablate.html'), 'w', encoding='utf-8').write(html)
print('page written, rows:', len(d['items']))
