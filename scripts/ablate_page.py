"""Build runs/listen/ablate.html from ablate/items.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/ablate/items.json'), encoding='utf-8'))

COLS = [('target', 'Chatterbox original', '#2fb8b8'),
        ('v04', 'v0.4 as trained', '#e05c5c'),
        ('harm', 'Harmonics only (learned phase)', '#8a6cc8'),
        ('harm_pure', 'Harmonics only, phase OFF', '#5aa86c'),
        ('fix1', 'FIX A: phase OFF + noise 50%', '#d69a2d'),
        ('fix2', 'FIX B: phase OFF + noise 25%', '#4a90d9'),
        ('noise', 'Noise part only', '#7d8b96')]

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
    p, f = m[k]
    mrows.append(f'<tr><td>{lbl}</td><td>{p:.3f}</td><td>{f:.3f}</td></tr>')
mrows.append('<tr class="ref"><td>Chatterbox original (goal)</td>'
             f'<td>{m["target"][0]:.3f}</td><td>{m["target"][1]:.3f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS ablation: where does the watery/phaser sound come from?</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1080px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:42%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
#rows.blind button.p{{color:#0d1b26;background:#2c4a5e}}
#rows.blind button.p.on{{background:#5fd6d6}}
#shuffle{{background:#2fb8b8;color:#0d1b26;border:0;border-radius:99px;padding:9px 16px;font:700 13px Inter,sans-serif;cursor:pointer}}
table.m td{{padding:6px 14px;font-size:13.5px}} table.m tr.ref td{{color:#5fd6d6;border-top:2px solid #2fb8b8}}
</style></head><body><main>
<h1>Kecil-TTS <span>ablation</span> &mdash; which ingredient makes it watery / phaser-y?</h1>
<p class="note">Every frame of v0.4 is built from a <b>harmonic part</b> (sine waves locked to the predicted pitch,
shaped by learned gains) plus a <b>noise part</b> (shaped random noise). The harmonic part also gets two
"clever" extras the network learned: a <b>phase rotation</b> per frequency region and a <b>pulse shift</b>.
This page plays each ingredient alone, with the same noise seed per sentence, so you can hear exactly
what each one contributes:</p>
<p class="note">&#9654; <b>Harmonics only (learned phase)</b> vs <b>phase OFF</b>: if the second sounds clearly
cleaner, the learned phase machinery is the phaser. &nbsp;&#9654; <b>FIX A / FIX B</b>: phase disabled AND the noise part
moved back into harmonics (energy-compensated, 50% / 25% of its voiced energy above 1.5&nbsp;kHz). If one of those
sounds like the Chatterbox original, that setting can be baked into the next ONNX export with no retraining.
All synth versions are loudness-matched.</p>
<p><button id="shuffle">Start blind test</button> <span id="hint" style="color:#9db1bf;font-size:13px"></span></p>
<table id="rows">{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(d['items'])} sentences</h2>
<p class="note" style="margin:2px 0 8px">Periodicity = clarity of the pitch line (autocorrelation peak; target is natural
speech, so ~0.70). Flatness 2&ndash;6&nbsp;kHz = how noise-like the presence band is (target 0.62).</p>
<table class="m"><tr><th>version</th><th>periodicity</th><th>flatness 2-6 kHz</th></tr>{''.join(mrows)}</table>
<p class="note" style="margin-top:18px"><b>Note on Michael:</b> same decoder head, so the winning setting transfers;
baking it into his ONNX needs his training checkpoint from the other workspace (or a re-export).</p>
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
const NAMES=['Chatterbox original','v0.4 as trained','Harmonics only (learned phase)','Harmonics only, phase OFF','FIX A: phase OFF + noise 50%','FIX B: phase OFF + noise 25%','Noise part only'];
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
