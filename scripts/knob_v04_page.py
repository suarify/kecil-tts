"""Build runs/listen/knob_v04.html from knob_v04/items.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/knob_v04/items.json'), encoding='utf-8'))

COLS = [('target', 'Chatterbox original', '#2fb8b8'),
        ('1.0', 'v0.4 as trained', '#e05c5c'),
        ('0.5', 'Knob 50%', '#5aa86c'),
        ('0.25', 'Knob 25%', '#8a6cc8'),
        ('eq', 'Knob 50% + brightness', '#d69a2d')]

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
    f, e = m[k]
    mrows.append(f'<tr><td>{lbl}</td><td>{f:.3f}</td><td>{e:.3f}</td></tr>')
mrows.append('<tr class="ref"><td>Chatterbox target (goal)</td>'
             f'<td>{m["target"][0]:.3f}</td><td>{m["target"][1]:.3f}</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS knob test: v0.4 watery/phaser fix (small sample)</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:980px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top}}
.tx{{width:46%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 11px;font:600 12.5px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
#rows.blind button.p{{color:#0d1b26;background:#2c4a5e}}
#rows.blind button.p.on{{background:#5fd6d6}}
#shuffle{{background:#2fb8b8;color:#0d1b26;border:0;border-radius:99px;padding:9px 16px;font:700 13px Inter,sans-serif;cursor:pointer}}
table.m td{{padding:6px 14px;font-size:13.5px}} table.m tr.ref td{{color:#5fd6d6;border-top:2px solid #2fb8b8}}
</style></head><body><main>
<h1>Kecil-TTS <span>knob test</span> &mdash; is the watery / phaser sound fixable without retraining?</h1>
<p class="note">Same sentence, five versions. <b>Chatterbox original</b> = the exact clip the decoder was trained on.
<b>v0.4 as trained</b> is the current sound (watery / phaser-y). The <b>knob</b> moves energy back from the decoder's
random-noise part to its pitch-locked harmonic part in voiced frames above 1.5&nbsp;kHz &mdash; no retraining, just a
different mix of the two sources the decoder already predicts. <b>+ brightness</b> adds a +4&nbsp;dB high shelf (1.5&ndash;4&nbsp;kHz)
to test the second half of the problem (the Chatterbox targets are themselves darker than the reference voice).</p>
<p class="note">All five use the <i>same random noise seed</i> per sentence, so differences you hear are the knob only.
Metrics: <b>flatness</b> 2&ndash;6&nbsp;kHz (target 0.62; higher = more watery), <b>&gt;4&nbsp;kHz</b> = brightness share (target 0.084).</p>
<p><button id="shuffle">Start blind test</button> <span id="hint" style="color:#9db1bf;font-size:13px"></span></p>
<table id="rows">{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(d['items'])} sentences</h2>
<table class="m"><tr><th>version</th><th>flatness 2-6 kHz &darr;</th><th>energy &gt; 4 kHz &uarr;</th></tr>{''.join(mrows)}</table>
<p class="note" style="margin-top:18px"><b>Note on Michael:</b> this test uses Yati&nbsp;v0.4 because her PyTorch
checkpoints are in this repo. Michael is the same decoder head, so the same knob applies &mdash; to bake it into his
ONNX we need his training checkpoint from the other workspace (or a re-export with the knob inside).</p>
</main>
<script>
let cur=null;
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  if(cur&&cur!==b){{cur.pause();cur.classList.remove('on');}}
  if(cur===b&&!b.paused)return;
  cur=b; b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play(); cur=a;
}});
let blind=false;
const shuffle=document.getElementById('shuffle'),tbl=document.getElementById('rows'),hint=document.getElementById('hint');
shuffle.onclick=()=>{{
  blind=!blind; tbl.classList.toggle('blind', blind); document.querySelectorAll('#rows td.bs').forEach(td=>{{
    const btns=[...td.children];
    if(blind){{const names=[...btns].map(b=>b.querySelector('.l').textContent);
      for(let i=names.length-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[names[i],names[j]]=[names[j],names[i]];}}
      btns.forEach((b,i)=>b.querySelector('.l').textContent=names[i]);}}
    else btns.forEach((b,i)=>b.querySelector('.l').textContent=' '+['Chatterbox original','v0.4 as trained','Knob 50%','Knob 25%','Knob 50% + brightness'][i]);
  }});
  hint.textContent=blind?'Buttons are shuffled - pick the one that sounds least watery, then press again to reveal.':'';
  shuffle.textContent=blind?'Reveal + stop blind test':'Start blind test';
}};
</script></body></html>"""
open(os.path.join(ROOT, 'runs/listen/knob_v04.html'), 'w', encoding='utf-8').write(html)
print('page written, rows:', len(d['items']))
