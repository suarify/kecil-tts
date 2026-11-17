"""Build runs/listen/air.html from air/items.json — two sections: Michael de-chop, Yati brightness."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/air/items.json'), encoding='utf-8'))

MCOLS = [('base', 'Michael base (published hang7)', '#e05c5c'),
         ('burstSoft', 'bursts soft (30% in voiced)', '#d69a2d'),
         ('burstFree', 'bursts free (old v0.2 level)', '#5aa86c'),
         ('smooth', 'base + gain smoothing in time', '#8a6cc8'),
         ('air4', 'base + air shelf +4 dB', '#2fb8b8'),
         ('air6x', 'base + air +6 dB + exciter', '#4a90d9'),
         ('keep60', 'base + 60% gap-noise into harmonics', '#b0752a')]

YCOLS = [('base', 'Yati base (v0.4.1)', '#e05c5c'),
         ('hang7', 'base with 7-frame hangover', '#5aa86c'),
         ('burstSoft', 'bursts soft (30% in voiced)', '#d69a2d'),
         ('smooth', 'base + gain smoothing in time', '#8a6cc8'),
         ('air4', 'base + air shelf +4 dB', '#2fb8b8'),
         ('air6x', 'base + air +6 dB + exciter', '#4a90d9'),
         ('keep60', 'base + 60% gap-noise into harmonics', '#b0752a')]


def section(title, note, items, cols):
    rows = []
    for it in items:
        rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
        btns = ''.join(
            f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
            for k, lbl, c in cols)
        rows.append(f'<tr><td class="tx"><span class="cat">{it["cat"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')
    return f'<h2>{title}</h2><p class="note">{note}</p><table>{chr(10).join(rows)}</table>'


mich = [it for it in d['items'] if it['voice'] == 'michael']
yati = [it for it in d['items'] if it['voice'] == 'yati']
mm, ym = d['metrics']['michael'], d['metrics']['yati']


def mtable(m, cols, extra=''):
    head = '<tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th><th>consonant energy</th></tr>'
    body = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td>'
                   f'<td>{m[k][3]:.3f}</td><td>{m[k][4]:.1f}</td></tr>' for k, lbl, _ in cols)
    return f'<table class="m">{head}{body}</table>'


html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Kecil-TTS round 4: de-chop Michael, brighten Yati</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1200px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
h2{{font-family:Sora,Inter,sans-serif;font-size:17px;margin-top:34px}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;font-weight:600;border-top:0}}
.tx{{width:27%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}} table.m tr:nth-child(1) td{{border-top:1px solid #22384a}}
</style></head><body><main>
<h1>Kecil-TTS <span>round 4</span> &mdash; de-chop Michael, brighten Yati</h1>
<p class="note">Michael: his harmonics and timbre are good now, the complaint is <b>choppiness</b>. So the burst path
gets progressively less aggressive (soft / free) and the gains get smoothed in time.<br>
Yati: measurement shows her speech has almost <b>no energy above 7 kHz (0.4% of total; Michael has 2.8%)</b> &mdash;
that is the muffled sound left over from her dark training targets. The fix that needs no retraining is an <b>air
shelf</b> (high-shelf EQ above 6.5 kHz) and optionally a <b>harmonic exciter</b> that synthesises the missing top
octave from her own harmonics.</p>
{section('Michael &mdash; choppiness', 'Base is the published v0.2.1 (hang7). Listen for characters being cut vs wateriness coming back.', mich, MCOLS)}
{mtable(mm, MCOLS)}
{section('Yati &mdash; brightness', 'Base is v0.4.1 as published. The air buttons only add top-end; the de-chop buttons test the same hangover idea.', yati, YCOLS)}
{mtable(ym, YCOLS)}
<p class="note">Tell me one pick per voice (michael: cleanest; yati: brightest without hiss) and I bake them into
the ONNX. The air shelf becomes a small linear-phase FIR inside the graph, so it costs almost nothing.</p>
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
open(os.path.join(ROOT, 'runs/listen/air.html'), 'w', encoding='utf-8').write(html)
dst = os.path.join(ROOT, 'runs/local_v041/air')
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(os.path.join(ROOT, 'runs/listen/air'), dst)
shutil.copy(os.path.join(ROOT, 'runs/listen/air.html'), os.path.join(ROOT, 'runs/local_v041/air.html'))
print('page written, michael rows', len(mich), 'yati rows', len(yati))
