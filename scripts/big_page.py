"""Build runs/listen/big.html: teacher vs small v0.3.0 vs big student variants."""
import json
import os
import shutil

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
d = json.load(open('runs/listen/big/items.json', encoding='utf-8'))
TEACH = {'iii': 'vc10/b/c0571.wav', 'call': 'vc15/old/c0184.wav'}


def mcd(a, b):
    import librosa
    M = lambda x: np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=24000, n_mels=64) + 1e-8)  # noqa: E731
    A, B = M(a), M(b)
    T = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :T] - B[:, :T]).mean())


COLS = [('small', 'SMALL student (v0.3.0, 8.3 MB)', '#e05c5c'),
        ('big', 'BIG student (11.9 MB, faithful)', '#d69a2d'),
        ('bigAir3', 'BIG + 3 dB air', '#5aa86c'),
        ('bigAir5', 'BIG + 5 dB air', '#4a90d9')]

rows, sm, bg = [], [], []
for it in d['items']:
    rel = {k: v.split('runs/listen/')[-1] for k, v in it['files'].items()}
    t = TEACH.get(it['id'])
    btns = (f'<button class="p" style="--c:#00b3b3" data-src="{t}">&#9654;<span class="l"> TEACHER</span></button>' if t else '')
    btns += ''.join(
        f'<button class="p" style="--c:{c}" data-src="{rel[k]}">&#9654;<span class="l"> {lbl}</span></button>'
        for k, lbl, c in COLS if k in rel)
    rows.append(f'<tr><td class="tx"><span class="cat">{it["id"]}</span> {it["text"]}</td><td class="bs">{btns}</td></tr>')
    if t and os.path.exists(f'runs/listen/{t}') and 'small' in rel:
        tt = sf.read(f'runs/listen/{t}')[0]
        sm.append(mcd(sf.read(f'runs/listen/{rel["small"]}')[0], tt))
        bg.append(mcd(sf.read(f'runs/listen/{rel["big"]}')[0], tt))

m = d['metrics']
mrows = ''.join(f'<tr><td>{lbl}</td><td>{m[k][0]:.3f}</td><td>{m[k][1]:.3f}</td><td>{m[k][2]:.3f}</td><td>{m[k][3]:.3f}</td></tr>'
                for k, lbl, _ in COLS if k in m)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Michael: bigger student (11.9 MB) vs current (8.3 MB)</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1200px;margin:0 auto}}
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
<h1>Kecil-TTS <span>Michael</span> &mdash; bigger student vs current</h1>
<p class="note">Head width 160&rarr;320, blocks 4&rarr;6 (8.3 MB &rarr; 11.9 MB), trunk transferred, 16,000 training steps.
Text side unchanged, so the timing/speed are identical. TEACHER is the Chatterbox target.</p>
<table>{''.join(rows)}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
<p class="note">Distance to the teacher (lower = closer): small student {np.mean(sm):.2f}, big student {np.mean(bg):.2f}
&mdash; on this evidence the extra capacity bought nothing measurable, and the held-out spectral loss was slightly
worse than the small model's (2.515 vs 2.497). Listen anyway; if your ear agrees, we keep v0.3.0 and drop this variant.</p>
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
open('runs/listen/big.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/big'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree('runs/listen/big', dst)
shutil.copy('runs/listen/big.html', 'runs/local_v041/big.html')
print('page written | distance small', round(float(np.mean(sm)), 2), 'big', round(float(np.mean(bg)), 2))
