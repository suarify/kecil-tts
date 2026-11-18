"""Loudness-match the student to the teacher and rebuild the nicole air page."""
import os
import shutil

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'runs/listen/nicole_air'
DST = 'runs/local_v041/nicole_loud'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(DST)

ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
texts = {}
import torch  # noqa: E402
for r in torch.load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False):
    texts[r['id']] = r['text']

# compute per-clip RMS of teacher, then gain student to the same RMS
rows = ''
for i in ids:
    t, _ = sf.read(f'{SRC}/{i}_teacher.wav')
    t_rms = float(np.sqrt((t.astype('float64') ** 2).mean()))
    fmap = {}
    for name in ('teacher', 'raw', 'air6', 'air9'):
        a, _ = sf.read(f'{SRC}/{i}_{name}.wav')
        if name != 'teacher':
            s_rms = float(np.sqrt((a.astype('float64') ** 2).mean()))
            a = a * (t_rms / (s_rms + 1e-12))       # loudness match
            a = np.clip(a, -1, 1)
        fn = f'{i}_{name}.wav'
        sf.write(f'{DST}/{fn}', a, 24000)
        fmap[name] = f'nicole_loud/{fn}'
    cols = [('teacher', '#00b3b3', 'TEACHER'), ('raw', '#e05c5c', 'raw (matched)'),
            ('air6', '#5aa86c', '+6 dB air (matched)'), ('air9', '#4a90d9', '+9 dB air (matched)')]
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{fmap[k]}">&#9654; {lbl}</button>' for k, c, lbl in cols)
    rows += f'<tr><td class="tx"><span class="cat">{i}</span> {texts.get(i,"")}</td><td>{btns}</td></tr>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Nicole loudness-matched</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 7px;border-top:1px solid #22384a;font-size:13px}}
.tx{{width:44%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 11px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Nicole</span> &mdash; loudness matched to teacher</h1>
<p class="note">The student was <b>19 dB quieter</b> (-36 dB vs teacher -17 dB RMS) — that's why everything sounded
"too soft". Here all versions are scaled to the <b>same RMS as the teacher</b>, so you're judging tone, not volume.
This gain can be baked into the ONNX for free — no retraining.</p>
<table>{rows}</table>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/nicole_loud.html', 'w', encoding='utf-8').write(html)
shutil.copy('runs/listen/nicole_loud.html', 'runs/local_v041/nicole_loud.html')
print('page written: loudness matched')
