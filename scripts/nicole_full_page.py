"""Final gate: Path-B teacher vs FULL 12k nicole student, loudness-matched."""
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

SR = 24000
OUT = 'runs/listen/nicole_full'
DST = 'runs/local_v041/nicole_full'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(OUT)

rows = torch.load('runs/kaggle_sm15e/nicoleb_rows.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession('runs/nicole_full_export/paradee_int8_webgpu.onnx', providers=['CPUExecutionProvider'])
# the small probe for comparison (if its export survived the disk cleanup)
probe = None
if os.path.exists('runs/probe_small_export/paradee_int8.onnx'):
    probe = ort.InferenceSession('runs/probe_small_export/paradee_int8.onnx', providers=['CPUExecutionProvider'])

rows_html = ''
for i in ids:
    teacher = sf.read(f'runs/kaggle_sm15b/converted/{i}.wav')[0]
    t_rms = float(np.sqrt((teacher.astype('float64') ** 2).mean()))
    ids_t = np.array([g2p.tokens(g2p(normalize(texts[i])))], dtype=np.int64)
    variants = [('teacher', teacher)]
    for name, s in [('full', sess)] + ([('probe2k', probe)] if probe else []):
        y = s.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        y = np.clip(y * (t_rms / (float(np.sqrt((y.astype('float64') ** 2).mean())) + 1e-12)), -1, 1)
        variants.append((name, y))
    for name, a in variants:
        sf.write(f'{OUT}/{i}_{name}.wav', a, SR)
    cols = [('teacher', '#00b3b3', 'TEACHER (Path B)')]
    if probe:
        cols.append(('probe2k', '#e05c5c', 'probe 2k'))
    cols.append(('full', '#5aa86c', 'FULL 12k'))
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="nicole_full/{i}_{k}.wav">&#9654; {lbl}</button>' for k, c, lbl in cols)
    rows_html += f'<tr><td class="tx"><span class="cat">{i}</span> {texts[i]}</td><td>{btns}</td></tr>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Nicole final gate</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1050px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 8px;border-top:1px solid #22384a;font-size:13px}}
.tx{{width:42%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 11px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Nicole</span> &mdash; FINAL gate: full 12k student</h1>
<p class="note">The decisive listen. TEACHER = clean Path-B target (Yati student + one VC to nicole). FULL 12k =
the trained 8.46 MB model, FIX H + air6 export, loudness-matched. Probe 2k shown where available so you can hear
the improvement. Publish as an 8 MB voice <b>only if</b> the full student is clean (no soft/watery/phasey).</p>
<table>{rows_html}</table>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/nicole_full.html', 'w', encoding='utf-8').write(html)
shutil.copytree(OUT, DST)
shutil.copy('runs/listen/nicole_full.html', 'runs/local_v041/nicole_full.html')
print('page written, rows:', len(ids), '| probe available:', probe is not None)
