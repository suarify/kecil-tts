"""All variants loudness-matched to the teacher: raw / air6 / FIX H / FIX H+air6 / keep08."""
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
OUT = 'runs/listen/nicole_loud2'
DST = 'runs/local_v041/nicole_loud2'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(OUT)
teacher = torch.load('runs/sm19_out/nicole.pt', map_location='cpu', weights_only=False)
rows = torch.load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
MODELS = {
    'teacher': None,
    'raw': 'runs/nicole_probe/export_faithful/paradee_int8.onnx',
    'air6': 'runs/nicole_probe/export_air6/paradee_int8.onnx',
    'fixH': 'runs/nicole_probe/export_yati/paradee_int8.onnx',
    'fixH_air6': 'runs/nicole_probe/export_fixh_air6/paradee_int8.onnx',
    'keep08': 'runs/nicole_probe/export_keep08/paradee_int8.onnx',
}
sess = {k: (ort.InferenceSession(v, providers=['CPUExecutionProvider']) if v else None) for k, v in MODELS.items()}

rows_html = ''
for i in ids:
    t = texts[i]
    t_audio = teacher[i].numpy().astype('float32') / 32767.0
    t_rms = float(np.sqrt((t_audio.astype('float64') ** 2).mean()))
    files = {}
    for name, s in sess.items():
        if s is None:
            a = t_audio
        else:
            ids_t = np.array([g2p.tokens(g2p(normalize(t)))], dtype=np.int64)
            a = s.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
            a = a * (t_rms / (float(np.sqrt((a.astype('float64') ** 2).mean())) + 1e-12))
            a = np.clip(a, -1, 1)
        fn = f'{i}_{name}.wav'
        sf.write(f'{OUT}/{fn}', a, SR)
        files[name] = f'nicole_loud2/{fn}'
    cols = [('teacher', '#00b3b3', 'TEACHER'), ('fixH_air6', '#5aa86c', 'FIX H + air6'),
            ('fixH', '#4a90d9', 'FIX H (Yati)'), ('air6', '#8a6cc8', 'air6, no FIX H'),
            ('keep08', '#d69a2d', 'gap keep0.8'), ('raw', '#e05c5c', 'raw')]
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{files[k]}">&#9654; {lbl}</button>' for k, c, lbl in cols)
    rows_html += f'<tr><td class="tx"><span class="cat">{i}</span> {t}</td><td>{btns}</td></tr>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Nicole: FIX H loudness matched</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1250px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:8px 7px;border-top:1px solid #22384a;font-size:13px}}
.tx{{width:36%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:6px 9px;font:600 11px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Nicole</span> &mdash; FIX H, now loudness-matched</h1>
<p class="note">Every student column is scaled to the <b>same RMS as the teacher</b>. This is the first fair listen of
FIX H on nicole: gap-masked noise + burst gate (Yati's fix), with and without the +6 dB air shelf. TEACHER is the
target. Find the column closest to the teacher's clarity — if none are close, the head itself is the limit.</p>
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
open('runs/listen/nicole_loud2.html', 'w', encoding='utf-8').write(html)
shutil.copytree(OUT, DST)
shutil.copy('runs/listen/nicole_loud2.html', 'runs/local_v041/nicole_loud2.html')
print('page written')
