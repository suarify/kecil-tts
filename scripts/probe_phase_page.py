"""Phase-stabilization A/B: Path-B teacher vs probe current / phase-smoothed / phase-shrunk."""
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
OUT = 'runs/listen/probe_phase'
DST = 'runs/local_v041/probe_phase'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(OUT)

rows = torch.load('runs/kaggle_sm15d/nicoleb_rows.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
MODELS = {
    'current': 'runs/probe_small_export/paradee_int8.onnx',
    'smooth': 'runs/probe_phase_smooth/paradee_int8.onnx',
    'shrink': 'runs/probe_phase_shrink/paradee_int8.onnx',
}
sess = {k: ort.InferenceSession(v, providers=['CPUExecutionProvider']) for k, v in MODELS.items()}

rows_html = ''
for i in ids:
    teacher = sf.read(f'runs/kaggle_sm15b/converted/{i}.wav')[0]
    t_rms = float(np.sqrt((teacher.astype('float64') ** 2).mean()))
    ids_t = np.array([g2p.tokens(g2p(normalize(texts[i])))], dtype=np.int64)
    files = {}
    for name, s in [('teacher', None)] + list(sess.items()):
        a = teacher if s is None else s.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        if s is not None:
            a = np.clip(a * (t_rms / (float(np.sqrt((a.astype('float64') ** 2).mean())) + 1e-12)), -1, 1)
        fn = f'{i}_{name}.wav'
        sf.write(f'{OUT}/{fn}', a, SR)
        files[name] = f'probe_phase/{fn}'
    cols = [('teacher', '#00b3b3', 'TEACHER'), ('current', '#e05c5c', 'probe current'),
            ('smooth', '#5aa86c', 'phase smooth 5'), ('shrink', '#4a90d9', 'phase x0.5')]
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{files[k]}">&#9654; {lbl}</button>' for k, c, lbl in cols)
    rows_html += f'<tr><td class="tx"><span class="cat">{i}</span> {texts[i]}</td><td>{btns}</td></tr>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Probe phase stabilization</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:8px 7px;border-top:1px solid #22384a;font-size:13px}}
.tx{{width:42%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:6px 10px;font:600 11px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Probe</span> &mdash; phase stabilization test</h1>
<p class="note">All loudness-matched. TEACHER = the clean Path-B target. The other three are the same 2k-step probe
student with different harmonic-phase handling: <b>current</b> (the slightly phasey one), <b>phase smooth 5</b>
(circular-average u over 5 frames), <b>phase x0.5</b> (shrink phase angles by half). Find the one closest to
TEACHER on the words that were phasey — no training involved.</p>
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
open('runs/listen/probe_phase.html', 'w', encoding='utf-8').write(html)
shutil.copytree(OUT, DST)
shutil.copy('runs/listen/probe_phase.html', 'runs/local_v041/probe_phase.html')
print('page written, rows:', len(ids))
