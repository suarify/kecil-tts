"""Compare FIX H variants on the same 500-step nicole decoder against the teacher."""
import csv
import json
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
OUT = 'runs/listen/nicole_tune'
os.makedirs(OUT, exist_ok=True)
teacher = torch.load('runs/sm19_out/nicole.pt', map_location='cpu', weights_only=False)
rows = torch.load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')

MODELS = {
    'teacher': None,
    'raw': 'runs/nicole_probe/export_faithful/paradee_int8.onnx',
    'yatiFix': 'runs/nicole_probe/export_yati/paradee_int8.onnx',
    'keep08': 'runs/nicole_probe/export_keep08/paradee_int8.onnx',
    'smoothAir': 'runs/nicole_probe/export_fixh/paradee_int8.onnx',
}
sess = {k: (ort.InferenceSession(v, providers=['CPUExecutionProvider']) if v else None) for k, v in MODELS.items()}


def metrics(a):
    import librosa
    a = np.asarray(a, np.float32)
    S = np.abs(librosa.stft(a, n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2
    tot = P.sum(0)
    keepf = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keepf]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    o4 = float((P[f > 4000][:, keepf].sum(0) / tot[keepf]).mean())
    o7 = float((P[f > 7000][:, keepf].sum(0) / tot[keepf]).mean())
    # periodicity
    an = a / (np.abs(a).max() + 1e-9)
    vals = []
    for i in range(0, len(an) - 960, 480):
        fr_ = an[i:i + 960]
        if np.sqrt((fr_ ** 2).mean()) < 0.02:
            continue
        fr_ = fr_ - fr_.mean()
        ac = np.correlate(fr_, fr_, 'full')[959:]
        ac /= ac[0] + 1e-12
        vals.append(float(ac[int(SR / 200):int(SR / 80)].max()))
    per = float(np.median(vals)) if vals else 0.0
    return [per, flat, o4, o7]


acc = {k: [] for k in MODELS}
rows_html = ''
for i in ids:
    t = texts[i]
    files = {}
    if i in teacher:
        y = teacher[i].numpy().astype('float32') / 32767.0
        sf.write(f'{OUT}/{i}_teacher.wav', y, SR)
        acc['teacher'].append(metrics(y))
        files['teacher'] = f'nicole_tune/{i}_teacher.wav'
    ids_t = np.array([g2p.tokens(g2p(normalize(t)))], dtype=np.int64)
    for name in ('raw', 'yatiFix', 'keep08', 'smoothAir'):
        y = sess[name].run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        sf.write(f'{OUT}/{i}_{name}.wav', y, SR)
        acc[name].append(metrics(y))
        files[name] = f'nicole_tune/{i}_{name}.wav'
    cols = [('teacher', '#00b3b3', 'TEACHER'), ('yatiFix', '#5aa86c', 'Yati FIX H'),
            ('keep08', '#4a90d9', 'gap keep 0.8'), ('smoothAir', '#8a6cc8', 'smooth+air'), ('raw', '#e05c5c', 'raw')]
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{files[k]}">&#9654; {lbl}</button>'
                   for k, c, lbl in cols if k in files)
    rows_html += f'<tr><td class="tx"><span class="cat">{i}</span> {t}</td><td>{btns}</td></tr>'

res = {k: [round(float(x), 3) for x in np.mean(v, 0)] for k, v in acc.items() if v}
print('model          period  flat    >4k     >7k')
for k, v in res.items():
    print(f'{k:12s}  {v[0]:.3f}   {v[1]:.3f}  {v[2]:.3f}  {v[3]:.3f}')
json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=1)

mrows = ''.join(f'<tr><td>{k}</td><td>{v[0]:.3f}</td><td>{v[1]:.3f}</td><td>{v[2]:.3f}</td><td>{v[3]:.3f}</td></tr>'
                for k, v in res.items())
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Tune nicole probe toward teacher</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1200px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:8px 7px;border-top:1px solid #22384a;font-size:13px}}
.tx{{width:42%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 7px;margin-right:5px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:6px 9px;font:600 11px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:4px 10px;font-size:12.5px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Nicole</span> &mdash; tune the probe toward the teacher</h1>
<p class="note">Same 500-step student decoder; only the export recipe changes. TEACHER is the target. "Yati FIX H" is
the exact recipe that made Yati acceptable (gap mask + burst gate, no smoothing/air). The "smooth+air" column is the
one you found too soft. Pick the column closest to the teacher's clarity.</p>
<table>{rows_html}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages</h2>
<table class="m"><tr><th>variant</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/nicole_tune.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/nicole_tune'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(OUT, dst)
shutil.copy('runs/listen/nicole_tune.html', 'runs/local_v041/nicole_tune.html')
print('page written')
