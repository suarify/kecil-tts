"""Path B page: TEACHER vs YATI student vs PATH B (Yati student converted to nicole), loudness-matched."""
import os
import shutil

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
TEACH = torch.load('runs/sm19_out/nicole.pt', map_location='cpu', weights_only=False)
rows = torch.load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
OUT = 'runs/listen/pathB'
DST = 'runs/local_v041/pathB'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(OUT)

rows_html = ''
for i in ids:
    t = texts[i]
    t_audio = TEACH[i].numpy().astype('float32') / 32767.0
    t_rms = float(np.sqrt((t_audio.astype('float64') ** 2).mean()))
    files = {}
    sources = {
        'teacher': t_audio,
        'yati': sf.read(f'runs/sm15b_in/yati_{i}.wav')[0],
        'pathB': sf.read(f'runs/kaggle_sm15b/converted/{i}.wav')[0],
    }
    for name, a in sources.items():
        s_rms = float(np.sqrt((a.astype('float64') ** 2).mean()))
        a = np.clip(a * (t_rms / (s_rms + 1e-12)), -1, 1)
        fn = f'{i}_{name}.wav'
        sf.write(f'{OUT}/{fn}', a, 24000)
        files[name] = f'pathB/{fn}'
    cols = [('teacher', '#00b3b3', 'TEACHER (nicole)'), ('pathB', '#5aa86c', 'PATH B (Yati→nicole VC)'),
            ('yati', '#e05c5c', 'YATI student (wrong voice)')]
    btns = ''.join(f'<button class="p" style="--c:{c}" data-src="{files[k]}">&#9654; {lbl}</button>' for k, c, lbl in cols)
    rows_html += f'<tr><td class="tx"><span class="cat">{i}</span> {t}</td><td>{btns}</td></tr>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Path B: Yati student → nicole VC</title>
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
<h1>Kecil-TTS <span>Path B</span> &mdash; Yati's clean student converted to nicole</h1>
<p class="note">All three are loudness-matched. TEACHER = the nicole training clip. PATH B = Yati v0.4.1 student audio
passed through ONE ChatterboxVC to the nicole reference (8 clips, 0 failures, 4.8 min CPU). YATI = the same student
without conversion (wrong voice, for reference). If PATH B keeps Yati's clarity while sounding like nicole, we skip
training a murky student entirely.</p>
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
open('runs/listen/pathB.html', 'w', encoding='utf-8').write(html)
shutil.copytree(OUT, DST)
shutil.copy('runs/listen/pathB.html', 'runs/local_v041/pathB.html')
print('page written:', len(ids), 'sentences')
