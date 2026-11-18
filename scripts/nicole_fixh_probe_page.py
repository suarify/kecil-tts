"""Critical re-test: same 500-step nicole decoder, rendered with FIX H vs faithful vs teacher."""
import csv
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

OUT = 'runs/listen/nicole_fixh_probe'
os.makedirs(OUT, exist_ok=True)
texts = {r['id']: r['text'] for r in __import__('torch').load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']
teacher = __import__('torch').load('runs/sm19_out/nicole.pt', map_location='cpu', weights_only=False)

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = {
    'teacher': None,
    'faithful': ort.InferenceSession('runs/nicole_probe/export_faithful/paradee_int8.onnx', providers=['CPUExecutionProvider']),
    'fixH': ort.InferenceSession('runs/nicole_probe/export_fixh/paradee_int8.onnx', providers=['CPUExecutionProvider']),
}

rows = ''
for i in ids:
    text = texts[i]
    sf.write(f'{OUT}/{i}_teacher.wav', teacher[i].numpy().astype('float32') / 32767.0, 24000)
    ids_t = np.array([g2p.tokens(g2p(normalize(text)))], dtype=np.int64)
    for name in ('faithful', 'fixH'):
        y = sess[name].run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        sf.write(f'{OUT}/{i}_{name}.wav', y, 24000)
    rows += (f'<tr><td class="tx"><span class="cat">{i}</span> {text}</td><td>'
             f'<button class="p" style="--c:#00b3b3" data-src="nicole_fixh_probe/{i}_teacher.wav">&#9654; TEACHER</button>'
             f'<button class="p" style="--c:#e05c5c" data-src="nicole_fixh_probe/{i}_faithful.wav">&#9654; student raw</button>'
             f'<button class="p" style="--c:#5aa86c" data-src="nicole_fixh_probe/{i}_fixH.wav">&#9654; student FIX H</button>'
             f'</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Critical re-test: FIX H on the nicole probe</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 8px;border-top:1px solid #22384a;font-size:13.5px}}
.tx{{width:46%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Critical re-test: <span>FIX H on the nicole probe</span></h1>
<p class="note">Same 500-step student decoder. The sample you heard was "student raw" (no processing). Yati's accepted
v0.4.1 uses FIX H: noise moved into the gaps between harmonics + burst gate + air shelf. Compare the three buttons on
the same sentences — this is the test I should have run before calling the student hopeless.</p>
<table>{rows}</table>
<p class="note">If "student FIX H" is clean, the route is: more training steps + FIX H, no new architecture needed.</p>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/nicole_fixh_probe.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/nicole_fixh_probe'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(OUT, dst)
shutil.copy('runs/listen/nicole_fixh_probe.html', 'runs/local_v041/nicole_fixh_probe.html')
print('page written:', len(ids), 'sentences')
