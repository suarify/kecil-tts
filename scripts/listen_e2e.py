"""Build runs/listen/e2e.html: the 8M ONNX int8 student next to the Chatterbox clip it was trained toward.

usage: listen_e2e.py <runs/k06 or runs/k07>
Held-out rows were never trained on; "train" rows were (the model has seen their audio).
"""
import json
import os
import shutil
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else 'runs/k06')
tag = os.path.basename(src.rstrip('/'))
out = os.path.join(ROOT, 'runs', 'listen', tag)
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
sc = pd.read_csv(os.path.join(src, 'e2e_scores.csv'))
rows = []
for r in sc.itertuples():
    a = f'{r.split}_{r.id}_onnx_int8.wav'
    t = f'{r.split}_{r.id}_target.wav'
    shutil.copy(os.path.join(src, 'audio', a), os.path.join(out, a))
    has_t = os.path.exists(os.path.join(src, 'audio', t))
    if has_t:
        shutil.copy(os.path.join(src, 'audio', t), os.path.join(out, t))
    rows.append(dict(id=r.id, split=r.split, text=r.text, heard=str(r.heard), cer=round(float(r.cer_onnx), 3),
                     student=f'{tag}/{a}', target=f'{tag}/{t}' if has_t else ''))
order = {'test': 0, 'held': 1, 'train': 2}
rows.sort(key=lambda x: (order.get(x['split'], 3), x['id']))
html = '''<!doctype html><html><head><meta charset="utf-8"><title>8M student check</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:900px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--mut)}.row{border-top:1px solid var(--line);padding:12px 0}
.t{font-weight:600}.s{font-size:12px;color:var(--mut)}.g{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}.g div{flex:1;min-width:260px}audio{width:100%;height:32px}
</style></head><body><h1>The 8 MB student vs the Chatterbox clip</h1>
<p>Left: the 8M int8 model speaking from text only (the file that would ship). Right: the Chatterbox clip it was trained toward. "held" sentences were never seen in training; "train" ones were.</p><div id="l"></div>
<script>const R=__R__;const L=document.getElementById("l");
for(const r of R){const d=document.createElement("div");d.className="row";
d.innerHTML=`<div class="s">${r.split} · ${r.id} · CER ${(r.cer*100).toFixed(0)}%</div><div class="t"></div><div class="s">Whisper heard: <span class="h"></span></div>
<div class="g"><div>8M student<audio controls preload="none" src="${r.student}"></audio></div>${r.target?`<div>Chatterbox target<audio controls preload="none" src="${r.target}"></audio></div>`:""}</div>`;
d.querySelector(".t").textContent=r.text;d.querySelector(".h").textContent=r.heard;L.appendChild(d)}</script></body></html>'''
open(os.path.join(ROOT, 'runs', 'listen', f'{tag}.html'), 'w', encoding='utf-8').write(html.replace('__R__', json.dumps(rows, ensure_ascii=False)))
print(len(rows), 'rows ->', os.path.join(ROOT, 'runs', 'listen', f'{tag}.html'))
