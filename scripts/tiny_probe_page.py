"""Super-small sample: the 4 held-out student wavs from the 500-step CPU probe, plus their targets."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'runs/nicole_probe/ckpt/dec'
DST = 'runs/local_v041/nicole_tiny'
shutil.rmtree(DST, ignore_errors=True)
os.makedirs(DST)
for f in os.listdir(SRC):
    if f.endswith('.wav'):
        shutil.copy(f'{SRC}/{f}', f'{DST}/{f}')

rows = ''
for i in (1, 2, 3, 4):
    rows += (f'<tr><td class="tx">Sentence {i}</td><td>'
             f'<button class="p" style="--c:#e05c5c" data-src="nicole_tiny/held_{i}_target.wav">&#9654; TEACHER target</button>'
             f'<button class="p" style="--c:#5aa86c" data-src="nicole_tiny/held_{i}_student.wav">&#9654; STUDENT (500 steps)</button>'
             f'</td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Nicole tiny probe: student after 500 steps</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:900px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;font-size:13.5px}}
.tx{{width:28%}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 12px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Nicole</span> &mdash; super-small student sample</h1>
<p class="note">This is the earliest possible student audio: after only <b>500 decoder steps</b> on the 624 clean
rows (~9 min on CPU). Held spectral loss already dropped from 6.28 to 2.69. Compare STUDENT vs TEACHER target:
listen for watery/phase distortion. It is a rough voice, not the final one — we only judge whether the tone/artefacts
are acceptable before spending GPU time.</p>
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
open('runs/listen/nicole_tiny.html', 'w', encoding='utf-8').write(html)
shutil.copy('runs/listen/nicole_tiny.html', 'runs/local_v041/nicole_tiny.html')
print('page written:', len(os.listdir(DST)), 'wavs')
