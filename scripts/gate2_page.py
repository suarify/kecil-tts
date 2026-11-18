"""Gate 2 page: 15 converted nicole teacher clips (raw vs converted not available locally, so listen for artefacts)."""
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'runs/listen/nicole_teacher'
DST = 'runs/local_v041/nicole_teacher'
shutil.rmtree(DST, ignore_errors=True)
shutil.copytree(SRC, DST)

texts = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = sorted(os.path.basename(p)[:-4] for p in os.listdir(SRC) if p.endswith('.wav'))
rows = ''.join(
    f'<tr><td class="tx"><span class="cat">{i}</span> {texts.get(i, "")}</td>'
    f'<td><button class="p" data-src="nicole_teacher/{i}.wav">&#9654;<span class="l"> play</span></button></td></tr>'
    for i in ids)
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Gate 2: nicole converted teacher clips</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1000px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:9px 8px;border-top:1px solid #22384a;font-size:13.5px}}
.tx{{width:80%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 12px;font:600 12px Inter,sans-serif;cursor:pointer}}
button.p:hover{{border-color:#2fb8b8;color:#2fb8b8}}
button.p.on{{background:#2fb8b8;color:#0d1b26}}
</style></head><body><main>
<h1>Kecil-TTS <span>Gate 2</span> &mdash; nicole converted teacher clips</h1>
<p class="note">Chatterbox Malay clips after the single ChatterboxVC pass to af_nicole. 1,281 clips generated, 1,281
converted, 0 failures. Listen for watery/phase artefacts on vowels, word endings and /s/ before we train a student.
This is the audio the student will imitate.</p>
<table>{rows}</table>
<p class="note">Pass = clean enough to train. Fail = stop and try another reference/conversion before spending GPU.</p>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/nicole_teacher.html', 'w', encoding='utf-8').write(html)
shutil.copy('runs/listen/nicole_teacher.html', 'runs/local_v041/nicole_teacher.html')
print('page written:', len(ids), 'clips')
