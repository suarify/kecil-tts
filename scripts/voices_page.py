"""Build runs/listen/voices.html: Kokoro voice candidates for a possible third Kecil voice."""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'runs/listen/voices'
VOICES = [('af_heart', 'American female, warm'), ('af_bella', 'American female, clear'),
          ('af_nicole', 'American female, soft/breathy'), ('af_sarah', 'American female, neutral'),
          ('bf_emma', 'British female'), ('am_puck', 'American male, light'),
          ('am_fenrir', 'American male, deep'), ('bm_george', 'British male')]

rows = ''
for v, desc in VOICES:
    p = f'{SRC}/{v}.wav'
    if not os.path.exists(p):
        continue
    rows += (f'<tr><td class="tx"><b>{v}</b><br><span class="cat">{desc}</span></td>'
             f'<td><button class="p" data-src="voices/{v}.wav">&#9654;<span class="l"> play</span></button></td></tr>')

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Candidate voices for a third Kecil voice</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:900px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td{{padding:10px 8px;border-top:1px solid #22384a;vertical-align:top;font-size:14px}}
.tx{{width:70%}} .cat{{font-size:12px;color:#9db1bf}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:8px 14px;font:600 12px Inter,sans-serif;cursor:pointer}}
button.p:hover{{border-color:#2fb8b8;color:#2fb8b8}}
button.p.on{{background:#2fb8b8;color:#0d1b26}}
</style></head><body><main>
<h1>Candidate voices for a <span>third Kecil voice</span></h1>
<p class="note">These are the Kokoro voices themselves (English sample). If you pick one, the pipeline is: take the
1,281 raw Chatterbox Malay clips, convert them ONCE to this voice (ChatterboxVC), rebuild the rows, train the 8M
student, export. Female voices should come out clearer than Michael for the reasons measured: higher pitch spacing
and a single conversion.</p>
<table>{rows}</table>
<p class="note">Tell me the voice name and I'll run a 15-clip Malay preview of it first (about 6 minutes of GPU)
before committing to the full ~1.5 hour build.</p>
</main>
<script>
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  document.querySelectorAll('button.p.on').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play();
}});
</script></body></html>"""
open('runs/listen/voices.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/voices'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(SRC, dst)
shutil.copy('runs/listen/voices.html', 'runs/local_v041/voices.html')
print('voices page written')
