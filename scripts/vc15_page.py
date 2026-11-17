"""Pull the sm-08-vc15 outputs and build runs/listen/vc15.html: single-conversion vs double-conversion clips."""
import glob
import json
import os
import shutil
import zipfile

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

api = KaggleApi(); api.authenticate()
api.kernels_output('suarifymalaysia/sm-08-vc15', path='runs/kaggle_sm08')

src = glob.glob('runs/kaggle_sm08/**/vc15.zip', recursive=True)
if not src:
    raise SystemExit('vc15.zip not found in the kernel output')
OUT = 'runs/listen/vc15'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
with zipfile.ZipFile(src[0]) as z:
    z.extractall(OUT)

SR = 24000


def periodicity(a):
    a = np.asarray(a, np.float32); a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480; vals = []
    for i in range(0, len(a) - win, hop):
        fr_ = a[i:i + win]
        if np.sqrt((fr_ ** 2).mean()) < 0.02:
            continue
        fr_ = fr_ - fr_.mean()
        ac = np.correlate(fr_, fr_, 'full')[win - 1:]
        ac /= ac[0] + 1e-12
        vals.append(float(ac[int(SR / 200):int(SR / 80)].max()))
    return float(np.median(vals)) if vals else 0.0


def texture(a):
    import librosa
    S = np.abs(librosa.stft(np.asarray(a, np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2; tot = P.sum(0); keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    o4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    o7 = float((P[f > 7000][:, keep].sum(0) / tot[keep]).mean())
    return flat, o4, o7


texts = {}
csvp = glob.glob('runs/sm_vc15_in/texts.csv')
if csvp:
    import csv as _csv
    for r in _csv.DictReader(open(csvp[0], encoding='utf-8')):
        texts[r['id']] = r['text']

ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{OUT}/single/*.wav'))
rows, met = [], {'single': [], 'old': []}
for i in ids:
    files = {}
    for k in ('single', 'old'):
        p = f'{OUT}/{k}/{i}.wav'
        if os.path.exists(p):
            y, _ = sf.read(p)
            files[k] = f'vc15/{k}/{i}.wav'
            met[k].append([periodicity(y), *texture(y)])
    if files:
        rows.append((i, texts.get(i, i), files))

btns = lambda fs: ''.join(
    f'<button class="p" style="--c:{c}" data-src="{fs[k]}">&#9654;<span class="l"> {lbl}</span></button>'
    for k, lbl, c in (('old', 'old: 2 conversions', '#e05c5c'), ('single', 'new: 1 conversion', '#5aa86c')) if k in fs)
body = ''.join(f'<tr><td class="tx"><span class="cat">{i}</span> {t}</td><td class="bs">{btns(f)}</td></tr>' for i, t, f in rows)

agg = {k: [round(float(x), 3) for x in np.mean(v, 0)] for k, v in met.items() if v}
mrows = ''.join(f'<tr><td>{k}</td><td>{v[0]:.3f}</td><td>{v[1]:.3f}</td><td>{v[2]:.3f}</td><td>{v[3]:.3f}</td></tr>'
                for k, v in agg.items())

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Michael POC: one voice conversion vs two</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1100px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:48%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael POC</span> &mdash; one voice conversion vs two</h1>
<p class="note">These are raw Chatterbox clips, NOT the student model. <b>old</b> = the pipeline used for training so
far (Chatterbox Malay with a female prompt, then ChatterboxVC to the male voice = two conversions).
<b>new</b> = Chatterbox with the Kokoro <code>am_michael</code> prompt directly (one conversion). If the new clips are
clearly cleaner, regenerating all 1,281 texts this way is worth the training run.</p>
<table>{body}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(rows)} clips</h2>
<table class="m"><tr><th>source</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
<p class="note">Listen for the damp/watery texture and for how natural the vowels are. Same texts, so the comparison
is direct.</p>
</main>
<script>
let cur=null;
document.addEventListener('click',e=>{{
  const b=e.target.closest('button.p'); if(!b)return;
  if(cur&&cur!==b){{cur.pause&&cur.pause();cur.classList.remove('on');}}
  if(cur===b&&cur.pause===undefined)return;
  cur=b; b.classList.add('on');
  const a=new Audio(b.dataset.src); a.onended=()=>b.classList.remove('on'); a.play(); cur=a;
}});
</script></body></html>"""
open('runs/listen/vc15.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/vc15'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(OUT, dst)
shutil.copy('runs/listen/vc15.html', 'runs/local_v041/vc15.html')
print('page written:', len(rows), 'clips |', agg)
