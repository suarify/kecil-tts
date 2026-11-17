"""Pull sm-09-vc15b outputs and build runs/listen/vc15b.html: raw vs 1 VC vs 2 VCs on the same clips."""
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
api.kernels_output('suarifymalaysia/sm-09-vc15b', path='runs/kaggle_sm09')

src = glob.glob('runs/kaggle_sm09/**/vc15b.zip', recursive=True)
if not src:
    raise SystemExit('vc15b.zip not found')
OUT = 'runs/listen/vc15b'
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


import csv as _csv  # noqa: E402
texts = {r['id']: r['text'] for r in _csv.DictReader(open('runs/sm_vc15b_flat/texts.csv', encoding='utf-8'))}

ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{OUT}/single/*.wav'))
met = {'raw': [], 'single': [], 'old': []}
rows = []
for i in ids:
    files = {}
    for k in ('raw', 'single', 'old'):
        p = f'{OUT}/{k}/{i}.wav'
        if os.path.exists(p):
            y, _ = sf.read(p)
            files[k] = f'vc15b/{k}/{i}.wav'
            met[k].append([periodicity(y), *texture(y)])
    if files:
        rows.append((i, texts.get(i, i), files))

COLS = (('raw', 'raw Chatterbox (Malay prompt)', '#8a6cc8'),
        ('old', 'old: 2 conversions', '#e05c5c'),
        ('single', 'new: 1 conversion', '#5aa86c'))
btns = lambda fs: ''.join(
    f'<button class="p" style="--c:{c}" data-src="{fs[k]}">&#9654;<span class="l"> {lbl}</span></button>'
    for k, lbl, c in COLS if k in fs)
body = ''.join(f'<tr><td class="tx"><span class="cat">{i}</span> {t}</td><td class="bs">{btns(f)}</td></tr>' for i, t, f in rows)
agg = {k: [round(float(x), 3) for x in np.mean(v, 0)] for k, v in met.items() if v}
mrows = ''.join(f'<tr><td>{k}</td><td>{v[0]:.3f}</td><td>{v[1]:.3f}</td><td>{v[2]:.3f}</td><td>{v[3]:.3f}</td></tr>'
                for k, v in agg.items())

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Michael POC 2: one voice conversion vs two (same raw clips)</title>
<style>
body{{font:15px/1.5 Inter,'Segoe UI',sans-serif;background:#0d1b26;color:#dcebf4;margin:0;padding:32px 16px 60px}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-family:Sora,Inter,sans-serif;font-size:22px}} h1 span{{color:#5fd6d6}}
p.note{{color:#9db1bf;font-size:13.5px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{padding:9px 8px;border-top:1px solid #22384a;vertical-align:top;text-align:left;font-size:13.5px}}
th{{color:#9db1bf;border-top:0}}
.tx{{width:42%}} .cat{{font-size:11px;color:#9db1bf;border:1px solid #22384a;border-radius:99px;padding:1px 8px;margin-right:6px}}
.bs{{white-space:nowrap}}
button.p{{background:#12242f;border:1px solid #2c4a5e;color:#dcebf4;border-radius:8px;padding:7px 10px;font:600 12px Inter,sans-serif;cursor:pointer;margin:2px 3px 2px 0}}
button.p:hover{{border-color:var(--c);color:var(--c)}}
button.p.on{{background:var(--c);color:#0d1b26}}
table.m td{{padding:5px 12px;font-size:13px}}
</style></head><body><main>
<h1>Kecil-TTS <span>Michael POC 2</span> &mdash; one voice conversion vs two, same raw clips</h1>
<p class="note">Same raw Chatterbox clip (Malay prompt), three versions: <b>raw</b> = before any conversion
(female voice), <b>old</b> = the current two-step chain (raw &rarr; female reference &rarr; am_michael),
<b>new</b> = one step (raw &rarr; am_michael). Pronunciation is identical in all three, so any difference is the
conversion chain itself. Includes the III/atau sentence (c0571).</p>
<table>{body}</table>
<h2 style="font-family:Sora,Inter,sans-serif;font-size:17px">Averages over these {len(rows)} clips</h2>
<table class="m"><tr><th>source</th><th>periodicity</th><th>flat 2-6k</th><th>&gt;4 kHz</th><th>&gt;7 kHz</th></tr>{mrows}</table>
<p class="note">If <b>new</b> is cleaner than <b>old</b> and keeps the pronunciation, converting the 1,281 raw clips
once (about 45-60 min GPU) and retraining is worthwhile. If they sound equal, we stop here and keep v0.3.0.</p>
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
open('runs/listen/vc15b.html', 'w', encoding='utf-8').write(html)
dst = 'runs/local_v041/vc15b'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(OUT, dst)
shutil.copy('runs/listen/vc15b.html', 'runs/local_v041/vc15b.html')
print('page written:', len(rows), 'clips |', agg)
