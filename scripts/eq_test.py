"""Brightness correction for the k09 student, no training (CPU).

Measures the average spectrum of the student against (a) its own Chatterbox targets and (b) clean Kokoro
speech, builds a smooth fixed EQ curve for each, applies it to student clips of unseen sentences and
writes runs/listen/eq.html to compare by ear. A fixed EQ can be baked into the runtime for free.

Run:  python scripts/eq_test.py
"""
import glob
import json
import os
import shutil

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K9 = os.path.join(ROOT, 'runs', 'k09')
SR, NFFT, HOP = 24000, 1024, 256
F = np.fft.rfftfreq(NFFT, 1 / SR)


def ltas(files):
    acc, n = np.zeros(NFFT // 2 + 1), 0
    for f in files:
        a, sr = sf.read(f, dtype='float32')
        P = np.abs(librosa.stft(a, n_fft=NFFT, hop_length=HOP)) ** 2
        tot = P.sum(0)
        keep = tot > np.percentile(tot, 40)
        P = P[:, keep] / (P[:, keep].sum(0, keepdims=True) + 1e-12)     # each frame counts equally
        acc += P.sum(1)
        n += P.shape[1]
    return acc / n


def smooth_db(ratio, lo=300.0, cap=(-3.0, 10.0)):
    db = 10 * np.log10(ratio + 1e-12)
    out = np.zeros_like(db)
    for i, f in enumerate(F):                                            # third-octave smoothing
        if f < lo:
            continue
        band = (F >= f / 1.12) & (F <= f * 1.12)
        out[i] = db[band].mean()
    out -= out[(F > 300) & (F < 1000)].mean()                            # leave the low-mid level where it is
    out[F < lo] = 0
    return np.clip(out, *cap)


def apply_eq(a, gain_db):
    S = librosa.stft(a, n_fft=NFFT, hop_length=HOP)
    y = librosa.istft(S * (10 ** (gain_db / 20))[:, None], hop_length=HOP, length=len(a))
    peak = np.abs(y).max()
    return (y * min(1.0, 0.97 / peak)).astype(np.float32) if peak > 0 else y


stu = sorted(glob.glob(os.path.join(K9, 'audio', 'held_*_onnx_int8.wav')))
tgt = [f.replace('onnx_int8', 'target') for f in stu]
kok = sorted(glob.glob(os.path.join(ROOT, 'runs', 'listen', 'tp', 'F1', '*.wav')))
Ls, Lt, Lk = ltas(stu), ltas(tgt), ltas(kok)
eq_cb = smooth_db(Lt / Ls)             # student -> its Chatterbox target
eq_ko = smooth_db(Lk / Ls)             # student -> clean Kokoro brightness
for name, g in (('to Chatterbox', eq_cb), ('to Kokoro', eq_ko)):
    print(f'EQ {name:14s} gain dB at 1k {g[np.argmin(abs(F - 1000))]:+.1f}  2k {g[np.argmin(abs(F - 2000))]:+.1f}  3k {g[np.argmin(abs(F - 3000))]:+.1f}  '
          f'4k {g[np.argmin(abs(F - 4000))]:+.1f}  6k {g[np.argmin(abs(F - 6000))]:+.1f}  8k {g[np.argmin(abs(F - 8000))]:+.1f}  10k {g[np.argmin(abs(F - 10000))]:+.1f}')
np.save(os.path.join(K9, 'eq_to_chatterbox_db.npy'), eq_cb)
np.save(os.path.join(K9, 'eq_to_kokoro_db.npy'), eq_ko)

out = os.path.join(ROOT, 'runs', 'listen', 'eq')
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
sc = pd.read_csv(os.path.join(K9, 'e2e_scores.csv'))
test = sc[(sc.split == 'test') & (sc.cer_onnx <= 0.15)].head(8)
held = sc[(sc.split == 'held') & (sc.cer_onnx <= 0.15)].head(4)
rows = []
for r in pd.concat([test, held]).itertuples():
    a, _ = sf.read(os.path.join(K9, 'audio', f'{r.split}_{r.id}_onnx_int8.wav'), dtype='float32')
    names = {}
    for tag, y in (('plain', a), ('eq_cb', apply_eq(a, eq_cb)), ('eq_ko', apply_eq(a, eq_ko))):
        sf.write(os.path.join(out, f'{r.id}_{tag}.wav'), y, SR)
        names[tag] = f'eq/{r.id}_{tag}.wav'
    t = os.path.join(K9, 'audio', f'{r.split}_{r.id}_target.wav')
    if os.path.exists(t):
        shutil.copy(t, os.path.join(out, f'{r.id}_target.wav'))
        names['target'] = f'eq/{r.id}_target.wav'
    rows.append(dict(id=r.id, text=r.text, **names))
html = '''<!doctype html><html><head><meta charset="utf-8"><title>Brightness test</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc}@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1000px}h1{font-size:22px;margin:0 0 4px}p{color:var(--mut)}
.row{border-top:1px solid var(--line);padding:12px 0}.t{font-weight:600}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;margin-top:6px}.g div{font-size:13px;color:var(--mut)}audio{width:100%;height:32px}
</style></head><body><h1>Does a fixed tone correction remove the damp sound?</h1>
<p>Same 8M model output three ways. "Plain" is what you heard. The two corrected versions lift the upper frequencies by a fixed curve (no retraining). Tell Claude which you prefer, and whether the damp or phasey quality is gone or only brighter.</p><div id="l"></div>
<script>const R=__R__;const L=document.getElementById("l");const N={plain:"Plain (as before)",eq_cb:"Corrected to Chatterbox tone",eq_ko:"Corrected to Kokoro tone (brighter)",target:"Chatterbox clip (reference)"};
for(const r of R){const d=document.createElement("div");d.className="row";d.innerHTML=`<div class="t"></div><div class="g"></div>`;d.querySelector(".t").textContent=r.text;
for(const k of ["plain","eq_cb","eq_ko","target"]){if(!r[k])continue;const e=document.createElement("div");e.innerHTML=`${N[k]}<audio controls preload="none" src="${r[k]}"></audio>`;d.querySelector(".g").appendChild(e)}L.appendChild(d)}</script></body></html>'''
open(os.path.join(ROOT, 'runs', 'listen', 'eq.html'), 'w', encoding='utf-8').write(html.replace('__R__', json.dumps(rows, ensure_ascii=False)))
print(len(rows), 'sentences -> runs/listen/eq.html')
