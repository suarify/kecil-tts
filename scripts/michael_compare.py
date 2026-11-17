"""Compare Michael v0.2 (old ONNX) with the FIX H export on the same sentences: texture metrics + wavs."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

import numpy as np
import onnxruntime as ort
import soundfile as sf
from ms_g2p import G2P
from ms_norm import normalize
from vhead import SR


def periodicity(a):
    a = np.asarray(a, np.float32)
    a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480
    vals = []
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
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    over4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    return flat, over4


g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
plan = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = ['c0313', 'c0551', 'c0571', 'c0726', 'c0746', 'c0747']
old = ort.InferenceSession('runs/michael_v02.onnx', providers=['CPUExecutionProvider'])
new = ort.InferenceSession('runs/michael_fixh/paradee_int8_webgpu.onnx', providers=['CPUExecutionProvider'])
os.makedirs('runs/listen/michael', exist_ok=True)
res = {'old': [], 'new': []}
for i in ids:
    t_ids = np.array([g2p.tokens(g2p(normalize(plan[i])))], dtype=np.int64)
    feed = {'input_ids': t_ids, 'speed': np.ones(1, np.float32)}
    yo = old.run(None, feed)[0][0]
    yn = new.run(None, feed)[0][0]
    SF.write(f'runs/listen/michael/{i}_old.wav', yo, SR) if False else sf.write(f'runs/listen/michael/{i}_old.wav', yo, SR)
    sf.write(f'runs/listen/michael/{i}_fixh.wav', yn, SR)
    for name, y in (('old', yo), ('new', yn)):
        p = periodicity(y)
        fl, o4 = texture(y)
        res[name].append((p, fl, o4))
    print(i, f'old {res["old"][-1][0]:.3f}/{res["old"][-1][1]:.3f}/{res["old"][-1][2]:.3f}  new {res["new"][-1][0]:.3f}/{res["new"][-1][1]:.3f}/{res["new"][-1][2]:.3f}', flush=True)
print('\nmean (target texture from Chatterbox refs: 0.699 / 0.621 / 0.080)')
for name in ('old', 'new'):
    m = np.mean(res[name], 0)
    print(f'  {name:4s} {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}')
