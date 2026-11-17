"""Verify the FIX H export: texture metrics (periodicity / flatness 2-6 kHz / >4 kHz share) of the
WebGPU-patched int8 ONNX output on the ablation sentences vs the PyTorch FIX H wavs and the
Chatterbox targets. usage: python scripts/verify_fixh.py [onnx_path]"""
import csv
import json
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
items = json.load(open('runs/listen/ablate/items.json', encoding='utf-8'))
items = items['items'] if isinstance(items, dict) and 'items' in items else items
ids = list(items.keys()) if isinstance(items, dict) else [x['id'] for x in items]

model = sys.argv[1] if len(sys.argv) > 1 else 'runs/k16_final/fixh_export/paradee_int8_webgpu.onnx'
sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])

rows = {'target': [], 'fixH(torch)': [], 'fixH(ort)': []}
print(f'{"id":8s}  {"periodicity":>10s}  {"flat2-6k":>8s}  {" >4k":>6s}   (ort vs torch fixH vs target)', flush=True)
for i in ids[:6]:
    ids_t = np.array([g2p.tokens(g2p(normalize(plan[i])))], dtype=np.int64)
    y_ort = sess.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
    fn_t = f'runs/listen/ablate/{i}_fixH.wav'
    y_t, _ = sf.read(fn_t)
    y_tgt, _ = sf.read(f'runs/listen/ablate/{i}_target.wav')
    sf.write(f'runs/listen/ablate/{i}_ortfixH.wav', y_ort, SR)
    vals = [('target', y_tgt), ('fixH(torch)', y_t), ('fixH(ort)', y_ort)]
    for name, y in vals:
        p = periodicity(y)
        fl, o4 = texture(y)
        rows[name].append((p, fl, o4))
    print(f'{i:8s}  ort {rows["fixH(ort)"][-1][0]:.3f}/{rows["fixH(ort)"][-1][1]:.3f}/{rows["fixH(ort)"][-1][2]:.3f}   '
          f'torch {rows["fixH(torch)"][-1][0]:.3f}/{rows["fixH(torch)"][-1][1]:.3f}/{rows["fixH(torch)"][-1][2]:.3f}   '
          f'target {rows["target"][-1][0]:.3f}/{rows["target"][-1][1]:.3f}/{rows["target"][-1][2]:.3f}', flush=True)

print('\nmean over sentences (target ~0.699 / 0.621 / 0.080):')
for name, v in rows.items():
    m = np.mean(v, 0)
    print(f'  {name:12s} {m[0]:.3f}  {m[1]:.3f}  {m[2]:.3f}')
