"""Build the v0.4 model-repo samples from the C4 export itself.

Same selection logic build_hf.py uses (3 test rows it got right, 3 typical held-out rows), but with
the C4 model's own CERs from runs/k18_final/test_scores_c4.csv, rendered from the v0.4 ONNX.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
KD = os.path.join(ROOT, 'runs', 'k16_final')
SCORES = os.path.join(ROOT, 'runs', 'k18_final', 'test_scores_c4.csv')
MODEL = os.path.join(KD, 'ms_c4_int8.onnx')

import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

d = pd.read_csv(SCORES, dtype={'id': str})
d = d[d.model == 'c4']
rows = [dict(split='held' if r.category == 'held' else 'test', id=r.id, text=r.text, cer=r.cer)
        for r in d.itertuples()]

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
os.makedirs(os.path.join(KD, 'audio'), exist_ok=True)
out = []
for k, r in enumerate(rows, 1):
    ids = g2p.tokens(g2p(normalize(r['text'])))
    y = sess.run(None, {'input_ids': np.array([ids], dtype=np.int64),
                        'speed': np.ones(1, np.float32)})[0][0]
    sf.write(os.path.join(KD, 'audio', f"A_{r['split']}_{r['id']}.wav"), y, 24000)
    out.append(dict(model='A', split=r['split'], id=r['id'], text=r['text'], cer=r['cer']))
    if k % 50 == 0:
        print('rendered', k, '/', len(rows), flush=True)
pd.DataFrame(out).to_csv(os.path.join(KD, 'v5_scores.csv'), index=False)
c = pd.DataFrame(out)
t, h = c[c.split == 'test'], c[c.split == 'held']
print('wrote', len(out), 'renders | test', len(t), '| held', len(h))
print('test median %.3f capped %.3f | held median %.3f capped %.3f' %
      (t.cer.median(), t.cer.mean(), h.cer.median(), h.cer.mean()))

# brightness / loudness comparison against v0.3 on the same sentences (no librosa: plain FFT)
def bands(y, sr=24000):
    Y = np.abs(np.fft.rfft(y.astype(np.float64)))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    p = Y ** 2
    return float(p[f > 4000].sum() / max(p.sum(), 1e-12)), float(np.sqrt(np.mean(y ** 2)))

for name, path in (('v0.3', os.path.join(ROOT, 'runs', 'k14_final', 'ms_clean_webgpu_int8.onnx')),
                   ('v0.4', MODEL)):
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    hi, rms = [], []
    for r in rows[:24]:
        ids = g2p.tokens(g2p(normalize(r['text'])))
        y = s.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]
        a, b = bands(y)
        hi.append(a)
        rms.append(b)
    print(f'{name}: energy above 4 kHz {np.mean(hi):.4f} | rms {np.mean(rms):.4f}')
