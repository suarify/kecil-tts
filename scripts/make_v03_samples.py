"""Build the v0.3 model-repo sample pool from the k14 clean export itself.

Renders every scored row (200 test + 60 held) with runs/k14_final/ms_clean_int8.onnx, writing
runs/k14_final/audio/A_<split>_<id>.wav and v5_scores.csv (model='A', k14 CERs). build_hf.py then
selects the 6 published samples exactly as it does for v0.2 (3 test rows it got right, 3 typical
held-out rows) without ever reusing v0.2 audio.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
KD = os.path.join(ROOT, 'runs', 'k14_final')

import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

d = pd.read_csv(os.path.join(KD, 'test_scores.csv'), dtype={'id': str})
rows = []
for r in d[d.model == 'clean'].itertuples():
    rows.append(dict(split='test', id=r.id, text=r.text, cer=r.cer))
for r in d[d.model == 'clean_held'].itertuples():
    rows.append(dict(split='held', id=r.id, text=r.text, cer=r.cer))

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = ort.InferenceSession(os.path.join(KD, 'ms_clean_int8.onnx'), so, providers=['CPUExecutionProvider'])
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
print('wrote', len(out), 'renders | test', len(c[c.split == 'test']), '| held', len(c[c.split == 'held']))
print('test median CER %.3f | held median %.3f' % (c[c.split == 'test'].cer.median(), c[c.split == 'held'].cer.median()))
print('test rows at or under 5%%: %d' % (c[(c.split == 'test') & (c.cer <= 0.05)].shape[0]))
