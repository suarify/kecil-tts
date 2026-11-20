"""Generate all 1,281 Yati v0.5 student clips locally for the Fenrir Path-B pipeline."""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

OUT = 'runs/pathb_yati_v05_clips'
os.makedirs(OUT, exist_ok=True)
rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('rows:', len(rows), flush=True)
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession('runs/p3_dec_gan_export/paradee_int8_webgpu.onnx', providers=['CPUExecutionProvider'])

t0 = time.time()
for idx, r in enumerate(rows):
    p = f'{OUT}/{r["id"]}.wav'
    if os.path.exists(p):
        continue
    ids_t = np.array([g2p.tokens(g2p(normalize(r['text'])))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(p, y, 24000)
    if idx % 100 == 99:
        print(idx + 1, '/', len(rows), '|', round((time.time() - t0) / 60, 1), 'min', flush=True)
print('DONE', len(os.listdir(OUT)), 'clips |', round((time.time() - t0) / 60, 1), 'min', flush=True)
