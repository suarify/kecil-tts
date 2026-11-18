"""Path B dataset: Yati v0.4.1 student wavs for 8 sentences + nicole reference."""
import os
import shutil

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = 'runs/sm15b_in'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

rows = torch.load('runs/kaggle_sm14b/e2e/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
texts = {r['id']: r['text'] for r in rows}
ids = ['c0184', 'c0569', 'c0075', 'c0014', 'c0240', 'c0192', 'c0487', 'c0467']

import sys
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession('runs/k16_final/fixh_export/paradee_int8_webgpu.onnx', providers=['CPUExecutionProvider'])
for i in ids:
    ids_t = np.array([g2p.tokens(g2p(normalize(texts[i])))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f'{OUT}/yati_{i}.wav', y, 24000)
    print(i, round(len(y) / 24000, 2), 's')

shutil.copy('runs/sm12_in/ref_nicole.wav', f'{OUT}/ref_nicole.wav')
with open(f'{OUT}/ids.txt', 'w') as f:
    f.write('\n'.join(ids))
import json
json.dump({'title': 'sm15b in', 'id': 'suarifymalaysia/sm15b-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
print('dataset files:', sorted(os.listdir(OUT)))
