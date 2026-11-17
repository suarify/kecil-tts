"""Generate Michael v0.2.1 repo samples with the FIX H export and assemble runs/michael_v021 upload folder."""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
os.chdir(ROOT)

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

OUT = 'runs/michael_v021'
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree('runs/michael_repo_src', OUT)
os.makedirs(f'{OUT}/onnx', exist_ok=True)
shutil.copy('runs/michael_fixh/paradee_int8_webgpu.onnx', f'{OUT}/onnx/model_quantized.onnx')

sess = ort.InferenceSession(f'{OUT}/onnx/model_quantized.onnx', providers=['CPUExecutionProvider'])
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
lines = open(f'{OUT}/samples/sentences.txt', encoding='utf-8').read().strip().split('\n')
os.makedirs(f'{OUT}/samples', exist_ok=True)
for ln in lines:
    k, split, cer, text = ln.split('\t')
    fn = f'{OUT}/samples/{k}_kecil_tts_michael.wav'
    ids = np.array([g2p.tokens(g2p(normalize(text)))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(fn, y, 24000)
    print(fn, round(len(y) / 24000, 2), 's')
print('assembled', OUT)
