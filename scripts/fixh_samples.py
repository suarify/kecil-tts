"""Regenerate the six publish sample wavs with the FIX H export (same sentences as v0.4's samples)."""
import os
import sys

os.environ['SUARADEE_MODEL'] = 'runs/k16_final/fixh_export/paradee_int8_webgpu.onnx'
sys.path.insert(0, 'python')
import soundfile as sf  # noqa: E402
import pandas as pd  # noqa: E402

from suaradee.tts import Suaradee  # noqa: E402

os.makedirs('runs/k16_final/fixh_export/audio', exist_ok=True)
d = pd.read_csv('runs/k16_final/v5_scores.csv')
t = Suaradee()
n = 0
for _, r in d.iterrows():
    fn = f'A_{r.split}_{r.id}.wav'
    if os.path.exists(os.path.join('runs/k16_final/audio', fn)):
        sf.write(f'runs/k16_final/fixh_export/audio/{fn}', t(r.text), 24000)
        n += 1
print('synthesized', n, 'files')
