"""Assemble the Suarify/kecil-tts-nicole-v0.1 upload folder from the michael repo template."""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
SRC = 'runs/michael_v031_repo'
OUT = 'runs/nicole_v010_repo'
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(SRC, OUT)
shutil.rmtree(f'{OUT}/.cache', ignore_errors=True)
shutil.rmtree(f'{OUT}/dist', ignore_errors=True)

# ONNX: the full 12k Path-B student, webgpu-patched
shutil.copy('runs/nicole_full_export/paradee_int8_webgpu.onnx', f'{OUT}/onnx/model_quantized.onnx')

# config: female voice name
c = json.load(open(f'{OUT}/config.json', encoding='utf-8'))
c['voices'] = ['nicole']
c['note'] = ('Single voice (female, af_nicole timbre). Inputs: input_ids (int64 [1,T], phoneme ids with 0 at '
             'both ends, T <= 402) and speed (float32 [1]). Phonemes come from the Malay front end in kecil_tts/ '
             '(Python) or js/ (JavaScript).')
json.dump(c, open(f'{OUT}/config.json', 'w', encoding='utf-8'), indent=1)

# package default repo ID: point to the new repo
for p in (f'{OUT}/kecil_tts/tts.py', f'{OUT}/js/suaradee.js', f'{OUT}/examples/python_example.py',
          f'{OUT}/examples/node_example.mjs'):
    if not os.path.exists(p):
        continue
    s = open(p, encoding='utf-8').read()
    s = s.replace('Suarify/kecil-tts-michael-v0.3.1', 'Suarify/kecil-tts-nicole-v0.1')
    s = s.replace('kecil-tts-michael-v0.3.1', 'kecil-tts-nicole-v0.1')
    open(p, 'w', encoding='utf-8').write(s)

# build a wheel with the new default repo (version 0.1)
import subprocess
tmp = 'runs/nicole_wheel_build'
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree('python', f'{tmp}/python', ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
s = open(f'{tmp}/python/suaradee/tts.py', encoding='utf-8').read()
s = s.replace("REPO = 'Suarify/kecil-tts-yati-v0.4.1'", "REPO = 'Suarify/kecil-tts-nicole-v0.1'")
s = s.replace("REPO = 'Suarify/kecil-tts-yati-v0.4'", "REPO = 'Suarify/kecil-tts-nicole-v0.1'")
open(f'{tmp}/python/suaradee/tts.py', 'w', encoding='utf-8').write(s)
p = f'{tmp}/python/pyproject.toml'
s = open(p, encoding='utf-8').read()
import re
s = re.sub(r'version = "[^"]+"', 'version = "0.1.0"', s)
open(p, 'w', encoding='utf-8').write(s)
os.makedirs(f'{OUT}/dist', exist_ok=True)
env = dict(os.environ, PYTHONUTF8='1')
subprocess.run([sys.executable, '-m', 'pip', 'wheel', f'{tmp}/python', '--no-deps', '-w', f'{OUT}/dist', '-q'],
               check=True, env=env)
print('wheel:', os.listdir(f'{OUT}/dist'))

# samples rendered by the model itself (6 sentences from the Path-B / teacher set)
import numpy as np
import onnxruntime as ort
import soundfile as sf
sys.path.insert(0, 'g2p')
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession(f'{OUT}/onnx/model_quantized.onnx', providers=['CPUExecutionProvider'])
SENTS = ['Selamat pagi, boleh saya bantu encik hari ini?',
         'Booking encik untuk hari Selasa dah confirmed.',
         'Kod pengesahan anda ialah 169484.',
         'Pesanan Dato akan dihantar ke Shah Alam esok pagi.',
         'Sila maklumkan kehadiran puan sebelum hari Rabu.',
         'Kami telah menerima aduan doktor dan sedang menyiasatnya.']
os.makedirs(f'{OUT}/samples', exist_ok=True)
lines = []
for k, text in enumerate(SENTS, 1):
    ids = np.array([g2p.tokens(g2p(normalize(text)))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f'{OUT}/samples/{k}_kecil_tts_nicole.wav', y, 24000)
    lines.append(f'{k}\tmodel-rendered\t{text}')
    print(k, round(len(y) / 24000, 2), 's')
open(f'{OUT}/samples/sentences.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('folder ready:', OUT)
