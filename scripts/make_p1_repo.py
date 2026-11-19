"""Assemble the Suarify/kecil-tts-yati-v0.5.0 (P1) upload folder from the v0.4.1 HF repo.

usage: python scripts/make_p1_repo.py runs/sota_p1_export/paradee_int8_webgpu.onnx
"""
import json
import os
import re
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ONNX = sys.argv[1] if len(sys.argv) > 1 else 'runs/sota_p1_export/paradee_int8_webgpu.onnx'
REPO = 'Suarify/kecil-tts-yati-v0.5.0'
OUT = 'runs/p1_v050_repo'
assert os.path.exists(ONNX), ONNX

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
for name in list_repo_files('Suarify/kecil-tts-yati-v0.4.1'):
    if name.endswith('.onnx') or name.startswith('dist/'):
        continue
    p = hf_hub_download('Suarify/kecil-tts-yati-v0.4.1', name, local_dir=OUT)
    if os.path.abspath(p) != os.path.abspath(os.path.join(OUT, name)):
        shutil.copy(p, os.path.join(OUT, name))
os.makedirs(f'{OUT}/onnx', exist_ok=True)
shutil.copy(ONNX, f'{OUT}/onnx/model_quantized.onnx')

c = json.load(open(f'{OUT}/config.json', encoding='utf-8'))
c['note'] = ('Single voice (female, Yati). REVIEW Phase 1: log-F0 + voiced/unvoiced prediction on the text side; '
             'FIX H export unchanged. Inputs: input_ids (int64 [1,T], phoneme ids with 0 at both ends, T <= 402) '
             'and speed (float32 [1]). Phonemes come from the Malay front end in kecil_tts/ (Python) or js/ (JavaScript).')
json.dump(c, open(f'{OUT}/config.json', 'w', encoding='utf-8'), indent=1)

# point packaged defaults at the new repo
for p in (f'{OUT}/kecil_tts/tts.py', f'{OUT}/js/suaradee.js', f'{OUT}/examples/python_example.py',
          f'{OUT}/examples/node_example.mjs'):
    if not os.path.exists(p):
        continue
    s = open(p, encoding='utf-8').read()
    s = s.replace('Suarify/kecil-tts-yati-v0.4.1', REPO)
    s = s.replace('kecil-tts-yati-v0.4.1', REPO)
    open(p, 'w', encoding='utf-8').write(s)

# wheel with the new default repo
tmp = 'runs/p1_wheel_build'
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree('python', f'{tmp}/python', ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
s = open(f'{tmp}/python/suaradee/tts.py', encoding='utf-8').read()
s = re.sub(r"REPO = 'Suarify/kecil-tts-[^']+'", f"REPO = '{REPO}'", s)
open(f'{tmp}/python/suaradee/tts.py', 'w', encoding='utf-8').write(s)
p = f'{tmp}/python/pyproject.toml'
s = open(p, encoding='utf-8').read()
s = re.sub(r'version = "[^"]+"', 'version = "0.5.0"', s)
open(p, 'w', encoding='utf-8').write(s)
os.makedirs(f'{OUT}/dist', exist_ok=True)
env = dict(os.environ, PYTHONUTF8='1')
subprocess.run([sys.executable, '-m', 'pip', 'wheel', f'{tmp}/python', '--no-deps', '-w', f'{OUT}/dist', '-q'],
               check=True, env=env)
print('wheel:', os.listdir(f'{OUT}/dist'))

# model-rendered samples
import numpy as np
import onnxruntime as ort
import soundfile as sf
sys.path.insert(0, 'g2p')
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
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
    sf.write(f'{OUT}/samples/{k}_kecil_tts_yati_v050.wav', y, 24000)
    lines.append(f'{k}\tmodel-rendered\t{text}')
    print(k, round(len(y) / 24000, 2), 's')
open(f'{OUT}/samples/sentences.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('folder ready:', OUT)
