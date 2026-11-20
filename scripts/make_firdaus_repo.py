"""Assemble the Firdaus v0.1 upload folder from the Yati v0.5 repo template (internal id: fenrir)."""
import json
import os
import re
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ONNX = 'runs/fenrir_gan_export/paradee_int8_webgpu.onnx'
REPO = 'Suarify/kecil-tts-firdaus-v0.1'
OUT = 'runs/firdaus_v010_repo'
assert os.path.exists(ONNX), ONNX

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
for name in list_repo_files('Suarify/kecil-tts-yati-v0.5.0'):
    if name.endswith('.onnx') or name.startswith('dist/') or name.startswith('samples/'):
        continue
    p = hf_hub_download('Suarify/kecil-tts-yati-v0.5.0', name, local_dir=OUT)
    if os.path.abspath(p) != os.path.abspath(os.path.join(OUT, name)):
        shutil.copy(p, os.path.join(OUT, name))
os.makedirs(f'{OUT}/onnx', exist_ok=True)
shutil.copy(ONNX, f'{OUT}/onnx/model_quantized.onnx')

c = json.load(open(f'{OUT}/config.json', encoding='utf-8'))
c['voices'] = ['firdaus']
c['note'] = ('Single voice (male, Firdaus). Path B from Yati v0.5 clips converted once to the am_fenrir '
             'reference, then text + decoder + short GAN trained from Yati v0.5 weights. '
             'Inputs: input_ids (int64 [1,T], phoneme ids with 0 at both ends, T <= 402) and speed (float32 [1]).')
json.dump(c, open(f'{OUT}/config.json', 'w', encoding='utf-8'), indent=1)

for p in (f'{OUT}/kecil_tts/tts.py', f'{OUT}/js/suaradee.js', f'{OUT}/examples/python_example.py',
          f'{OUT}/examples/node_example.mjs'):
    if not os.path.exists(p):
        continue
    s = open(p, encoding='utf-8').read().replace('Suarify/kecil-tts-yati-v0.5.0', REPO)
    open(p, 'w', encoding='utf-8').write(s)

tmp = 'runs/firdaus_wheel_build'
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree('python', f'{tmp}/python', ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
s = open(f'{tmp}/python/suaradee/tts.py', encoding='utf-8').read()
s = re.sub(r"REPO = 'Suarify/kecil-tts-[^']+'", f"REPO = '{REPO}'", s)
open(f'{tmp}/python/suaradee/tts.py', 'w', encoding='utf-8').write(s)
p = f'{tmp}/python/pyproject.toml'
s = open(p, encoding='utf-8').read()
s = re.sub(r'version = "[^"]+"', 'version = "0.1.0"', s)
open(p, 'w', encoding='utf-8').write(s)
os.makedirs(f'{OUT}/dist', exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'wheel', f'{tmp}/python', '--no-deps', '-w', f'{OUT}/dist', '-q'],
               check=True, env=dict(os.environ, PYTHONUTF8='1'))
print('wheel:', os.listdir(f'{OUT}/dist'))

import numpy as np
import onnxruntime as ort
import soundfile as sf
sys.path.insert(0, 'g2p')
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
SENTS = ['Selamat pagi, boleh saya bantu encik hari ini?', 'Booking encik untuk hari Selasa dah confirmed.',
         'Kod pengesahan anda ialah 169484.', 'Pesanan Dato akan dihantar ke Shah Alam esok pagi.',
         'Sila maklumkan kehadiran puan sebelum hari Rabu.', 'Kami telah menerima aduan doktor dan sedang menyiasatnya.']
os.makedirs(f'{OUT}/samples', exist_ok=True)
lines = []
for k, text in enumerate(SENTS, 1):
    ids = np.array([g2p.tokens(g2p(normalize(text)))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f'{OUT}/samples/{k}_kecil_tts_firdaus.wav', y, 24000)
    lines.append(f'{k}\tmodel-rendered\t{text}')
open(f'{OUT}/samples/sentences.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('folder ready:', OUT)
