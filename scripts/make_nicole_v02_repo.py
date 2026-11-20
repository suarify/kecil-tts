"""Assemble the Suarify/kecil-tts-nicole-v0.2 upload folder (new Nicole, +25% gain)."""
import json
import os
import re
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ONNX = 'runs/nicole_v02_gain_export/paradee_int8_webgpu.onnx'
REPO = 'Suarify/kecil-tts-nicole-v0.2'
OUT = 'runs/nicole_v020_repo'
assert os.path.exists(ONNX)
import configparser
cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
for name in list_repo_files('Suarify/kecil-tts-nicole-v0.1'):
    if name.endswith('.onnx') or name.startswith('dist/') or name.startswith('samples/'):
        continue
    p = hf_hub_download('Suarify/kecil-tts-nicole-v0.1', name, local_dir=OUT)
    if os.path.abspath(p) != os.path.abspath(os.path.join(OUT, name)):
        shutil.copy(p, os.path.join(OUT, name))
os.makedirs(f'{OUT}/onnx', exist_ok=True)
shutil.copy(ONNX, f'{OUT}/onnx/model_quantized.onnx')

c = json.load(open(f'{OUT}/config.json', encoding='utf-8'))
c['voices'] = ['nicole']
c['note'] = ('Single voice (female, Nicole). Built from Yati v0.5 clips via one ChatterboxVC pass, '
             'loudness-normalized, then text + decoder + short GAN. Output volume +25%. '
             'Inputs: input_ids (int64 [1,T]) and speed (float32 [1]).')
json.dump(c, open(f'{OUT}/config.json', 'w', encoding='utf-8'), indent=1)

# rebuild wheel from the renamed local package
tmp = 'runs/nicole_v02_wheel'
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree('python', f'{tmp}/python',
                ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
p = f'{tmp}/python/kecil_tts/tts.py'
s = open(p, encoding='utf-8').read()
s = re.sub(r"REPO\s*=\s*['\"][^'\"]+['\"]", f"REPO = '{REPO}'", s)
open(p, 'w', encoding='utf-8').write(s)
p = f'{tmp}/python/pyproject.toml'
s = open(p, encoding='utf-8').read()
s = re.sub(r'version = "[^"]+"', 'version = "0.2.0"', s)
open(p, 'w', encoding='utf-8').write(s)
os.makedirs(f'{OUT}/dist', exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'wheel', f'{tmp}/python', '--no-deps', '-w', f'{OUT}/dist', '-q'],
               check=True, env=dict(os.environ, PYTHONUTF8='1'))
print('wheel:', os.listdir(f'{OUT}/dist'))

# samples
import numpy as np
import onnxruntime as ort
import soundfile as sf
sys.path.insert(0, 'g2p')
from ms_g2p import G2P  # noqa
from ms_norm import normalize  # noqa
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
SENTS = ['Selamat pagi, boleh saya bantu encik hari ini?', 'Booking encik untuk hari Selasa dah confirmed.',
         'Kod pengesahan anda ialah 169484.', 'Pesanan Dato akan dihantar ke Shah Alam esok pagi.',
         'Sila maklumkan kehadiran puan sebelum hari Rabu.', 'Kami telah menerima aduan doktor dan sedang menyiasatnya.']
os.makedirs(f'{OUT}/samples', exist_ok=True)
lines = []
for k, t in enumerate(SENTS, 1):
    ids = np.array([g2p.tokens(g2p(normalize(t)))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f'{OUT}/samples/{k}_kecil_tts_nicole_v02.wav', y, 24000)
    lines.append(f'{k}\tmodel-rendered\t{t}')
open(f'{OUT}/samples/sentences.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('folder ready', OUT)
