# sm 09 vc15b: one voice conversion vs two, on the SAME raw Chatterbox clips.
#   old    = raw -> ChatterboxVC(female ref) -> ChatterboxVC(am_michael)   (2 conversions, current training data)
#   single = raw -> ChatterboxVC(am_michael)                                (1 conversion)
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

import soundfile as sf

OUT = '/kaggle/working'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
        raise SystemExit(f'FAILED {cmd}')
    return p.stdout


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile')
sh('pip uninstall -y -q torchvision')          # broken in this image, not needed
sh('pip install -q "transformers==4.46.3"')
sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC; print(\'vc import ok\')"')

import pandas as pd  # noqa: E402
import torch  # noqa: E402

REF = glob.glob('/kaggle/input/**/ref_michael.wav', recursive=True)[0]
TEXTS = glob.glob('/kaggle/input/**/texts.csv', recursive=True)[0]
RAWDIR = os.path.dirname(glob.glob('/kaggle/input/**/raw_*.wav', recursive=True)[0])
plan = pd.read_csv(TEXTS)
print('clips:', len(plan), '| ref:', REF, flush=True)

os.makedirs(f'{OUT}/single', exist_ok=True)
os.makedirs(f'{OUT}/old', exist_ok=True)
os.makedirs(f'{OUT}/raw', exist_ok=True)
for i in plan.id:
    for sub, pref in (('old', 'old_'), ('raw', 'raw_')):
        p = f'{RAWDIR}/{pref}{i}.wav'
        if os.path.exists(p):
            shutil.copy(p, f'{OUT}/{sub}/{i}.wav')

from chatterbox.vc import ChatterboxVC  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"')
vc = ChatterboxVC.from_pretrained('cuda')
t0 = time.time()
n = 0
for r in plan.itertuples():
    src = f'{OUT}/raw/{r.id}.wav'
    if not os.path.exists(src):
        print('missing raw', r.id, flush=True)
        continue
    try:
        w = vc.generate(audio=src, target_voice_path=REF)
    except Exception as e:
        print('FAIL', r.id, type(e).__name__, repr(e)[:120], flush=True)
        continue
    a = w.squeeze(0).float().cpu().numpy()
    sf.write(f'{OUT}/single/{r.id}.wav', a, 24000)
    n += 1
    print(f'{r.id} {len(a) / 24000:.2f}s | {(time.time() - t0) / 60:.1f} min', flush=True)

with zipfile.ZipFile(f'{OUT}/vc15b.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for sub in ('single', 'old', 'raw'):
        for w in glob.glob(f'{OUT}/{sub}/*.wav'):
            z.write(w, f'{sub}/{os.path.basename(w)}')
print('DONE single', n, '| zip MB', round(os.path.getsize(f'{OUT}/vc15b.zip') / 1e6, 1), flush=True)
