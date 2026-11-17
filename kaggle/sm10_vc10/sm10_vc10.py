# sm 10 vc10: conversion-quality check on 10 raw clips with TWO am_michael references
#   a = Kokoro-generated reference (25 s, ours)
#   b = ref_en_michael.wav used by the original pipeline
# Goal: hear whether the conversion itself introduces watery/phase before spending quota on the full set.
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
sh('pip uninstall -y -q torchvision')
sh('pip install -q "transformers==4.46.3"')
sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC; print(\'vc import ok\')"')

import pandas as pd  # noqa: E402

plan = pd.read_csv(glob.glob('/kaggle/input/**/texts.csv', recursive=True)[0])
REF_A = glob.glob('/kaggle/input/**/ref_a_michael.wav', recursive=True)[0]
REF_B = glob.glob('/kaggle/input/**/ref_b_michael.wav', recursive=True)[0]
RAWDIR = os.path.dirname(glob.glob('/kaggle/input/**/raw_*.wav', recursive=True)[0])
print('clips:', len(plan), '| refs:', REF_A, REF_B, flush=True)

for sub in ('raw', 'a', 'b'):
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)
for i in plan.id:
    p = f'{RAWDIR}/raw_{i}.wav'
    if os.path.exists(p):
        shutil.copy(p, f'{OUT}/raw/{i}.wav')

from chatterbox.vc import ChatterboxVC  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"')
vc = ChatterboxVC.from_pretrained('cuda')
t0 = time.time()
for ref_name, ref in (('a', REF_A), ('b', REF_B)):
    for r in plan.itertuples():
        src = f'{OUT}/raw/{r.id}.wav'
        if not os.path.exists(src):
            continue
        try:
            w = vc.generate(audio=src, target_voice_path=ref)
        except Exception as e:
            print('FAIL', ref_name, r.id, type(e).__name__, repr(e)[:120], flush=True)
            continue
        a = w.squeeze(0).float().cpu().numpy()
        sf.write(f'{OUT}/{ref_name}/{r.id}.wav', a, 24000)
        print(f'{ref_name} {r.id} {len(a) / 24000:.2f}s | {(time.time() - t0) / 60:.1f} min', flush=True)

with zipfile.ZipFile(f'{OUT}/vc10.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for sub in ('raw', 'a', 'b'):
        for w in glob.glob(f'{OUT}/{sub}/*.wav'):
            z.write(w, f'{sub}/{os.path.basename(w)}')
print('DONE | zip MB', round(os.path.getsize(f'{OUT}/vc10.zip') / 1e6, 1), flush=True)
