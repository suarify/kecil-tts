# sm 08 vc15: proof of concept - generate 15 Malay sentences with Chatterbox Multilingual using the
# Kokoro am_michael reference DIRECTLY (one conversion) instead of generate -> female -> male (two
# conversions). Saves the single-conversion clips and the existing double-converted clips for A/B.
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

OUT = '/kaggle/working'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        raise SystemExit(f'FAILED {cmd}')


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile')
# The Kaggle image's torchvision does not match its torch ("operator torchvision::nms does not exist"), and
# transformers pulls it in while importing LlamaModel. Chatterbox does not need torchvision -> remove it and
# pin a transformers version that imports cleanly without it.
sh('pip uninstall -y -q torchvision')
sh('pip install -q "transformers==4.46.3"')
sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; print(\'chatterbox import ok\')"')

import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

REF = glob.glob('/kaggle/input/**/ref_michael.wav', recursive=True)[0]
TEXTS = glob.glob('/kaggle/input/**/texts.csv', recursive=True)[0]
OLD_PT = glob.glob('/kaggle/input/**/cb_michael.pt', recursive=True)[0]
plan = pd.read_csv(TEXTS)
print('texts:', len(plan), '| ref:', REF, '| old clips:', OLD_PT, flush=True)

os.makedirs(f'{OUT}/single', exist_ok=True)
os.makedirs(f'{OUT}/old', exist_ok=True)

# --- old (double-converted) clips for the same ids ---
old = torch.load(OLD_PT, map_location='cpu', weights_only=False)
n_old = 0
for r in plan.itertuples():
    if r.id in old:
        a = old[r.id].numpy().astype(np.float32) / 32767.0
        sf.write(f'{OUT}/old/{r.id}.wav', a, 24000)
        n_old += 1
print('old clips written:', n_old, flush=True)

# --- single conversion: Chatterbox with the am_michael prompt directly ---
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as C; C.from_pretrained(device=\'cpu\')"')
tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')
t0 = time.time()
n = 0
for r in plan.itertuples():
    torch.manual_seed(1234)
    try:
        wav = tts.generate(r.text, language_id='ms', audio_prompt_path=REF,
                           exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
    except Exception as e:
        print('FAIL', r.id, type(e).__name__, repr(e)[:120], flush=True)
        continue
    a = wav.squeeze(0).float().cpu().numpy()
    sf.write(f'{OUT}/single/{r.id}.wav', a, tts.sr)
    n += 1
    print(f'{r.id} {len(a) / tts.sr:.2f}s | {(time.time() - t0) / 60:.1f} min', flush=True)

with zipfile.ZipFile(f'{OUT}/vc15.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for sub in ('single', 'old'):
        for w in glob.glob(f'{OUT}/{sub}/*.wav'):
            z.write(w, f'{sub}/{os.path.basename(w)}')
print('DONE single', n, 'old', n_old, 'zip MB', round(os.path.getsize(f'{OUT}/vc15.zip') / 1e6, 1), flush=True)
