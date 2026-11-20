# sm28-nicole-vc: one ChatterboxVC pass from Yati v0.5 clips -> am_fenrir.
# Loudness-normalized to RMS ~0.07 (Yati training-row level) to fix the "soft voice" issue.
# Input is ONE .pt bundle (Kaggle auto-expands wav folders and silently drops files: 948/1281 bug).
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

OUT = '/kaggle/working'


def sh(cmd, name=None):
    print('===', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    if p.returncode:
        print((p.stdout or '')[-1500:], (p.stderr or '')[-2500:], flush=True)
        json.dump(dict(failed=name, tail=(p.stderr or '')[-2500:]), open(f'{OUT}/sm28_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return p.stdout


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile', 'pip_cb')
sh('pip uninstall -y -q torchvision torchao torchtext || true', 'rm_extras')
sh('pip install -q "transformers==4.46.3"', 'pip_tr')
sh(f'{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchaudio==2.11.0 '
   '--index-url https://download.pytorch.org/whl/cu130', 'torch_pair')
sh(f'{sys.executable} -c "import torch, torchaudio; print(\'PAIR\', torch.__version__, torchaudio.__version__, torch.cuda.is_available())"',
   'verify_pair')

BUNDLE = glob.glob('/kaggle/input/**/yati_v05_clips.pt', recursive=True)[0]
REF = glob.glob('/kaggle/input/**/ref_nicole.wav', recursive=True)[0]
print('bundle', BUNDLE, '| ref', REF, flush=True)

import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from chatterbox.vc import ChatterboxVC  # noqa: E402

clips = torch.load(BUNDLE, map_location='cpu', weights_only=False)
print('clips', len(clips), flush=True)
assert len(clips) >= 1200, f'only {len(clips)} clips'

CLIPDIR = f'{OUT}/yati_wav'
os.makedirs(CLIPDIR, exist_ok=True)
for k, v in clips.items():
    sf.write(f'{CLIPDIR}/{k}.wav', v.numpy().astype('float32') / 32767, 24000)

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"', 'warm_cache')
vc = ChatterboxVC.from_pretrained('cuda')

OUTP = f'{OUT}/nicoleB_v05.pt'
res = {}
for p in glob.glob(f'{OUT}/fenrirB_partial*.pt'):
    try:
        res.update({k: v for k, v in torch.load(p, weights_only=False).items() if k not in res})
    except Exception:
        pass
print('resume so far:', len(res), flush=True)


def alarm(signum, frame):
    raise TimeoutError('clip timeout')


signal.signal(signal.SIGALRM, alarm)
TARGET_RMS = 0.07   # match the Yati teacher training rows (RMS ~0.06-0.08)
t0 = time.time()
for n, (k, src) in enumerate(clips.items()):
    if k in res:
        continue
    p = f'{CLIPDIR}/{k}.wav'
    try:
        signal.alarm(150)
        w = vc.generate(audio=p, target_voice_path=REF)
        signal.alarm(0)
        a = w.squeeze(0).float().cpu().numpy()
        rms = float(np.sqrt(np.mean(a ** 2)))
        if rms > 1e-6:
            a = a * (TARGET_RMS / rms)              # loudness match
        peak = float(np.abs(a).max())
        if peak > 0.95:
            a = a * (0.95 / peak)
        res[k] = torch.from_numpy((a * 32767).astype('int16'))
    except Exception as e:
        signal.alarm(0)
        print('FAIL', k, type(e).__name__, str(e)[:120], flush=True)
        continue
    if len(res) % 100 == 0:
        print('converted', len(res), '|', round((time.time() - t0) / 60, 1), 'min', flush=True)
        torch.save(res, f'{OUT}/nicoleB_v05_partial.pt')

torch.save(res, OUTP)
print('DONE', len(res), 'clips', 'fails', len(clips) - len(res), '|', round((time.time() - t0) / 60, 1), 'min', flush=True)
json.dump(dict(clips=len(res), fails=len(clips) - len(res), target_rms=TARGET_RMS,
               minutes=round((time.time() - t0) / 60, 1)), open(f'{OUT}/sm28_report.json', 'w'), indent=1)
