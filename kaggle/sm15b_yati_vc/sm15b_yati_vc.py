# sm 15b path B: convert Yati's clean student output to nicole with one ChatterboxVC pass (CPU).
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import soundfile as sf

OUT = '/kaggle/working'
FAILS = []


def sh(cmd, name=None):
    print('===', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        if name:
            json.dump(dict(failed=name, tail=p.stderr[-2000:]), open(f'{OUT}/sm15b_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return p.stdout


def alarm(signum, frame):
    raise TimeoutError('clip timeout')


signal.signal(signal.SIGALRM, alarm)
TIMEOUT = 300

print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile')
sh('pip uninstall -y -q torchvision torchao torchtext || true')
sh('pip install -q "transformers==4.46.3"')
# matched torch pair last (Kaggle's default torchaudio is ABI-broken)
FIX = f'''
import sys, subprocess
p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                    "torch==2.11.0", "torchaudio==2.11.0",
                    "--index-url", "https://download.pytorch.org/whl/cu130"], capture_output=True, text=True)
print("torch pair rc", p.returncode, (p.stderr or "")[-200:], flush=True)
import torch, torchaudio
print("PAIR", torch.__version__, torchaudio.__version__, "cuda", torch.cuda.is_available(), flush=True)
'''
open(f'{OUT}/fix_torch.py', 'w').write(FIX)
sh(f'{sys.executable} {OUT}/fix_torch.py')
sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC; print(\'vc import ok\')"')

REF = glob.glob('/kaggle/input/**/ref_nicole.wav', recursive=True)[0]
WAVS = sorted(glob.glob('/kaggle/input/**/yati_*.wav', recursive=True))
print('wavs', len(WAVS), 'ref', REF, flush=True)

from chatterbox.vc import ChatterboxVC  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"')
vc = ChatterboxVC.from_pretrained('cpu')
os.makedirs(f'{OUT}/converted', exist_ok=True)
t0 = time.time()
for p in WAVS:
    name = os.path.basename(p)
    i = name[5:-4]
    try:
        signal.alarm(TIMEOUT)
        w = vc.generate(audio=p, target_voice_path=REF)
        signal.alarm(0)
        a = w.squeeze(0).float().cpu().numpy()
        sf.write(f'{OUT}/converted/{i}.wav', a, 24000)
        print(i, round(len(a) / 24000, 2), 's |', round((time.time() - t0) / 60, 1), 'min', flush=True)
    except Exception as e:
        signal.alarm(0)
        FAILS.append(dict(id=i, err=type(e).__name__ + ' ' + repr(e)[:120]))
        print('FAIL', i, type(e).__name__, flush=True)

json.dump(dict(converted=len(os.listdir(f'{OUT}/converted')), fails=FAILS, minutes=round((time.time() - t0) / 60, 1)),
          open(f'{OUT}/sm15b_report.json', 'w'), indent=1)
print('DONE converted', len(os.listdir(f'{OUT}/converted')), flush=True)
