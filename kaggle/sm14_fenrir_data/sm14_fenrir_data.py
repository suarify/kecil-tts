# sm 14 fenrir data: convert the raw clips produced by sm-12 to am_fenrir (one VC). Reuses raws.pt.
import glob
import json
import os
import signal
import subprocess
import sys
import time

import soundfile as sf
import torch

OUT = '/kaggle/working'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
FAILS = []


def sh(cmd, name=None):
    t = time.time()
    print(f'=== {time.strftime("%H:%M:%S")} $ {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode:
        print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
        raise SystemExit(f'FAILED {cmd}')
    return p.stdout


def alarm(signum, frame):
    raise TimeoutError('clip timeout')


signal.signal(signal.SIGALRM, alarm)
TIMEOUT = int(os.environ.get('CLIP_TIMEOUT', 150))

subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile')
sh('pip uninstall -y -q torchvision')
FIX = f'''
import torch, subprocess, sys
tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cu = ("cu" + torch.version.cuda.replace(".", "")) if torch.version.cuda else "cpu"
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
for cmd in ([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", f"torchaudio=={{tv}}.*",
             "--index-url", f"https://download.pytorch.org/whl/{{cu}}"],
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", f"torchaudio=={{tv}}.*"]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    print("install rc", p.returncode, (p.stderr or "")[-300:], flush=True)
    if p.returncode == 0:
        break
import torchaudio
print("torchaudio", torchaudio.__version__, "ok", flush=True)
'''
open(f'{OUT}/fix_audio.py', 'w').write(FIX)
sh(f'{sys.executable} {OUT}/fix_audio.py')
sh('pip install -q "transformers==4.46.3"')
sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC; print(\'vc ok\')"')

REF = glob.glob('/kaggle/input/**/ref_fenrir.wav', recursive=True)[0]
raws = torch.load(glob.glob('/kaggle/input/**/raws.pt', recursive=True)[0], map_location='cpu', weights_only=False)
print('raws:', len(raws), '| ref:', REF, flush=True)

OUTP = f'{OUT}/fenrir.pt'
fen = torch.load(OUTP, weights_only=False) if os.path.exists(OUTP) else {}
for p in glob.glob(f'{OUT}/fenrir_partial*.pt'):
    try:
        fen.update({k: v for k, v in torch.load(p, weights_only=False).items() if k not in fen})
    except Exception:
        pass
print('fenrir so far:', len(fen), flush=True)

from chatterbox.vc import ChatterboxVC  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"')
vc = ChatterboxVC.from_pretrained('cuda')
os.makedirs(f'{OUT}/tmpwav', exist_ok=True)
t0 = time.time()
n = 0
for i in raws:
    if i in fen:
        continue
    p = f'{OUT}/tmpwav/{i}.wav'
    try:
        sf.write(p, raws[i].numpy().astype('float32') / 32767.0, 24000)
        signal.alarm(TIMEOUT)
        w = vc.generate(audio=p, target_voice_path=REF)
        signal.alarm(0)
        a = w.squeeze(0).float().cpu().numpy()
        fen[i] = torch.from_numpy((a * 32767).astype('int16'))
        n += 1
    except Exception as e:
        signal.alarm(0)
        FAILS.append(dict(id=i, err=type(e).__name__ + ' ' + repr(e)[:100]))
        print('VC FAIL', i, type(e).__name__, flush=True)
    finally:
        if os.path.exists(p):
            os.remove(p)
    if n % 100 == 0 and n:
        torch.save(fen, f'{OUT}/fenrir_partial.pt')
        torch.save(fen, OUTP)
        print(f'vc {len(fen)}/{len(raws)} | {(time.time() - t0) / 60:.1f} min', flush=True)
torch.save(fen, OUTP)
json.dump(dict(fenrir=len(fen), fails=FAILS, minutes=round((time.time() - t0) / 60, 1)),
          open(f'{OUT}/sm14_report.json', 'w'), indent=1)
print('DONE fenrir', len(fen), 'fails', len(FAILS), flush=True)
