# sm 12 nicole data: generate all 1,281 raw Malay Chatterbox clips with ONE consistent configuration,
# then convert them ONCE to the af_nicole voice. Checkpoints every 100 clips; per-clip timeout; failures logged.
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
STAGES = {}
FAILS = []


def sh(cmd, name=None):
    t = time.time()
    print(f'=== [{time.strftime("%H:%M:%S")}] $ {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.stdout.strip():
        print(p.stdout[-2000:], flush=True)
    if p.returncode:
        print(p.stderr[-3000:], flush=True)
        json.dump(dict(failed=name, stages=STAGES), open(f'{OUT}/sm12_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    if name:
        STAGES[name] = round(time.time() - t, 1)
    return p.stdout


def alarm(signum, frame):
    raise TimeoutError('clip timeout')


signal.signal(signal.SIGALRM, alarm)
TIMEOUT = int(os.environ.get('CLIP_TIMEOUT', 150))

print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q chatterbox-tts soundfile', 'pip')
sh('pip uninstall -y -q torchvision', 'pip_rm_tv')
sh('pip install -q "transformers==4.46.3"', 'pip_tr')
sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from chatterbox.vc import ChatterboxVC; print(\'imports ok\')"', 'imports')

import pandas as pd  # noqa: E402

plan = pd.read_csv(glob.glob('/kaggle/input/**/texts.tsv', recursive=True)[0], sep='\t', dtype=str, keep_default_na=False)
REF_MS = glob.glob('/kaggle/input/**/ref_ms.wav', recursive=True)[0]
REF_N = glob.glob('/kaggle/input/**/ref_nicole.wav', recursive=True)[0]
print('rows:', len(plan), '| refs:', REF_MS, REF_N, flush=True)

RAWS = f'{OUT}/raws.pt'
raws = torch.load(RAWS, weights_only=False) if os.path.exists(RAWS) else {}
# resume: also pick up any partial file left by a previous failed run
for p in glob.glob(f'{OUT}/raws_partial*.pt'):
    try:
        raws.update({k: v for k, v in torch.load(p, weights_only=False).items() if k not in raws})
    except Exception:
        pass
print('raws so far:', len(raws), flush=True)

# ---------------- 1. generate the raw Malay clips (one fixed configuration) ----------------
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as C; C.from_pretrained(device=\'cpu\')"', 'warm_cache')
tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')
t0 = time.time()
n = 0
for r in plan.itertuples():
    if r.id in raws:
        continue
    try:
        torch.manual_seed(1234)
        signal.alarm(TIMEOUT)
        wav = tts.generate(r.teacher, language_id='ms', audio_prompt_path=REF_MS,
                           exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
        signal.alarm(0)
        a = wav.squeeze(0).float().cpu().numpy()
        raws[r.id] = torch.from_numpy((a * 32767).astype('int16'))
        n += 1
    except Exception as e:
        signal.alarm(0)
        FAILS.append(dict(stage='gen', id=r.id, err=type(e).__name__ + ' ' + repr(e)[:100]))
        print('GEN FAIL', r.id, type(e).__name__, flush=True)
    if n % 100 == 0 and n:
        torch.save(raws, f'{OUT}/raws_partial.pt')
        torch.save(raws, RAWS)
        print(f'gen {len(raws)}/{len(plan)} | {(time.time() - t0) / 60:.1f} min', flush=True)
torch.save(raws, RAWS)
STAGES['generate'] = round(time.time() - t0, 1)
print(f'generation done: {len(raws)} clips, {STAGES["generate"] / 60:.1f} min', flush=True)

# ---------------- 2. one voice conversion -> af_nicole ----------------
NIC = f'{OUT}/nicole.pt'
nic = torch.load(NIC, weights_only=False) if os.path.exists(NIC) else {}
for p in glob.glob(f'{OUT}/nicole_partial*.pt'):
    try:
        nic.update({k: v for k, v in torch.load(p, weights_only=False).items() if k not in nic})
    except Exception:
        pass
print('nicole so far:', len(nic), flush=True)

from chatterbox.vc import ChatterboxVC  # noqa: E402

sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC as C; C.from_pretrained(\'cpu\')"', 'warm_vc')
vc = ChatterboxVC.from_pretrained('cuda')
os.makedirs(f'{OUT}/tmpwav', exist_ok=True)
t0 = time.time()
n = 0
for i in raws:
    if i in nic:
        continue
    p = f'{OUT}/tmpwav/{i}.wav'
    try:
        sf.write(p, raws[i].numpy().astype('float32') / 32767.0, 24000)
        signal.alarm(TIMEOUT)
        w = vc.generate(audio=p, target_voice_path=REF_N)
        signal.alarm(0)
        a = w.squeeze(0).float().cpu().numpy()
        nic[i] = torch.from_numpy((a * 32767).astype('int16'))
        n += 1
    except Exception as e:
        signal.alarm(0)
        FAILS.append(dict(stage='vc', id=i, err=type(e).__name__ + ' ' + repr(e)[:100]))
        print('VC FAIL', i, type(e).__name__, flush=True)
    finally:
        if os.path.exists(p):
            os.remove(p)
    if n % 100 == 0 and n:
        torch.save(nic, f'{OUT}/nicole_partial.pt')
        torch.save(nic, NIC)
        print(f'vc {len(nic)}/{len(raws)} | {(time.time() - t0) / 60:.1f} min', flush=True)
torch.save(nic, NIC)
STAGES['vc'] = round(time.time() - t0, 1)
print(f'vc done: {len(nic)} clips, {STAGES["vc"] / 60:.1f} min', flush=True)

json.dump(dict(stages=STAGES, raws=len(raws), nicole=len(nic), fails=FAILS),
          open(f'{OUT}/sm12_report.json', 'w'), indent=1)
print('DONE raws', len(raws), 'nicole', len(nic), 'fails', len(FAILS), flush=True)
