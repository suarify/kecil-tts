# sm 11 big: train a WIDER Michael decoder (head width 320, 6 blocks) from the existing trunk.
# The text side is reused unchanged (durations/F0/N stay as validated); only the audio decoder gets more capacity.
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import torch

T0 = time.time()
STAGES = {}
OUT = '/kaggle/working'
W = f'{OUT}/e2e'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd, name=None, cwd=None):
    t = time.time()
    print(f'=== [{time.time() - T0:6.0f}s] $ {cmd}', flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors='replace', bufsize=1)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not re.search(r'FutureWarning|%\||Xet', line):
            print('   ' + line.rstrip()[:400], flush=True)
    p.wait()
    if name:
        STAGES[name] = round(time.time() - t, 1)
        print(f'=== stage {name}: {STAGES[name]} s (exit {p.returncode})', flush=True)
    if p.returncode != 0:
        json.dump(dict(failed=name or cmd, stages=STAGES, tail=''.join(lines)[-3000:]), open(f'{OUT}/sm11_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return ''.join(lines)


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru', 'pip_misc')
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')
sys.path.insert(0, os.path.dirname(glob.glob('/kaggle/input/**/ms_train.py', recursive=True)[0]))

PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_train.py', recursive=True)[0])
SHARD = glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0]
DEC_INIT = glob.glob('/kaggle/input/**/dec_init.pt', recursive=True)[0]
os.makedirs(f'{W}/data/teacher', exist_ok=True)
shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
shutil.copy(SHARD, f'{W}/data/teacher/shard_0000.pt')

# ---- patch the trainer to build the wider head (robust to argument order / spacing) ----
p = f'{W}/scripts/ms_train.py'
s = open(p).read()
new_line = "VDecoder(head='v2', dim=320, blocks=6, K=128)"
s2, n_sub = re.subn(r"VDecoder\([^)]*\)", new_line, s)
print('patched VDecoder call(s):', n_sub, flush=True)
assert n_sub >= 1, 'no VDecoder call found in ms_train.py'
open(p, 'w').write(s2)
for i, line in enumerate(open(p).read().splitlines(), 1):
    if 'VDecoder(' in line and 'import' not in line:
        print(f'  patched line {i}: {line.strip()}', flush=True)

# ---- build the init checkpoint: copy the old trunk, random-init the wider head ----
from vhead import VDecoder  # noqa: E402

old = torch.load(DEC_INIT, map_location='cpu', weights_only=False)['model']
new = VDecoder(head='v2', dim=320, blocks=6, K=128).eval()
sd = new.state_dict()
copied, skipped = [], []
for k, v in sd.items():
    if k in old and old[k].shape == v.shape:
        sd[k] = old[k].clone(); copied.append(k)
    else:
        skipped.append(k)
new.load_state_dict(sd)
print(f'init transfer: copied {len(copied)} tensors, new/random {len(skipped)} (head)', flush=True)
print('skipped sample:', skipped[:6], flush=True)
os.makedirs(f'{W}/init', exist_ok=True)
torch.save({'model': new.state_dict(), 'step': 0}, f'{W}/init/big.pt')
print('big init params:', round(sum(p.numel() for p in new.parameters()) / 1e6, 2), 'M', flush=True)

HELD = 60
PY = sys.executable
patch_check = [ln for ln in open(p).read().splitlines() if 'VDecoder(' in ln and 'import' not in ln][0]
assert 'dim=320' in patch_check, patch_check
sh(f'{PY} scripts/ms_train.py dec --init init/big.pt --data data/teacher --held {HELD} '
   f'--steps 16000 --bs 16 --max-minutes 50 --out ckpt/bigdec', 'train_big_decoder', W)

best = None
for cand in ('ckpt/bigdec/best.pt', 'ckpt/bigdec/last.pt'):
    if os.path.exists(f'{W}/{cand}'):
        best = cand
if best is None:
    raise SystemExit('no checkpoint produced')
shutil.copy(f'{W}/{best}', f'{OUT}/dec_big_last.pt')
ck = torch.load(f'{OUT}/dec_big_last.pt', map_location='cpu', weights_only=False)
json.dump(dict(stage=STAGES, dec_step=ck.get('step'), params=round(sum(p.numel() for p in new.parameters()) / 1e6, 2),
               total_s=round(time.time() - T0, 1)), open(f'{OUT}/sm11_report.json', 'w'), indent=1)
print('DONE', json.dumps(dict(dec_step=ck.get('step'), total_s=round(time.time() - T0, 1))), flush=True)
