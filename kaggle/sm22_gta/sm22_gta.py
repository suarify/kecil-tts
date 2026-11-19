# sm22-gta: REVIEW Phase 2 - fine-tune the Yati v0.4.1 decoder on its OWN GTA text-model outputs.
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
OUT = '/kaggle/working'
W = f'{OUT}/e2e'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
STAGES = {}


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
        json.dump(dict(failed=name, stages=STAGES, tail=''.join(lines)[-4000:]),
                  open(f'{OUT}/sm22_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return ''.join(lines)


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)
    if not os.listdir(d):
        raise SystemExit(f'EMPTY INPUT {d}')

# ---- environment (matched torch pair LAST) ----
sh('pip install -q soundfile loguru', 'pip_misc')
sh('pip uninstall -y -q torchvision torchao torchtext || true', 'rm_extras')
sh('pip install -q "transformers==4.46.3"', 'pip_tr')
sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh(f'{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchaudio==2.11.0 '
   '--index-url https://download.pytorch.org/whl/cu130', 'torch_pair')
sh(f'{sys.executable} -c "import torch, torchaudio; print(\'PAIR\', torch.__version__, torchaudio.__version__, torch.cuda.is_available())"',
   'verify_pair')

stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')

# ---- inputs ----
SCRIPTS = glob.glob('/kaggle/input/**/ms_train.py', recursive=True)[0]
SDIR = os.path.dirname(SCRIPTS)
ROWS = glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0]
DEC = glob.glob('/kaggle/input/**/dec_c4_last.pt', recursive=True)[0]
assert 'gta_F0' in torch.load(ROWS, map_location='cpu', weights_only=False)[0], 'shard has no GTA fields'
print('scripts', SDIR, '| gta rows', ROWS, '| dec init', DEC, flush=True)

shutil.copytree(SDIR, f'{W}/scripts', dirs_exist_ok=True)
os.makedirs(f'{W}/data/teacher_gta', exist_ok=True)
shutil.copy(ROWS, f'{W}/data/teacher_gta/shard_0000.pt')
PY = sys.executable

# ---- fine-tune the decoder on its own predicted inputs ----
sh(f'{PY} scripts/ms_train.py dec --gta-p 0.7 --init {DEC} --data data/teacher_gta --held 60 '
   f'--steps 3000 --bs 16 --seg 64 --lr 1e-4 --eval-every 500 --max-minutes 90 --out ckpt/dec', 'train_gta', W)

src = f'{W}/ckpt/dec/last.pt'
assert os.path.exists(src), 'no checkpoint produced'
shutil.copy(src, f'{OUT}/dec_gta.pt')
print('saved dec_gta.pt', flush=True)
json.dump(dict(stages=STAGES, total_s=round(time.time() - T0, 1)),
          open(f'{OUT}/sm22_report.json', 'w'), indent=1)
print('SM22 DONE', flush=True)
