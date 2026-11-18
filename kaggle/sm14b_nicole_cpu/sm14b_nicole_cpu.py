# sm 13a nicole probe: SHORT training (3,000 steps) to hear whether the student voice works
# before spending the full 12k-step run. Same data/init as the full run.
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
        json.dump(dict(failed=name, stages=STAGES, tail=''.join(lines)[-2000:]), open(f'{OUT}/sm13a_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return ''.join(lines)


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)
    if not os.listdir(d):
        raise SystemExit(f'EMPTY INPUT {d}')

sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru', 'pip_misc')
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')

PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_rows.py', recursive=True)[0])
DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
NIC = glob.glob('/kaggle/input/**/nicole.pt', recursive=True)[0]
TSV = glob.glob('/kaggle/input/**/texts.tsv', recursive=True)[0]
DEC_INIT = glob.glob('/kaggle/input/**/yati_c4_dec_last.pt', recursive=True)[0]
TEXT_INIT = glob.glob('/kaggle/input/**/textA_c4_last.pt', recursive=True)[0]
print('inputs ok | nicole:', NIC, flush=True)

import soundfile as sf  # noqa: E402

shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
CLIPS = f'{OUT}/nicole_wav'
os.makedirs(CLIPS, exist_ok=True)
nic = torch.load(NIC, map_location='cpu', weights_only=False)
for i, a in nic.items():
    sf.write(f'{CLIPS}/{i}.wav', a.numpy().astype('float32') / 32767.0, 24000)
print('clip wavs:', len(nic), flush=True)

PY = sys.executable
sh(f'{PY} scripts/ms_rows.py --clips {CLIPS} --tsv {TSV} --g2p {DS} --out data/teacher/shard_0000.pt '
   f'--device cpu --letters', 'rows', W)
rows = torch.load(f'{W}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('rows:', len(rows), flush=True)
if len(rows) < 800:
    raise SystemExit(f'too few rows: {len(rows)}')
shutil.copy(f'{W}/data/teacher/shard_0000.pt', f'{OUT}/nicole_rows.pt')

HELD = 60
sh(f'{PY} scripts/ms_train.py dec --init {DEC_INIT} --data data/teacher --held {HELD} '
   f'--steps 3000 --bs 16 --max-minutes 15 --out ckpt/dec', 'train_decoder_probe', W)
sh(f'{PY} scripts/ms_train.py text --init {TEXT_INIT} --data data/teacher --held {HELD} '
   f'--steps 2500 --bs 16 --eval-every 500 --score dur_asr --max-minutes 15 --out ckpt/text', 'train_text_probe', W)

for src, dst in (('ckpt/dec/last.pt', 'dec_nicole_probe.pt'), ('ckpt/text/last.pt', 'text_nicole_probe.pt')):
    p = f'{W}/{src}'
    if os.path.exists(p):
        shutil.copy(p, f'{OUT}/{dst}')
        print('saved', dst, flush=True)
json.dump(dict(stages=STAGES, rows=len(rows), total_s=round(time.time() - T0, 1)),
          open(f'{OUT}/sm13a_report.json', 'w'), indent=1)
print('PROBE DONE', flush=True)
