# sm 07 delta: continue training the Michael C4 decoder and text side on the delta (1,261 rows),
# then save checkpoints for the local FIX H export (K=128, gap-masked noise, 22 ms burst hangover,
# 75-200 ms duration clamp, noise-gain smoothing, +3 dB air).
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

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
        if not re.search(r'FutureWarning|%\||Xet|warn', line):
            print('   ' + line.rstrip()[:400], flush=True)
    p.wait()
    out = ''.join(lines)
    if name:
        STAGES[name] = round(time.time() - t, 1)
        print(f'=== stage {name}: {STAGES[name]} s (exit {p.returncode})', flush=True)
    if p.returncode != 0:
        json.dump(dict(failed=name or cmd, stages=STAGES, tail=out[-3000:]), open(f'{OUT}/sm07_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED: {cmd}')
    return out


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /kaggle/working | tail -1', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input dataset:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru onnx onnxruntime', 'pip_misc')
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')

PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_train.py', recursive=True)[0])
SHARD = glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0]
DEC_INIT = glob.glob('/kaggle/input/**/dec_init.pt', recursive=True)[0]
TEXT_INIT = glob.glob('/kaggle/input/**/text_init.pt', recursive=True)[0]
os.makedirs(W, exist_ok=True)
shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
os.makedirs(f'{W}/data/teacher', exist_ok=True)
shutil.copy(SHARD, f'{W}/data/teacher/shard_0000.pt')
HELD = 60
PY = sys.executable

import torch  # noqa: E402

rows = torch.load(f'{W}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('rows:', len(rows), flush=True)

# ---- decoder: continue from step 12,000 with a lower learning rate (the delta pass) ----
sh(f'{PY} scripts/ms_train.py dec --init {DEC_INIT} --data data/teacher --held {HELD} '
   f'--steps 4000 --bs 16 --lr 2e-4 --max-minutes 30 --out ckpt/dec2', 'train_decoder', W)

# ---- text side: continue from step 7,500, duration-focused checkpoint selection ----
sh(f'{PY} scripts/ms_train.py text --init {TEXT_INIT} --data data/teacher --held {HELD} '
   f'--steps 6000 --bs 16 --lr 2e-4 --eval-every 500 --score dur_asr --max-minutes 30 --out ckpt/text2',
   'train_text', W)

for src, dst in (('ckpt/dec2/last.pt', 'dec_delta_last.pt'), ('ckpt/text2/last.pt', 'textA_delta_last.pt')):
    p = f'{W}/{src}'
    if os.path.exists(p):
        shutil.copy(p, f'{OUT}/{dst}')
        print('saved', dst, flush=True)

d = torch.load(f'{OUT}/dec_delta_last.pt', map_location='cpu', weights_only=False)
t = torch.load(f'{OUT}/textA_delta_last.pt', map_location='cpu', weights_only=False)
json.dump(dict(dec_step=d.get('step'), text_step=t.get('step'), stages=STAGES, total_s=round(time.time() - T0, 1)),
          open(f'{OUT}/sm07_report.json', 'w'), indent=1)
print('DONE', json.dumps(dict(dec_step=d.get('step'), text_step=t.get('step'), total_s=round(time.time() - T0, 1))), flush=True)
