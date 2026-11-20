# sm27-fenrir-train: full Fenrir training pipeline from Yati v0.5 data.
# rows (male f0-min 50) -> text 10k UV -> GTA rows -> decoder 12k GTA -> GAN 2k.
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
                  open(f'{OUT}/sm27_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    return ''.join(lines)


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

sh('pip install -q soundfile loguru librosa', 'pip_misc')
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
G2P = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
VC = glob.glob('/kaggle/input/**/fenrirB.pt', recursive=True)[0]
TEXTS = glob.glob('/kaggle/input/**/texts.tsv', recursive=True)[0]
TINIT = glob.glob('/kaggle/input/**/text_init.pt', recursive=True)[0]
DINIT = glob.glob('/kaggle/input/**/dec_init.pt', recursive=True)[0]
print('inputs ok | VC', VC, '| texts', TEXTS, flush=True)

shutil.copytree(SDIR, f'{W}/scripts', dirs_exist_ok=True)
CLIPS = f'{OUT}/clips'
os.makedirs(CLIPS, exist_ok=True)
import soundfile as sf  # noqa: E402
clips = torch.load(VC, map_location='cpu', weights_only=False)
for k, v in clips.items():
    sf.write(f'{CLIPS}/{k}.wav', v.numpy().astype('float32') / 32767, 24000)
print('clips', len(clips), flush=True)
assert len(clips) >= 900
PY = sys.executable

# 1. rows (male voice: f0-min 50, am_fenrir)
sh(f'{PY} scripts/ms_rows.py --clips {CLIPS} --tsv {TEXTS} --g2p {G2P} --out data/teacher/shard_0000.pt '
   f'--device cuda --voice am_fenrir --f0-min 50 --f0-max 300 --letters --max-last 1.0 --max-tail 99',
   'rows', W)
rows = torch.load(f'{W}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('ROWS', len(rows), flush=True)
assert len(rows) >= 800, f'too few rows: {len(rows)}'

# 2. text side (10k UV steps)
sh(f'{PY} scripts/ms_train.py text --uv-text --init {TINIT} --data data/teacher --held 60 '
   f'--steps 10000 --bs 16 --eval-every 500 --max-minutes 45 --out ckpt/text', 'train_text', W)
# 3. GTA rows
sh(f'{PY} scripts/make_gta_cli.py --rows data/teacher/shard_0000.pt --text ckpt/text/last.pt '
   f'--out data/teacher_gta/shard_0000.pt', 'gta', W)
# 4. decoder (12k GTA steps)
sh(f'{PY} scripts/ms_train.py dec --gta-p 0.7 --init {DINIT} --data data/teacher_gta --held 60 '
   f'--steps 12000 --bs 16 --seg 64 --lr 1e-4 --eval-every 500 --max-minutes 60 --out ckpt/dec', 'train_dec', W)
# 5. short GAN (2k, D warmup 100, lower D lr for balance)
sh(f'{PY} scripts/ms_train.py dec --gan --gan-warmup 100 --gta-p 0.7 --init {DINIT} '
   f'--data data/teacher_gta --held 60 --steps 2100 --bs 16 --seg 32 --lr 1e-4 --dlr 1e-4 '
   f'--eval-every 500 --max-minutes 60 --out ckpt/gan', 'train_gan', W)

for src, dst in (('ckpt/text/last.pt', 'fenrir_text.pt'), ('ckpt/dec/last.pt', 'fenrir_dec.pt'),
                 ('ckpt/gan/last.pt', 'fenrir_gan.pt')):
    p = f'{W}/{src}'
    if os.path.exists(p):
        shutil.copy(p, f'{OUT}/{dst}')
        print('saved', dst, flush=True)
json.dump(dict(stages=STAGES, rows=len(rows), total_s=round(time.time() - T0, 1)),
          open(f'{OUT}/sm27_report.json', 'w'), indent=1)
print('SM27 DONE', flush=True)
