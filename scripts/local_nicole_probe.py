"""Local CPU smoke-test training on the 624-row nicole shard from sm-14b.

No GPU needed, no Kaggle queue. Steps:
  - locate the rows shard in the pulled sm-14b output
  - train decoder 500 steps and text 300 steps from the Yati init (female)
  - save checkpoints to runs/nicole_probe/ckpt
"""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PMS = os.path.abspath('kaggle/ds_pms')
STUBS = os.path.abspath('.stubs')
WORK = os.path.abspath('runs/nicole_probe')

cands = glob.glob('runs/kaggle_sm14b/**/shard_0000.pt', recursive=True)
if not cands:
    raise SystemExit('rows shard not pulled yet (sm-14b output download still running)')
src = cands[0]
print('using rows:', src, flush=True)

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(f'{WORK}/data/teacher')
shutil.copy(src, f'{WORK}/data/teacher/shard_0000.pt')

import torch  # noqa: E402

rows = torch.load(f'{WORK}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('rows:', len(rows), flush=True)

env = dict(os.environ)
env['PYTHONPATH'] = PMS + os.pathsep + STUBS + os.pathsep + env.get('PYTHONPATH', '')
PY = sys.executable


def run(args, name):
    print('===', name, flush=True)
    p = subprocess.run([PY, f'{PMS}/ms_train.py'] + args, cwd=WORK, env=env)
    if p.returncode:
        raise SystemExit(f'FAILED {name}')


run(['dec', '--init', f'{ROOT}/runs/k16_final/dec_c4_last.pt', '--data', 'data/teacher',
     '--held', '4', '--steps', '500', '--bs', '8', '--device', 'cpu', '--max-minutes', '40',
     '--out', 'ckpt/dec'], 'decoder_500')

run(['text', '--init', f'{ROOT}/runs/k16_final/textA_c4_last.pt', '--data', 'data/teacher',
     '--held', '4', '--steps', '300', '--bs', '8', '--device', 'cpu', '--eval-every', '100',
     '--score', 'dur_asr', '--max-minutes', '40', '--out', 'ckpt/text'], 'text_300')

print('PROBE DONE ->', f'{WORK}/ckpt/dec/last.pt', f'{WORK}/ckpt/text/last.pt', flush=True)
