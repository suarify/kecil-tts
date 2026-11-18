"""Rebuild nicole rows from all 1,281 clips with the correct tail flags (local CPU), then continue training."""
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PMS = os.path.abspath('runs/pms_ds')
STUBS = os.path.abspath('.stubs')
G2P_DIR = os.path.abspath('g2p')
WORK = os.path.abspath('runs/nicole_full')
CLIPS = os.path.abspath('runs/nicole_full_clips')
TSV = os.path.abspath('runs/sm12_in/texts.tsv')

# extract all 1,281 converted clips to wavs
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

os.makedirs(CLIPS, exist_ok=True)
clips = torch.load('runs/sm19_out/nicole.pt', map_location='cpu', weights_only=False)
for k, a in clips.items():
    p = f'{CLIPS}/{k}.wav'
    if not os.path.exists(p):
        sf.write(p, a.numpy().astype('float32') / 32767.0, 24000)
print('clip wavs:', len(os.listdir(CLIPS)), flush=True)

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(f'{WORK}/data/teacher')
env = dict(os.environ)
env['PYTHONPATH'] = PMS + os.pathsep + G2P_DIR + os.pathsep + STUBS + os.pathsep + env.get('PYTHONPATH', '')
PY = sys.executable


def run(args, name):
    print('===', name, flush=True)
    p = subprocess.run([PY, f'{PMS}/ms_rows.py'] + args, cwd=WORK, env=env)
    if p.returncode:
        raise SystemExit(f'FAILED {name}')


run(['--clips', CLIPS, '--tsv', TSV, '--g2p', G2P_DIR, '--out', 'data/teacher/shard_0000.pt',
     '--device', 'cpu', '--letters', '--max-last', '1.0', '--max-tail', '99'], 'rows_all1281')
rows = torch.load(f'{WORK}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('ROWS:', len(rows), flush=True)
