"""Stage + upload the Fenrir training dataset (VC bundle + scripts + init ckpts)."""
import glob
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
VC = 'runs/kaggle_sm26/fenrirB.pt'
assert os.path.exists(VC), f'{VC} not present yet'

STAGE = os.path.abspath('runs/kaggle_sm27_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

# UV-capable trainer scripts
for p in glob.glob('paradee_ms/*.py'):
    shutil.copy(p, os.path.join(STAGE, os.path.basename(p)))
# Kaggle ms_rows with --voice / --f0-min / --max-tail (male voice needed)
shutil.copy('runs/kcheck/ms_rows.py', f'{STAGE}/ms_rows.py')
shutil.copy('scripts/make_gta_cli.py', f'{STAGE}/make_gta_cli.py')
# VC clips + texts + ref
shutil.copy(VC, f'{STAGE}/fenrirB.pt')
import csv
import torch
_rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
with open(f'{STAGE}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'text', 'teacher'])
    for r in _rows:
        w.writerow([r['id'], r['text'], r['text']])
shutil.copy('runs/sm12_in/ref_fenrir.wav', f'{STAGE}/ref_fenrir.wav')
# init from Yati v0.5
shutil.copy('runs/text_f0uv.pt', f'{STAGE}/text_init.pt')
shutil.copy('runs/kaggle_sm21gan/dec_gan.pt', f'{STAGE}/dec_init.pt')
json.dump({"title": "sm27 fenrir train in", "id": "suarifymalaysia/sm27-fenrir-train-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(os.listdir(STAGE)), 'files, MB', round(sum(os.path.getsize(f'{STAGE}/{x}') for x in os.listdir(STAGE)) / 1e6, 1))

api = KaggleApi(); api.authenticate()
r = api.dataset_create_new(folder=STAGE, dir_mode='skip', convert_to_csv=False, public=False)
print('RESULT', r)
