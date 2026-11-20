"""Stage + upload the Nicole training dataset (VC bundle + scripts + init ckpts from Yati v0.5)."""
import csv
import glob
import json
import os
import shutil

import torch

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
VC = 'runs/kaggle_sm28/nicoleB_v05.pt'
assert os.path.exists(VC), f'{VC} not present yet'

STAGE = os.path.abspath('runs/kaggle_sm29_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)
for p in glob.glob('paradee_ms/*.py'):
    shutil.copy(p, os.path.join(STAGE, os.path.basename(p)))
shutil.copy('runs/kcheck/ms_rows.py', f'{STAGE}/ms_rows.py')
shutil.copy('scripts/make_gta_cli.py', f'{STAGE}/make_gta_cli.py')
shutil.copy(VC, f'{STAGE}/nicoleB_v05.pt')
rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
with open(f'{STAGE}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t'); w.writerow(['id', 'text', 'teacher'])
    for r in rows:
        w.writerow([r['id'], r['text'], r['text']])
shutil.copy('runs/text_f0uv.pt', f'{STAGE}/text_init.pt')
shutil.copy('runs/kaggle_sm21gan/dec_gan.pt', f'{STAGE}/dec_init.pt')
json.dump({"title": "sm29 nicole train in", "id": "suarifymalaysia/sm29-nicole-train-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(os.listdir(STAGE)), 'files')
api = KaggleApi(); api.authenticate()
r = api.dataset_create_new(folder=STAGE, dir_mode='skip', convert_to_csv=False, public=False)
print('RESULT', r)
