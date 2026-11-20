"""Stage + upload the Fenrir VC dataset: 1281 Yati v0.5 clips bundled as ONE .pt + ref + texts."""
import csv
import glob
import json
import os
import shutil

import soundfile as sf
import torch

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CLIPS = 'runs/pathb_yati_v05_clips'
files = sorted(glob.glob(f'{CLIPS}/*.wav'))
assert len(files) == 1281, f'expected 1281 Yati v0.5 clips, got {len(files)}'

STAGE = os.path.abspath('runs/kaggle_sm26_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

bundle = {}
for p in files:
    a, sr = sf.read(p, dtype='float32')
    assert sr == 24000
    bundle[os.path.basename(p)[:-4]] = torch.from_numpy((a * 32767).astype('int16'))
torch.save(bundle, f'{STAGE}/yati_v05_clips.pt')
print('bundled', len(bundle), 'clips, MB', round(os.path.getsize(f'{STAGE}/yati_v05_clips.pt') / 1e6, 1))

rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
with open(f'{STAGE}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'text', 'teacher'])
    for r in rows:
        w.writerow([r['id'], r['text'], r['text']])
shutil.copy('runs/sm12_in/ref_fenrir.wav', f'{STAGE}/ref_fenrir.wav')
json.dump({"title": "sm26 fenrir vc in", "id": "suarifymalaysia/sm26-fenrir-vc-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)

api = KaggleApi(); api.authenticate()
r = api.dataset_create_new(folder=STAGE, dir_mode='skip', convert_to_csv=False, public=False)
print('RESULT', r)
