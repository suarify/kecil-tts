"""Stage + upload the p4 timing dataset: 50 new texts + ref_ms.wav."""
import csv
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm23_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

rows = list(csv.DictReader(open('eval/p4_new_texts_final.tsv', encoding='utf-8'), delimiter='\t'))[:50]
with open(f'{STAGE}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'text', 'teacher'])
    for r in rows:
        w.writerow([r['id'], r['text'], r['teacher']])
shutil.copy('runs/sm12_in/ref_ms.wav', f'{STAGE}/ref_ms.wav')
json.dump({"title": "sm23 p4 timing in", "id": "suarifymalaysia/sm23-p4-timing-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(os.listdir(STAGE)), 'files, 50 texts')
api = KaggleApi(); api.authenticate()
r = api.dataset_create_new(folder=STAGE, dir_mode='skip', convert_to_csv=False, public=False)
print('RESULT', r)
