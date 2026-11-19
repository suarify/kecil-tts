"""Stage + upload the representative 50-sentence CER retest dataset."""
import csv
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm25_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

rows = list(csv.DictReader(open('eval/p4_cer50.tsv', encoding='utf-8'), delimiter='\t'))
with open(f'{STAGE}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'text', 'teacher', 'category'])
    for r in rows:
        w.writerow([r['id'], r['text'], r['teacher'], r['category']])
shutil.copy('runs/sm12_in/ref_ms.wav', f'{STAGE}/ref_ms.wav')
json.dump({"title": "sm25 cer50 in", "id": "suarifymalaysia/sm25-cer50-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(rows), 'texts')
api = KaggleApi(); api.authenticate()
r = api.dataset_create_new(folder=STAGE, dir_mode='skip', convert_to_csv=False, public=False)
print('RESULT', r)
