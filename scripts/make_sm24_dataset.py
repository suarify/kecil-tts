"""Stage + upload 15 generated wavs + refs for CER scorer verification."""
import csv
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm24_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

plan = list(csv.DictReader(open('eval/p4_new_texts_final.tsv', encoding='utf-8'), delimiter='\t'))[:15]
with open(f'{STAGE}/refs.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'teacher', 'text'])
    for i, r in enumerate(plan):
        w.writerow([str(i), r['teacher'], r['text']])
        shutil.copy(f'runs/kaggle_sm23/p4/A_baseline/{i}.wav', f'{STAGE}/wav_{i}.wav')
json.dump({"title": "sm24 cer verify in", "id": "suarifymalaysia/sm24-cer-verify-in",
           "licenses": [{"name": "CC0-1.0"}]}, open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len([x for x in os.listdir(STAGE) if x.startswith('wav_')]), 'wavs')
api = KaggleApi(); api.authenticate()
r = api.dataset_create_version(folder=STAGE, version_notes='flat wavs', dir_mode='skip', convert_to_csv=False)
print('RESULT', r)
