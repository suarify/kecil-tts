"""Re-upload the sm21 dataset with all files flat at the root (Kaggle API skipped subdirectories)."""
import glob
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm21_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)

for p in glob.glob('paradee_ms/*.py'):
    shutil.copy(p, os.path.join(STAGE, os.path.basename(p)))
shutil.copy('data/teacher/shard_0000.pt', f'{STAGE}/shard_0000.pt')
shutil.copy('runs/k16_final/textA_c4_last.pt', f'{STAGE}/textA_c4_last.pt')
json.dump({"title": "sm21 f0uv in", "id": "suarifymalaysia/sm21-f0uv-in",
           "licenses": [{"name": "CC0-1.0"}]},
          open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(os.listdir(STAGE)), 'files:', sorted(os.listdir(STAGE)))

api = KaggleApi()
api.authenticate()
r = api.dataset_create_version(folder=STAGE, version_notes='flat: scripts + 1281 rows + text init',
                               dir_mode='skip', convert_to_csv=False)
print('RESULT', r)
fs = api.dataset_list_files('suarifymalaysia/sm21-f0uv-in').files
print('KAGGLE FILES', len(fs), [f.name for f in fs])
