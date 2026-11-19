"""Stage and upload the sm21-f0uv Kaggle dataset: new UV scripts + exact 1281 teacher rows + text init."""
import glob
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm21_in')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(f'{STAGE}/scripts', exist_ok=True)
os.makedirs(f'{STAGE}/rows', exist_ok=True)

for p in glob.glob('paradee_ms/*.py'):
    shutil.copy(p, f'{STAGE}/scripts/')
shutil.copy('data/teacher/shard_0000.pt', f'{STAGE}/rows/shard_0000.pt')
shutil.copy('runs/k16_final/textA_c4_last.pt', f'{STAGE}/textA_c4_last.pt')

json.dump({"title": "sm21 f0uv in", "id": "suarifymalaysia/sm21-f0uv-in",
           "licenses": [{"name": "CC0-1.0"}]},
          open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('files:', {p: round(os.path.getsize(f'{STAGE}/{p}') / 1e6, 1)
                 for p in os.listdir(STAGE) if os.path.isfile(f'{STAGE}/{p}')})
print('scripts:', len(os.listdir(f'{STAGE}/scripts')))

api = KaggleApi()
api.authenticate()
api.dataset_create_new(folder=STAGE, dir_mode='tarfile', convert_to_csv=False, public=False)
print('dataset uploaded: suarifymalaysia/sm21-f0uv-in')
