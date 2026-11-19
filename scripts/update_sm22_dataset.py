"""Re-upload sm22-gta-in with the updated GAN-capable ms_train.py."""
import glob
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
STAGE = os.path.abspath('runs/kaggle_sm22_gan_flat')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)
for p in glob.glob('paradee_ms/*.py'):
    shutil.copy(p, os.path.join(STAGE, os.path.basename(p)))
shutil.copy('data/teacher_gta/shard_0000.pt', f'{STAGE}/shard_0000.pt')
shutil.copy('runs/k16_final/dec_c4_last.pt', f'{STAGE}/dec_c4_last.pt')
json.dump({"title": "sm22 gta in", "id": "suarifymalaysia/sm22-gta-in",
           "licenses": [{"name": "CC0-1.0"}]},
          open(f'{STAGE}/dataset-metadata.json', 'w'), indent=2)
print('staged', len(os.listdir(STAGE)))
api = KaggleApi(); api.authenticate()
r = api.dataset_create_version(folder=STAGE, version_notes='GAN-capable trainer for Phase 3',
                               dir_mode='skip', convert_to_csv=False)
print('RESULT', r)
