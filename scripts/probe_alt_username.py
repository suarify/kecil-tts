"""Push a throwaway kernel with the alt token to discover the alt Kaggle username."""
import json
import os
import shutil

from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
probe = os.path.join(os.path.expanduser('~'), '.kaggle', '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
os.environ['KAGGLE_CONFIG_DIR'] = probe

STAGE = os.path.abspath('runs/kaggle_alt_probe')
shutil.rmtree(STAGE, ignore_errors=True)
os.makedirs(STAGE, exist_ok=True)
open(f'{STAGE}/probe.py', 'w').write('print("probe")\n')
json.dump({"id": "suarifymalaysia/altprobe", "title": "username probe delete me", "code_file": "probe.py", "language": "python",
           "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
           "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
           "model_sources": []}, open(f'{STAGE}/kernel-metadata.json', 'w'), indent=2)
api = KaggleApi(); api.authenticate()
try:
    p = api.kernels_push(folder=STAGE)
    print('PUSHED REF', p.ref)
except Exception as e:
    print('PUSH ERR', type(e).__name__, str(e)[:300])
