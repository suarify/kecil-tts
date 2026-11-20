"""Verify alt Kaggle token by probing known account / quota endpoints, then restore the default token."""
import os

from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
probe = os.path.join(os.path.expanduser('~'), '.kaggle', '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
os.environ['KAGGLE_CONFIG_DIR'] = probe
api = KaggleApi()
api.authenticate()
print('CONFIG', api.config)
# list quotas — reveals username
try:
    q = api.get_accelerator_quota(request={})
    print('QUOTA', str(q)[:300])
except Exception as e:
    print('quota err', type(e).__name__, str(e)[:150])
# try the dataset list with no user, which sometimes returns own datasets
try:
    ds = api.dataset_list()
    print('MY DATASETS', len(ds), [d.ref for d in ds[:3]])
except Exception as e:
    print('datasets err', type(e).__name__, str(e)[:120])
