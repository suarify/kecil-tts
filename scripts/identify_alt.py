"""Identify the alt Kaggle account by scanning the user's kernels via the SDK."""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
probe = os.path.join(os.path.expanduser('~'), '.kaggle', '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
os.environ['KAGGLE_CONFIG_DIR'] = probe
api = KaggleApi(); api.authenticate()
# kernels_list without user returns own kernels when authenticated
for kw in ({}, {'user': ''}, {'mine': True}):
    try:
        k = api.kernels_list(**kw)
        if k:
            print('OWN KERNELS', len(k), [(getattr(x, 'ref', ''), getattr(x, 'owner', '')) for x in k[:3]])
            break
    except Exception as e:
        print('err', type(e).__name__, str(e)[:100])
# quota via the new SDK client if present
try:
    from kagglesdk.kernels.kernels_api_service import KernelsApiService
    print('new sdk available')
except Exception as e:
    print('no new sdk', e)
