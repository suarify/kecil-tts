"""Verify the alt token and get its username via the Kaggle SDK config API."""
import os

from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
probe = os.path.join(os.path.expanduser('~'), '.kaggle', '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
os.environ['KAGGLE_CONFIG_DIR'] = probe
api = KaggleApi()
api.authenticate()
for m in ('whoami', 'get_user', 'config'):
    print(m, hasattr(api, m))
try:
    print('WHOAMI', api.whoami())
except Exception as e:
    print('whoami err', type(e).__name__, str(e)[:150])
# try to find a kernel by listing (reveals owner)
for uname in ('kgat_be0b407504dea7b8644b3d68b045d40d',):
    try:
        k = api.kernels_list(user=uname)
        print('LIST', uname, len(k))
    except Exception as e:
        print('list err', type(e).__name__, str(e)[:120])
