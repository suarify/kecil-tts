"""Configure the owner-provided second Kaggle token as a separate profile (does NOT replace the current one).

usage: python scripts/setup_kaggle_alt.py
Creates ~/.kaggle/access_token_alt.json and verifies who the token belongs to.
"""
import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
HOME = os.path.expanduser('~')
KD = os.path.join(HOME, '.kaggle')
os.makedirs(KD, exist_ok=True)

# Kaggle accepts a JSON credentials file with a username + key. Discover the username via the API first
# by temporarily pointing KAGGLE_CONFIG_DIR at a folder with an access_token file.
probe = os.path.join(KD, '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
old = os.environ.get('KAGGLE_CONFIG_DIR')
os.environ['KAGGLE_CONFIG_DIR'] = probe
try:
    api = KaggleApi()
    api.authenticate()
    who = api.whoami() if hasattr(api, 'whoami') else None
    uname = (who or {}).get('username', 'unknown')
    print('ALT TOKEN USER:', uname)
except Exception as e:
    print('AUTH ERR', type(e).__name__, str(e)[:200])
    uname = 'unknown'
finally:
    if old is None:
        os.environ.pop('KAGGLE_CONFIG_DIR', None)
    else:
        os.environ['KAGGLE_CONFIG_DIR'] = old

# Save a reusable credentials JSON + a helper config note (token kept local only)
cred = {'username': uname, 'key': TOKEN}
for p in (os.path.join(KD, 'kaggle-alt.json'), os.path.join(KD, 'access_token_alt')):
    if p.endswith('.json'):
        json.dump(cred, open(p, 'w'), indent=2)
    else:
        open(p, 'w').write(TOKEN)
print('saved alt credentials ->', os.path.join(KD, 'kaggle-alt.json'))
