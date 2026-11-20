"""Test whether the alt Kaggle account can see the suarifymalaysia datasets."""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

TOKEN = 'KGAT_be0b407504dea7b8644b3d68b045d40d'
probe = os.path.join(os.path.expanduser('~'), '.kaggle', '_probe_alt')
os.makedirs(probe, exist_ok=True)
open(os.path.join(probe, 'access_token'), 'w').write(TOKEN)
os.environ['KAGGLE_CONFIG_DIR'] = probe
api = KaggleApi(); api.authenticate()
for ds in ('suarifymalaysia/sm29-nicole-train-in', 'suarifymalaysia/sm-g2p'):
    try:
        fs = api.dataset_list_files(ds).files
        print('CAN SEE', ds, len(fs))
    except Exception as e:
        print('CANNOT SEE', ds, type(e).__name__, str(e)[:100])
