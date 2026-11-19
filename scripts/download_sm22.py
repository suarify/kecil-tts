"""Download sm-22-gta kernel outputs with retry."""
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = 'suarifymalaysia/sm-22-gta'
OUT = os.path.abspath('runs/kaggle_sm22')
os.makedirs(OUT, exist_ok=True)
api = KaggleApi()
api.authenticate()
for attempt in range(8):
    try:
        api.kernels_output(REF, path=OUT)
        print('OUTPUT FILES:', sorted(os.listdir(OUT)))
        break
    except Exception as e:
        print('retry', attempt, type(e).__name__, str(e)[:120], flush=True)
        time.sleep(30)
else:
    raise SystemExit('download failed')
