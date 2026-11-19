"""Wait for the sm-21-f0uv Kaggle kernel and download its outputs."""
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = 'suarifymalaysia/sm-21-f0uv-text-head'
OUT = os.path.abspath('runs/kaggle_sm21')
os.makedirs(OUT, exist_ok=True)
api = KaggleApi()
api.authenticate()

TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}
t0 = time.time()
last = None
while time.time() - t0 < 4 * 3600:
    s = api.kernels_status(REF)
    status = getattr(s.status, 'name', None) or str(s.status)
    fail = getattr(s, 'failureMessage', None) or getattr(s, 'failure_message', None)
    if status != last:
        print(f'[{time.strftime("%H:%M:%S")}] status={status} fail={fail}', flush=True)
        last = status
    if status in TERMINAL:
        print('FINAL', status, 'fail=', fail, flush=True)
        if status == 'COMPLETE':
            api.kernels_output(REF, path=OUT)
            print('OUTPUT FILES:', sorted(os.listdir(OUT)), flush=True)
        break
    time.sleep(60)
else:
    print('TIMEOUT waiting for kernel', flush=True)
