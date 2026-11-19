"""Wait for the sm-22-gta Kaggle kernel and download its outputs."""
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

REF = 'suarifymalaysia/sm-22-gta'
OUT = os.path.abspath('runs/kaggle_sm22')
os.makedirs(OUT, exist_ok=True)
api = KaggleApi()
api.authenticate()
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}
t0 = time.time()
last = None
while time.time() - t0 < 3 * 3600:
    s = None
    for attempt in range(6):
        try:
            s = api.kernels_status(REF)
            break
        except Exception as e:
            print(f'[{time.strftime("%H:%M:%S")}] status retry {attempt}: {type(e).__name__}', flush=True)
            time.sleep(30)
    if s is None:
        continue
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
