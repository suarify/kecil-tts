"""Build the one-VC-vs-two-VC dataset: 15 raw Chatterbox clips + their current double-converted male
versions + the am_michael reference. Kernel: ChatterboxVC(raw -> am_michael) = one conversion."""
import csv
import json
import os
import shutil

import numpy as np
import soundfile as sf
import torch

OUT = 'runs/sm_vc15b_in'
shutil.rmtree(OUT, ignore_errors=True)
for d in ('raw', 'old'):
    os.makedirs(f'{OUT}/{d}')
shutil.copy('kaggle/ds_ref/michael_en/ref_michael.wav', f'{OUT}/ref_michael.wav')

raw = torch.load('runs/k15_v3/cb_raw_all.pt', map_location='cpu', weights_only=False)
old = torch.load('runs/kaggle_sm05/cb_michael2.pt', map_location='cpu', weights_only=False)
by_id = {}
for k in raw:
    by_id.setdefault(k.rsplit('_', 1)[0], sorted((kk for kk in raw if kk.startswith(k.rsplit('_', 1)[0] + '_'))))

# pick a spread: prefer varied texts
want = ['c0313', 'c0551', 'c0571', 'c0726', 'c0746', 'c0747', 'c0184', 'c0685', 'c0567', 'c0075']

texts = {}
for r in csv.DictReader(open('runs/sm_vc15_in/texts.csv', encoding='utf-8')):
    texts[r['id']] = r['text']
# texts.csv had the old ids; map the new ids from the teacher plan
import csv as _csv  # noqa: E402
for r in _csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8')):
    if r['id'] in by_id and r['id'] not in texts:
        texts[r['id']] = r['text']

ids = [i for i in want if i in by_id and i in old][:15]
for i in by_id:
    if len(ids) >= 15:
        break
    if i in old and i not in ids:
        ids.append(i)

for i in ids:
    k = by_id[i][0]
    a = raw[k].numpy().astype(np.float32) / 32767.0
    sf.write(f'{OUT}/raw/{i}.wav', a, 24000)
    b = old[i].numpy().astype(np.float32) / 32767.0
    sf.write(f'{OUT}/old/{i}.wav', b, 24000)

with open(f'{OUT}/texts.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'text'])
    for i in ids:
        w.writerow([i, texts.get(i, '')])

json.dump({'title': 'sm vc15b in', 'id': 'suarifymalaysia/sm-vc15b-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
print('clips:', len(ids), '| files:', sum(len(fs) for _, _, fs in os.walk(OUT)))
for i in ids[:6]:
    print(' ', i, texts.get(i, '')[:60])
