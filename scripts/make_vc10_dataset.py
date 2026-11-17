"""Build the 10-clip conversion-quality test: raw clips + two am_michael references (ours and the pipeline's)."""
import csv
import json
import os
import shutil

import numpy as np
import soundfile as sf
import torch

OUT = 'runs/sm_vc10_in'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

# references: ours (Kokoro am_michael, 25 s) and the one the pipeline used
shutil.copy('kaggle/ds_ref/michael_en/ref_michael.wav', f'{OUT}/ref_a_michael.wav')
shutil.copy('runs/kaggle_sm05/ref_en_michael.wav', f'{OUT}/ref_b_michael.wav')

raw = torch.load('runs/k15_v3/cb_raw_all.pt', map_location='cpu', weights_only=False)
by_id = {}
for k in raw:
    by_id.setdefault(k.rsplit('_', 1)[0], sorted(kk for kk in raw if kk.startswith(k.rsplit('_', 1)[0] + '_')))

# texts chosen to expose artefacts: long vowels, diphthongs, fricatives, code-switch
plan = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
want_texts = [
    'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.',
    'Saya rasa susah sangat nak faham apa yang dia cakap.',
    'Kami akan sampai sekitar pukul tiga petang nanti.',
    'Perlahan-lahan dia membuka pintu itu dan melihat ke luar.',
    'Boleh encik ulang sekali lagi nombor telefon itu?',
    'Pesanan pizza itu sudah sampai di rumah saya semalam.',
    'Zaman sekarang semua orang guna telefon pintar.',
    'Sila ambil masa anda untuk memikirkan tawaran ini.',
    'Wang sebanyak tiga ribu ringgit telah dipindahkan.',
    'Terima kasih kerana sudi datang ke majlis kami.',
]
ids = []
for t in want_texts:
    for i in by_id:
        if plan.get(i) == t and i in by_id:
            ids.append(i)
            break
if len(ids) < 10:
    for i in by_id:
        if i not in ids:
            ids.append(i)
        if len(ids) == 10:
            break

for i in ids:
    a = raw[by_id[i][0]].numpy().astype(np.float32) / 32767.0
    sf.write(f'{OUT}/raw_{i}.wav', a, 24000)

with open(f'{OUT}/texts.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'text'])
    for i in ids:
        w.writerow([i, plan.get(i, '')])

json.dump({'title': 'sm vc10 in', 'id': 'suarifymalaysia/sm-vc10-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
print('clips:', len(ids), '| files:', len(os.listdir(OUT)))
for i in ids[:10]:
    print(' ', i, plan.get(i, '')[:66])
