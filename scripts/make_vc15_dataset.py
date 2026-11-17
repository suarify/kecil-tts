"""Assemble the 15-clip single-conversion POC dataset: am_michael reference + 15 texts (ids from the
teacher shard, so the kernel can pair them with the existing double-converted clips)."""
import csv
import json
import os
import shutil

import torch

OUT = 'runs/sm_vc15_in'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(f'{OUT}/refs', exist_ok=True)
shutil.copy('kaggle/ds_ref/michael_en/ref_michael.wav', f'{OUT}/refs/ref_michael.wav')

rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
by_text = {r['text']: r for r in rows}
wanted = [
    'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.',
    'Kita boleh pergi sekarang atau tunggu sehingga petang.',
    'Booking encik untuk hari Selasa dah confirmed.',
    'Azman di sini, saya mewakili pasukan khidmat pelanggan.',
    'Sila maklumkan kehadiran puan sebelum hari Rabu.',
    'Kod pengesahan anda ialah 169484.',
    'Berat bungkusan itu 30 kg.',
    'Permohonan cuti Dato telah diluluskan oleh penyelia.',
    'Delivery cik akan sampai sebelum pukul empat setengah petang.',
    'Kami telah menerima aduan doktor dan sedang menyiasatnya.',
]
sel = [by_text[t] for t in wanted if t in by_text]
# top up from the shard if any text differs slightly
if len(sel) < 15:
    for r in rows:
        if r['id'] not in {s['id'] for s in sel}:
            sel.append(r)
        if len(sel) == 15:
            break

with open(f'{OUT}/texts.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'text'])
    for r in sel:
        text = by_text.get(r['text'], r)['text'] if isinstance(r, dict) else r
        w.writerow([r['id'], r['text']])
print('texts:', len(sel))
json.dump({'title': 'sm vc15 in', 'id': 'suarifymalaysia/sm-vc15-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
for r in sel:
    print(' ', r['id'], r['text'][:60])
