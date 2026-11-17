"""Assemble the dataset for the new-voice kernel: all 1,281 texts + Malay generation reference + nicole reference."""
import csv
import json
import os
import shutil

import torch

OUT = 'runs/sm12_in'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
with open(f'{OUT}/texts.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'teacher', 'text'])
    for r in rows:
        w.writerow([r['id'], r['text'], r['text']])
print('texts:', len(rows))

shutil.copy('runs/k01/refs/ref_ms.wav', f'{OUT}/ref_ms.wav')
shutil.copy('runs/refs_new/ref_af_nicole.wav', f'{OUT}/ref_nicole.wav')
json.dump({'title': 'sm12 in', 'id': 'suarifymalaysia/sm12-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
print('files:', os.listdir(OUT))
