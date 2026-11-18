"""Dataset for Path-B full run: 1,281 Yati student clips + nicole reference."""
import os
import shutil
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = 'runs/pathb_yati_clips'
OUT = 'runs/sm15c_in'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
for f in os.listdir(SRC):
    if f.endswith('.wav'):
        shutil.copy(f'{SRC}/{f}', f'{OUT}/{f}')
shutil.copy('runs/sm12_in/ref_nicole.wav', f'{OUT}/ref_nicole.wav')
json.dump({'title': 'sm15c in', 'id': 'suarifymalaysia/sm15c-in', 'licenses': [{'name': 'apache-2.0'}]},
          open(f'{OUT}/dataset-metadata.json', 'w', encoding='utf-8'), indent=1)
print('files:', len(os.listdir(OUT)) - 1, 'wavs')
