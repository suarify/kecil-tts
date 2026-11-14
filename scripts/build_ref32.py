"""Build eval/ref_32.tsv and the kaggle/ds_ref staging folder for the teacher search (continue.plan.md A0)."""
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, 'runs', 'k03', 'pilot_scores.csv'))
gen = d[(d.category == 'general') & (d.cer <= 0.02) & (d.norm.str.split().str.len() >= 4)].copy()
gen['n'] = gen.norm.str.len()
gen = gen.sort_values('n')
idx = [int(i * (len(gen) - 1) / 23) for i in range(24)]
gen = gen.iloc[idx]
cs = d[(d.category == 'codeswitch') & (d.cer <= 0.05)].sort_values('cer').head(8)
ref = pd.concat([gen, cs])[['id', 'category', 'text', 'norm']]
ref.to_csv(os.path.join(ROOT, 'eval', 'ref_32.tsv'), sep='\t', index=False, lineterminator='\n')
print(len(ref), ref.category.value_counts().to_dict())

out = os.path.join(ROOT, 'kaggle', 'ds_ref')
shutil.rmtree(out, ignore_errors=True)
for sub in ('T2', 'T2v', 'bella_en'):
    os.makedirs(os.path.join(out, sub))
for i in ref.id:
    for sub in ('T2', 'T2v'):
        shutil.copy(os.path.join(ROOT, 'runs', 'k03', 'pilot', sub, f'{i}.wav'), os.path.join(out, sub))
for f in sorted(os.listdir(os.path.join(ROOT, 'runs', 'k01', 'refs', 'bella_en'))):
    shutil.copy(os.path.join(ROOT, 'runs', 'k01', 'refs', 'bella_en', f), os.path.join(out, 'bella_en'))
shutil.copy(os.path.join(ROOT, 'eval', 'ref_32.tsv'), out)
with open(os.path.join(out, 'dataset-metadata.json'), 'w') as f:
    f.write('{"title":"kmb-ref","id":"seanlon/kmb-ref","licenses":[{"name":"other"}]}')
print('staged', sum(len(fs) for _, _, fs in os.walk(out)), 'files')
