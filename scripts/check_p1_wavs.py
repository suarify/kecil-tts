import csv, os
rows = list(csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'))
have = {f[:-4] for f in os.listdir('runs/sota/p1') if f.endswith('.wav')}
missing = [r['id'] for r in rows if r['id'] not in have]
print('have', len(have), 'of', len(rows), '| first missing:', missing[:5])
print('last rows:', [r['id'] for r in rows[-5:]])
