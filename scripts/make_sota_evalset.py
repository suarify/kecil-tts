"""Phase 0 fixed eval set: 40 general + 20 code-switch + 20 numbers (stable ids)."""
import csv
import os

rows = list(csv.DictReader(open('data/text/test_200.tsv', encoding='utf-8'), delimiter='\t'))
by = {}
for r in rows:
    by.setdefault(r['category'], []).append(r)
sel = by['general'][:40] + by['codeswitch'][:20] + by['numbers'][:20]
os.makedirs('eval', exist_ok=True)
with open('eval/sota_eval.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'category', 'text'])
    for r in sel:
        w.writerow([r['id'], r['category'], r['text']])
print('wrote', len(sel), 'sentences:', {c: sum(1 for r in sel if r['category'] == c) for c in by})
