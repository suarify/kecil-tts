"""Pick a representative 50-sentence CER test set from the 3,400 Phase 4 sentences."""
import csv
import random
from collections import Counter

rows = list(csv.DictReader(open('eval/p4_new_texts_final.tsv', encoding='utf-8'), delimiter='\t'))
print('categories', Counter(r['category'] for r in rows))
by = {}
for r in rows:
    by.setdefault((r['category'], r['source']), []).append(r)
for k, v in by.items():
    print(k, len(v))

rng = random.Random(5050)
# stratified: general 20, conversational 10, codeswitch 10, numbers 10
sel = []
for cat, n in (('general', 20), ('conversational', 10), ('codeswitch', 10), ('numbers', 10)):
    pool = [r for r in rows if r['category'] == cat]
    if len(pool) < n:
        pool = [r for r in rows if r['category'] == 'general']
    sel += rng.sample(pool, min(n, len(pool)))
rng.shuffle(sel)
assert len(sel) == 50
out = 'eval/p4_cer50.tsv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'category', 'text', 'teacher', 'source'])
    for r in sel:
        w.writerow([r['id'], r['category'], r['text'], r['teacher'], r['source']])
print('wrote', out, Counter(r['category'] for r in sel))
