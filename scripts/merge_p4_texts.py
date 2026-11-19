"""Merge the Phase 4 selections: 2500 greedy + 600 code-switch + 300 numbers = 3400 rows."""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

rows = list(csv.DictReader(open('eval/p4_new_texts.tsv', encoding='utf-8'), delimiter='\t'))
synth = list(csv.DictReader(open('eval/p4_codeswitch_synth.tsv', encoding='utf-8'), delimiter='\t'))
cs_have = [r for r in rows if r['source'] == 'codeswitch']
need = 600 - len(cs_have)
assert need >= 0 and need <= len(synth), (len(cs_have), need, len(synth))
rest = [r for r in rows if r['source'] != 'codeswitch']
final = rest + cs_have + synth[:need]
from collections import Counter
print('counts', Counter(r['source'] for r in final), 'total', len(final))
assert len(final) == 3400
out = 'eval/p4_new_texts_final.tsv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'category', 'text', 'teacher', 'source'], delimiter='\t')
    w.writeheader(); w.writerows(final)
print('wrote', out)
