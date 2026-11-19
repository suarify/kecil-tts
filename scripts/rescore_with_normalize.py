"""Re-score sm25 hypotheses with the CORRECT k18 clean_txt (normalize + strip)."""
import json, re, sys
sys.path.insert(0, 'g2p')
from ms_norm import normalize

r = json.load(open('runs/kaggle_sm25/sm25_report.json'))
def clean(s): return re.sub(r'[^a-z]', '', normalize(s).lower())
def cer(ref, hyp):
    a, b = clean(ref), clean(hyp)
    prev = list(range(len(b)+1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return min(1.0, prev[-1]/max(1,len(a)))

import numpy as np
from collections import defaultdict
by = defaultdict(list)
for x in r['rows']:
    by[x['cat']].append(cer(x['ref'], x['hyp']))
for cat, v in by.items():
    print(cat, len(v), 'median', round(float(np.median(v)),4), 'mean', round(float(np.mean(v)),4))
allv = [c for v in by.values() for c in v]
print('OVERALL median', round(float(np.median(allv)),4), 'mean', round(float(np.mean(allv)),4))
