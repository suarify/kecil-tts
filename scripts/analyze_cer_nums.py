"""Diagnostic: how much of the CER is number-format mismatch?"""
import json, re
r = json.load(open('runs/kaggle_sm24/sm24_report.json'))
NUMWORDS = set('kosong satu dua tiga empat lima enam tujuh lapan sembilan sepuluh sebelas dua belas tiga belas '
               'empat belas lima belas enam belas tujuh belas lapan belas sembilan belas dua puluh tiga puluh '
               'empat puluh lima puluh enam puluh tujuh puluh lapan puluh sembilan puluh seratus ribu juta '
               'perpuluhan peratus puluh ratus ribu'.split())

def clean(s): return re.sub(r'[^a-z]', '', s.lower())
def strip_nums(s):
    toks = [w for w in re.findall(r'[a-z]+|\d+', s.lower()) if w not in NUMWORDS and not w.isdigit()]
    return ''.join(toks)

def cer(a, b):
    prev = list(range(len(b)+1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca!=cb)))
        prev = cur
    return min(1.0, prev[-1]/max(1,len(a)))

vals = []
for x in r['rows']:
    ref = x['ref']  # note: truncated to 90 chars in the kernel report
    hyp = x['pipe_hyp']
    vals.append((x['pipe_cer'], cer(strip_nums(ref), strip_nums(hyp))))
print('raw median', round(sorted(v[0] for v in vals)[len(vals)//2],4))
print('num-stripped median', round(sorted(v[1] for v in vals)[len(vals)//2],4))
print('per clip (raw -> stripped):')
for a,b in vals: print(' ',round(a,3),'->',round(b,3))
