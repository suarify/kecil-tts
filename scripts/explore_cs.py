import csv, re
from collections import Counter
rows = list(csv.DictReader(open('data/text/corpus_train.tsv', encoding='utf-8'), delimiter='\t'))
used = set()
import torch
used = {r['text'].strip() for r in torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)}
pool = [r for r in rows if r['text'].strip() not in used]
cs = [r for r in pool if r['category'] == 'codeswitch']
# collect words that appear in codeswitch category, then count pool rows sharing them
cw = Counter()
for r in cs:
    for w in re.findall(r'[A-Za-z]{3,}', r['text'].lower()):
        cw[w] += 1
print('codeswitch pool', len(cs), '| distinct long words', len(cw))
print(cw.most_common(40))
# how many general rows contain >=1 of the top-100 codeswitch words
top = {w for w, c in cw.most_common(120)}
general = [r for r in pool if r['category'] not in ('codeswitch', 'numbers')]
hit = [r for r in general if any(w in top for w in re.findall(r'[A-Za-z]{3,}', r['text'].lower()))]
print('general rows with >=1 codeswitch word:', len(hit))
