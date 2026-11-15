"""C3.1: extract error-driven word candidates from Whisper mishearings (teacher_clips.csv / test_scores.csv).

For every row, tokens of the reference text that are missing from the heard transcript (or heard as a
close edit <= 2 edits) are mispronunciation candidates. Output: one word per line, sorted by how often
it is misheard, with a tag (english-in-lexicon / english-not-in-lexicon / malay / other).

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/error_words.py CSV... --out FILE
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'lex_src')
G2P = os.path.join(ROOT, 'g2p')

ap = argparse.ArgumentParser()
ap.add_argument('csvs', nargs='+')
ap.add_argument('--out', default=None)
A = ap.parse_args()

tok = re.compile(r"[a-zA-Z']+")
current = json.load(open(os.path.join(G2P, 'lexicon_en.json'), encoding='utf-8'))
en10k = {w.strip().lower() for w in open(os.path.join(SRC, 'en_10k.txt'), encoding='utf-8')}

malay = set()
for path in ('msa_latn_broad.tsv',):
    for line in open(os.path.join(SRC, path), encoding='utf-8'):
        malay.add(line.split('\t')[0].lower())
for line in open(os.path.join(SRC, 'ms_50k.txt'), encoding='utf-8'):
    malay.add(line.split(' ')[0].strip().lower())

def lev(a, b, cap=3):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

miss = collections.Counter()      # text token totally absent from heard
close = collections.Counter()     # text token heard as something <= 2 edits away
examples = collections.defaultdict(list)
for path in A.csvs:
    for row in csv.DictReader(open(path, encoding='utf-8')):
        text_toks = [w.lower() for w in tok.findall(row['text'])]
        heard_toks = [w.lower() for w in tok.findall(row.get('heard', ''))]
        heard_set = set(heard_toks)
        for w in text_toks:
            if w in heard_set or len(w) < 2:
                continue
            near = [h for h in heard_set if lev(w, h) <= 2]
            if near:
                close[w] += 1
                if len(examples[w]) < 2:
                    examples[w].append(f"{w}->{near[0]}")
            else:
                miss[w] += 1

def tag(w):
    if w in current:
        return 'in-lexicon'
    if w in malay:
        return 'malay'
    if w in en10k:
        return 'english'
    return 'other'

rows = []
for w, n in miss.most_common():
    rows.append((n, w, 'MISS', tag(w), ' '.join(examples.get(w, []))))
for w, n in close.most_common():
    rows.append((n, w, 'CLOSE', tag(w), ' '.join(examples.get(w, []))))
rows.sort(key=lambda x: (-x[0], x[2], x[1]))
print(f"{'n':>4} {'kind':5} {'tag':12} word  example")
for n, w, k, t, ex in rows[:80]:
    print(f"{n:>4} {k:5} {t:12} {w:22} {ex}")
if A.out:
    with open(A.out, 'w', encoding='utf-8') as f:
        for n, w, k, t, ex in rows:
            f.write(f'{n}\t{k}\t{t}\t{w}\t{ex}\n')
    print('wrote', A.out)
