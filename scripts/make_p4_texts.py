"""Phase 4 step 1: select 2,500 greedy phoneme-bigram coverage + 600 code-switch + 300 number sentences.

Excludes the 1,281 texts already used. Output: eval/p4_new_texts.tsv with id/category/text/teacher/source.
"""
import csv
import os
import re

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

used = {r['text'].strip() for r in torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)}
rows = list(csv.DictReader(open('data/text/corpus_train.tsv', encoding='utf-8'), delimiter='\t'))
pool = [r for r in rows if r['text'].strip() not in used]
print('corpus', len(rows), '| already used', len(used), '| pool', len(pool))

# ---------- 2,500 by greedy phoneme-bigram coverage ----------
base = [r for r in pool if r['category'] not in ('codeswitch', 'numbers')]
# token bigrams (space-split phonemes) have a ~40k space; char bigrams saturate after ~250 sentences
bigrams = {r['id']: set(zip(r['phonemes'].split(), r['phonemes'].split()[1:])) for r in base}
by_id = {r['id']: r for r in base}
covered = set()
chosen_ids = set()
chosen = []
for _ in range(2500):
    best, bestgain = None, -1
    for rid, bg in bigrams.items():
        if rid in chosen_ids:
            continue
        gain = len(bg - covered)
        if gain > bestgain:
            best, bestgain = rid, gain
    if best is None or bestgain <= 0:
        break
    chosen.append(by_id[best])
    chosen_ids.add(best)
    covered |= bigrams[best]
print('greedy chosen', len(chosen), '| bigrams covered', len(covered))

# ---------- 600 code-switch (corpus codeswitch + English-loanword general) ----------
EN = re.compile(r'\b(booking|confirm|confirmed|appointment|download|resit|transfer|save|order|delivery|stock|report|'
                r'meeting|schedule|login|password|email|phone|office|team|project|budget|invoice|claim|loan|card|cash|'
                r'account|check|double|update|send|reply|call|message|link|file|data|system|service|customer|agent|'
                r'event|deadline|submission|submit|payment|online|whatsapp|driver|router|internet|slow|restart|profile|'
                r'sorry|driver|confirmation|status|request|approve|approved|reject|claim|form|register|registration|'
                r'promo|discount|voucher|coupon|delivery|shipping|tracking|parcel|courier|refund|receipt|price|deal|'
                r'chat|group|post|share|like|comment|story|reel|video|stream|live|play|pause|next|previous)\b', re.I)
cs_pool = [r for r in pool if r['category'] == 'codeswitch' or EN.search(r['text'])]
cs_pool = [r for r in cs_pool if r not in chosen]
chosen_cs = cs_pool[:600]
print('codeswitch chosen', len(chosen_cs), '(corpus codeswitch available:', sum(1 for r in pool if r['category'] == 'codeswitch'), ')')

# ---------- 300 numbers / dates / RM / percent / abbreviations ----------
num_pool = [r for r in pool if r['category'] == 'numbers' and r not in chosen and r not in chosen_cs]
chosen_num = num_pool[:300]
print('numbers chosen', len(chosen_num))

# ---------- write ----------
out = 'eval/p4_new_texts.tsv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'category', 'text', 'teacher', 'source'])
    for i, r in enumerate(chosen):
        w.writerow([f'p4g{i:05d}', r['category'], r['text'], r['teacher'], 'greedy_bigram'])
    for i, r in enumerate(chosen_cs):
        w.writerow([f'p4c{i:05d}', r['category'], r['text'], r['teacher'], 'codeswitch'])
    for i, r in enumerate(chosen_num):
        w.writerow([f'p4n{i:05d}', r['category'], r['text'], r['teacher'], 'numbers'])
print('wrote', out, '| total', len(chosen) + len(chosen_cs) + len(chosen_num))
