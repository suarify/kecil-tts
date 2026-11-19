import csv, re, torch
rows = list(csv.DictReader(open('data/text/corpus_train.tsv', encoding='utf-8'), delimiter='\t'))
used = {r['text'].strip() for r in torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)}
pool = [r for r in rows if r['text'].strip() not in used]
import importlib.util
spec = importlib.util.spec_from_file_location('m', 'scripts/make_p4_texts.py')
# just reuse regex literal
EN = re.compile(r'\b(booking|confirm|confirmed|appointment|download|resit|transfer|save|order|delivery|stock|report|'
                r'meeting|schedule|login|password|email|phone|office|team|project|budget|invoice|claim|loan|card|cash|'
                r'account|check|double|update|send|reply|call|message|link|file|data|system|service|customer|agent|'
                r'event|deadline|submission|submit|payment|online|whatsapp|driver|router|internet|slow|restart|profile|'
                r'sorry|confirmation|status|request|approve|approved|reject|claim|form|register|registration|'
                r'promo|discount|voucher|coupon|shipping|tracking|parcel|courier|refund|receipt|price|deal|'
                r'chat|group|post|share|like|comment|story|reel|video|stream|live|play|pause|next|previous)\b', re.I)
for cat in ('codeswitch', 'general', 'conversational', 'short', 'question'):
    hits = [r for r in pool if r['category'] == cat and EN.search(r['text'])]
    print(cat, 'pool', sum(1 for r in pool if r['category'] == cat), 'hits', len(hits))
