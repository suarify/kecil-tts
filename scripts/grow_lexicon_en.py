"""C3.2: grow g2p/lexicon_en.json from ~430 toward ~3,000 English words.

Candidate sources, in priority order:
  1. --errors FILE  (one word per line): words Whisper is heard to mispronounce from the k14
     kernel CSVs (teacher_clips.csv / test_scores.csv). Highest priority.
  2. corpus_train.tsv tokens (7,500 unseen Malaysian sentences), weighted by frequency and by
     how often they appear in code-switch/general sentences.
  3. data/lex_src/en_10k.txt: general English frequency list (google-10000, no swears).

All candidates must have a pronunciation in misaki's us_gold.json / us_silver.json (same sources
build_lexicon_en.py uses) with every symbol inside g2p/vocab.json. Every word already in the
lexicon is kept. The Malay exclusion set (msa_latn_broad + ms_50k + schwa_dict + lexicon_e +
review overrides, ~96k) also contains English loans written inside Malay text; a candidate
inside that set is kept only if it is in KEEP, came from the error list, or appears >= 2 times
in code-switch corpus sentences (the ONLY category that counts as English context - 'general'
text is mostly Malay and words like unit/rom/vitamin must keep their Malay pronunciation; the
golden test + parity.mjs caught exactly that mistake in the first draft).

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/grow_lexicon_en.py [--target 3000] [--dry-run]
        [--errors FILE] [--min-freq 3]
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
ap.add_argument('--target', type=int, default=3000)
ap.add_argument('--min-freq', type=int, default=3)
ap.add_argument('--errors', default=None, help='file with one error word per line (top priority)')
ap.add_argument('--dry-run', action='store_true')
A = ap.parse_args()

tok = re.compile(r"[a-zA-Z]+")
ms50k = {}
counts = collections.Counter()
cs_counts = collections.Counter()  # code-switch category ONLY: real English context
cap_counts = collections.Counter()
with open(os.path.join(ROOT, 'kaggle/ds_g2p/corpus_train.tsv'), encoding='utf-8') as f:
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3 or parts[0] == 'id':
            continue
        cat, text = parts[1], parts[2]
        for w in tok.findall(text):
            lw = w.lower()
            counts[lw] += 1
            if w[0].isupper():
                cap_counts[lw] += 1
            if cat == 'code-switch':
                cs_counts[lw] += 1
print('distinct tokens:', len(counts))

malay = set()
for line in open(os.path.join(SRC, 'msa_latn_broad.tsv'), encoding='utf-8'):
    malay.add(line.split('\t')[0].lower())
for line in open(os.path.join(SRC, 'ms_50k.txt'), encoding='utf-8'):
    bits = line.split(' ')
    ms50k[bits[0].strip().lower()] = int(bits[1]) if len(bits) > 1 else 0
    malay.add(bits[0].strip().lower())
with open(os.path.join(SRC, 'schwa_dict.csv'), encoding='utf-8') as f:
    for row in csv.reader(f):
        if row:
            malay.add(row[0].lower())
malay |= set(json.load(open(os.path.join(G2P, 'lexicon_e.json'), encoding='utf-8')))
with open(os.path.join(SRC, 'review_top.tsv'), encoding='utf-8') as f:
    for line in f:
        malay.add(line.split('\t')[0].lower())
current = json.load(open(os.path.join(G2P, 'lexicon_en.json'), encoding='utf-8'))
print('malay set', len(malay), '| current lexicon_en', len(current))

# Hand-curated: words used constantly in Malay text but absent from ms_50k (names, countries,
# religion, honorifics, Malay months). English pronunciation would be wrong, so never add them.
MALAY_ALSO = set("""
malaysia singapore brunei jakarta bangkok mekah madinah indonesia thailand vietnam philippine filipina
china india japan korea australia england british british america brazil portugal spain italy france
germany russia iran iraq pakistan afghanistan arabia arab europe asia africa eropa yaman syria
palestin israel timur tengah sabah sarawak selangor johor melaka kedah kelantan terengganu perak
pahang melbourne boston london paris tokyo beijing sydney dubai cairo istanbul moscow berlin madrid
belgium britain switzerland hungary norway denmark sweden nigeria taiwan bangladesh mexico amsterdam
islam allah muslim muslims kristian buddha hindu quran hadith doa iman syariah umrah haji hajah
april november jun bin al mas dr ts ir hj
meter hospital diet petroleum mac pas tan sri sing tun penang dayak mat tamil guru data naskah wahyu muktamar olimpik
""".split())
malay |= MALAY_ALSO
no_bypass = MALAY_ALSO | {w for w, c in ms50k.items() if c >= 2000}

gold = json.load(open(os.path.join(SRC, 'us_gold.json'), encoding='utf-8'))
silver = json.load(open(os.path.join(SRC, 'us_silver.json'), encoding='utf-8'))
KEEP = {'the', 'and', 'you', 'okay', 'sorry', 'please', 'thanks', 'meeting', 'booking', 'online',
        'cancel', 'confirm', 'password', 'check', 'order', 'email', 'manager', 'problem', 'pending',
        'open', 'edit', 'member', 'target', 'laptop', 'printer', 'transfer', 'register', 'item', 'we'}

# general English frequency list (rank 0 = most frequent)
en10k = {}
for i, line in enumerate(open(os.path.join(SRC, 'en_10k.txt'), encoding='utf-8')):
    w = line.strip().lower()
    if w and w not in en10k:
        en10k[w] = i
print('en_10k entries:', len(en10k))


def pron(w):
    ph = gold.get(w, silver.get(w))
    if isinstance(ph, dict):
        ph = ph.get('DEFAULT') or next(iter(ph.values()))
    if not ph and w.endswith('s') and len(w) > 2 and w[:-1] in gold:
        b = gold[w[:-1]]
        b = b.get('DEFAULT') if isinstance(b, dict) else b
        ph = b + ('s' if b[-1] in 'ptkfθ' else 'z')
    if not ph and w.endswith('ed') and len(w) > 3 and w[:-2] in gold:
        b = gold[w[:-2]]
        b = b.get('DEFAULT') if isinstance(b, dict) else b
        ph = b + ('ᵻd' if b[-1] in 'td' else ('t' if b[-1] in 'pkfsʃʧθ' else 'd'))
    return ph


vocab = json.load(open(os.path.join(G2P, 'vocab.json'), encoding='utf-8'))

# candidate universe: corpus tokens + en_10k words (+ error words on top)
err_set = set()
if A.errors:
    err_set = {w.strip().lower() for w in open(A.errors, encoding='utf-8') if w.strip()}
    print('error words given:', len(err_set))
universe = set(counts) | set(en10k) | err_set

cands = {}  # w -> (cs_count, corpus_count, en10k_rank)
for w in universe:
    if w in current or len(w) < 2 or not w.isalpha():
        continue
    cs, n = cs_counts.get(w, 0), counts.get(w, 0)
    if w in malay and w not in KEEP and w not in err_set and not (
            w in en10k and cs >= 2 and w not in no_bypass):
        continue  # Malay always wins; loans bypass only with English-list + code-switch evidence
    if cap_counts.get(w, 0) * 2 > n and w not in KEEP and w not in err_set and cs < 3:
        continue  # proper noun (name/place) -> English respelling would be wrong
    if n < A.min_freq and w not in err_set:
        continue  # needs corpus evidence (en10k-only words: too risky, may be Malay proper nouns)
    ph = pron(w)
    if not ph or any(c not in vocab for c in ph):
        continue
    cands[w] = (cs, n, en10k.get(w, len(en10k)))
err_list = [w for w in err_set if w in cands]
rest = sorted((w for w in cands if w not in err_set),
              key=lambda w: (-cands[w][0], -cands[w][1], cands[w][2], w))
print('candidates with a usable misaki pronunciation:', len(cands),
      f'({len(err_list)} error-driven, {len(rest)} frequency-driven)')
room = max(0, A.target - len(current))
new_words = (err_list + rest)[:room]
print(f'adding {len(new_words)} (target {A.target})')
print('ALL NEW:', ' '.join(new_words))

WORDS = ' '.join(sorted(set(list(current) + new_words)))
src = open(os.path.join(ROOT, 'scripts/build_lexicon_en.py'), encoding='utf-8').read()
src = re.sub(r'WORDS = """.*?"""', 'WORDS = """\n' + WORDS + '\n"""', src, flags=re.S)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location('ble', os.path.join(ROOT, 'scripts', '_grow_tmp.py'))
open(spec.origin, 'w', encoding='utf-8').write(src)
ble = importlib.util.module_from_spec(spec)
sys.argv = ['ble']
spec.loader.exec_module(ble)
os.remove(spec.origin)
if not A.dry_run:
    ble.main(os.path.join(SRC, 'us_gold.json'), os.path.join(SRC, 'us_silver.json'))
    json.dump(json.load(open(os.path.join(G2P, 'lexicon_en.json'), encoding='utf-8')),
              open(os.path.join(ROOT, 'web/g2p/lexicon_en.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)
    print('synced web/g2p/lexicon_en.json')
