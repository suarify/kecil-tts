"""Build g2p/lexicon_e.json: word -> respelling, 'é' = e taling [e], 'e' = e pepet [ə].

Sources, highest priority first:
  1. g2p/lexicon_e_overrides.txt   hand-written Malay labels ("word respelling" per line)
  2. WikiPron msa_latn_broad.tsv    Malay Wiktionary IPA (CC-BY-SA source data)
  3. g2p-id schwa_dict.csv          Indonesian, MIT ('ê' = pepet; words absent have no pepet)
  4. WikiPron ind_latn_broad.tsv    Indonesian Wiktionary IPA

Also writes data/lex_src/review_top.tsv: the most frequent Malay e-words with the
chosen label, its source and any disagreement, for hand-checking.
"""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'lex_src')
sys.path.insert(0, os.path.join(ROOT, 'g2p'))

PEPET = {'ə', 'ɘ', 'ɨ'}
TALING = {'e', 'ɛ', 'e̞', 'ɛ̝'}
IPA_VOWELS = PEPET | TALING | {'a', 'i', 'o', 'u', 'ɔ', 'ɪ', 'ʊ', 'ɐ', 'æ', 'ɑ', 'ə̆'}
WORD_OK = re.compile(r'^[a-z]+$')


def clean_ipa(tok):
    return tok.replace('ː', '').replace('̯', '').replace('ˈ', '').replace('ˌ', '')


def label_from_ipa(word, ipa):
    """Return respelling or None when the vowels cannot be aligned unambiguously."""
    toks = [clean_ipa(t) for t in ipa.split()]
    nuclei = []
    for j, t in enumerate(toks):
        if t in IPA_VOWELS:
            nuclei.append((j, t))
    vpos = [i for i, c in enumerate(word) if c in 'aeiou']
    if len(vpos) != len(nuclei):
        # merge orthographic diphthongs ai/au/oi when IPA shows a glide
        merged = []
        k = 0
        while k < len(vpos):
            i = vpos[k]
            if (k + 1 < len(vpos) and vpos[k + 1] == i + 1 and word[i:i + 2] in ('ai', 'au', 'oi', 'ei')):
                merged.append(i)
                k += 2
            else:
                merged.append(i)
                k += 1
        vpos = merged
        if len(vpos) != len(nuclei):
            return None
    chars = list(word)
    for i, (_, v) in zip(vpos, nuclei):
        if word[i] != 'e':
            continue
        if v in PEPET:
            chars[i] = 'e'
        elif v in TALING:
            chars[i] = 'é'
        else:
            return None
    return ''.join(chars)


def load_wikipron(name):
    out = {}
    path = os.path.join(SRC, name)
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            w = parts[0].lower()
            if not WORD_OK.match(w) or 'e' not in w or w in out:
                continue
            r = label_from_ipa(w, parts[1])
            if r:
                out[w] = r
    return out


def load_schwa():
    out = {}
    with open(os.path.join(SRC, 'schwa_dict.csv'), encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0] == 'word':
                continue
            w, s = row[0].lower(), row[1]
            if not WORD_OK.match(w):
                continue
            s2 = []
            for ch in s:
                if ch in 'êĕěȇӗ':
                    s2.append('e')
                elif ch in 'eéè':
                    s2.append('é')
                else:
                    s2.append(ch.lower())
            s2 = ''.join(s2)
            if s2.replace('é', 'e') != w:
                continue
            out[w] = s2
    return out


def load_overrides():
    out = {}
    path = os.path.join(ROOT, 'g2p', 'lexicon_e_overrides.txt')
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            for item in line.split(','):
                item = item.strip()
                if not item:
                    continue
                spelled = item
                w = spelled.replace('é', 'e')
                out[w] = spelled
    return out


def main():
    ov = load_overrides()
    msa = load_wikipron('msa_latn_broad.tsv')
    schwa = load_schwa()
    ind = load_wikipron('ind_latn_broad.tsv')
    print('overrides', len(ov), 'msa', len(msa), 'schwa', len(schwa), 'ind', len(ind))

    lex, source = {}, {}
    for name, d in (('ind', ind), ('schwa', schwa), ('msa', msa), ('override', ov)):
        for w, s in d.items():
            lex[w] = s
            source[w] = name
    lex = {w: s for w, s in lex.items() if 'e' in w}
    with open(os.path.join(ROOT, 'g2p', 'lexicon_e.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(lex.items())), f, ensure_ascii=False, indent=0)
    print('lexicon_e.json entries:', len(lex))

    # Review list: frequent e-words, how they will be pronounced.
    from ms_g2p import G2P
    g = G2P()
    rows = []
    with open(os.path.join(SRC, 'ms_50k.txt'), encoding='utf-8') as f:
        for rank, line in enumerate(f, 1):
            w = line.split(' ')[0].lower()
            if not WORD_OK.match(w) or 'e' not in w:
                continue
            spelled = g.respell(w)
            src = source.get(w, 'root' if g._decompose(w) else 'heuristic')
            alts = {n: d[w] for n, d in (('msa', msa), ('schwa', schwa), ('ind', ind)) if w in d}
            conflict = len(set(alts.values()) | ({spelled} if alts else set())) > 1
            rows.append((rank, w, spelled, src, 'CONFLICT' if conflict else '',
                         ' '.join(f'{k}={v}' for k, v in alts.items())))
    with open(os.path.join(SRC, 'review_top.tsv'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    top = rows[:1500]
    print('frequent e-words reviewed:', len(top),
          '| by source:', {s: sum(1 for r in top if r[3] == s) for s in set(r[3] for r in top)},
          '| conflicts:', sum(1 for r in top if r[4]))


if __name__ == '__main__':
    main()
