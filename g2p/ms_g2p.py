"""Malay grapheme-to-phoneme for Kokoro (kokoro-malay-bella).

Emits only symbols that exist in Kokoro-82M's 178-token vocab. Rules are in
spec.md; ms_g2p.js is a line-for-line port and must stay in sync (checked by
tests/golden.jsonl).

    from ms_g2p import G2P
    g2p = G2P()                       # default options = frozen spec
    phonemes = g2p("Selamat pagi, apa khabar?")
    ids = g2p.tokens(phonemes)        # [0, ..., 0] ready for the model
"""
import json
import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


VOCAB = _load('vocab.json')

# Letter names for ALL-CAPS acronyms (Malaysian English pronunciation).
LETTERS = {
    'a': 'ˈA', 'b': 'bˈi', 'c': 'sˈi', 'd': 'dˈi', 'e': 'ˈi', 'f': 'ˈɛf', 'g': 'ʤˈi',
    'h': 'ˈAʧ', 'i': 'ˈI', 'j': 'ʤˈA', 'k': 'kˈA', 'l': 'ˈɛl', 'm': 'ˈɛm', 'n': 'ˈɛn',
    'o': 'ˈO', 'p': 'pˈi', 'q': 'kjˈu', 'r': 'ˈɑr', 's': 'ˈɛs', 't': 'tˈi', 'u': 'jˈu',
    'v': 'vˈi', 'w': 'dˈʌbəljˌu', 'x': 'ˈɛks', 'y': 'wˈI', 'z': 'zˈɛd',
}

# Unstressed function words.
FUNCTION_WORDS = {
    'di', 'ke', 'dari', 'daripada', 'yang', 'dan', 'atau', 'untuk', 'dengan', 'pada',
    'kepada', 'akan', 'pun', 'lah', 'kah', 'tah', 'si', 'sang', 'para', 'oleh', 'bagi',
    'ia', 'nya', 'ini', 'itu', 'tu', 'ni', 'je', 'la', 'ke', 'kan', 'tak',
}

# Affixes used only to find the root for e-vowel lookup.
PREFIX_OUTER = [
    ('meng', ('', 'k')), ('meny', ('s',)), ('mem', ('', 'p')), ('men', ('', 't')), ('me', ('',)),
    ('peng', ('', 'k')), ('peny', ('s',)), ('pem', ('', 'p')), ('pen', ('', 't')),
    ('per', ('',)), ('pel', ('',)), ('pe', ('',)),
    ('ber', ('',)), ('bel', ('',)), ('be', ('',)), ('ter', ('',)), ('te', ('',)),
    ('di', ('',)), ('ke', ('',)), ('se', ('',)),
]
PREFIX_INNER = [('per', ''), ('ber', ''), ('ter', ''), ('ke', ''), ('se', ''), ('pe', '')]
SUF_DERIV = ['kan', 'an', 'i']
SUF_ENCL = ['nya', 'lah', 'kah', 'tah', 'ku', 'mu']
SUFFIX_STRINGS = set([''])
for _d in [''] + SUF_DERIV:
    for _e in [''] + SUF_ENCL:
        SUFFIX_STRINGS.add(_d + _e)

VOWEL_LETTERS = 'aeéiou'
NUCLEI = set('aeəiouAIOWY')
SINGLE = {
    'a': 'a', 'é': 'e', 'e': 'ə', 'i': 'i', 'o': 'o', 'u': 'u',
    'b': 'b', 'c': 'ʧ', 'd': 'd', 'f': 'f', 'g': 'ɡ', 'h': 'h', 'j': 'ʤ', 'k': 'k',
    'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'q': 'k', 's': 's', 't': 't', 'v': 'v',
    'w': 'w', 'y': 'j', 'z': 'z',
}
DIGRAPHS = {
    'ng': 'ŋ', 'ny': 'ɲ', 'sy': 'ʃ', 'sh': 'ʃ', 'kh': 'x', 'gh': 'ɣ', 'ch': 'ʧ',
    'th': 't', 'dh': 'd', 'ph': 'f', 'dz': 'z',
}
DIPHTHONGS = {'ai': ('aj', 'I'), 'au': ('aw', 'W'), 'oi': ('oj', 'Y'), 'éi': ('ej', 'A')}
PUNCT = ',.!?;:…—'
KEEP_FINAL_A = set('ya data drama kamera formula agenda media visa pizza opera asia indonesia astro '
                   'sofa kuota zebra panda lava'.split())

TOKEN_RE = re.compile(r"[A-Za-zé']+|[,.!?;:…—]")


def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', text)


class G2P:
    def __init__(self, stress=True, r='r', diph='glide', final_a='a'):
        self.final_a = final_a          # 'a' = spelling reading, 'ə' = Johor-Riau reading (saya -> sajə)
        self.stress = stress
        self.r = r                      # 'r' or 'ɾ'
        self.diph = 0 if diph == 'glide' else 1
        self.lex_e = _load('lexicon_e.json')
        self.lex_x = _load('lexicon_exceptions.json')
        self.lex_en = _load('lexicon_en.json')

    # ---------- e-vowel respelling ----------
    def _decompose(self, w):
        """Return the respelled word (é = taling, e = pepet) using lexicon roots, or None."""
        best = None
        best_free = None
        for encl in [''] + SUF_ENCL:
            if encl and not w.endswith(encl):
                continue
            w1 = w[:len(w) - len(encl)]
            for der in [''] + SUF_DERIV:
                if der and not w1.endswith(der):
                    continue
                w2 = w1[:len(w1) - len(der)]
                suffix = der + encl
                for prefix, root, drop in self._prefix_splits(w2):
                    if len(root) < 2:
                        continue
                    n_aff = len(prefix) + len(suffix)
                    if root in self.lex_e:
                        cand = (len(root), -n_aff, prefix, root, drop, suffix)
                        if best is None or cand[:2] > best[:2]:
                            best = cand
                    elif 'e' not in root and prefix and sum(c in VOWEL_LETTERS for c in root) >= 2:
                        cand = (len(root), -n_aff, prefix, root, drop, suffix)
                        if best_free is None or cand[:2] > best_free[:2]:
                            best_free = cand
        pick = best or best_free
        if pick is None:
            return None
        _, _, prefix, root, drop, suffix = pick
        spelled = self.lex_e.get(root, root)
        return prefix + spelled[drop:] + suffix

    @staticmethod
    def _prefix_splits(w):
        yield '', w, 0
        for p, restores in PREFIX_OUTER:
            if not w.startswith(p) or len(w) <= len(p):
                continue
            rest = w[len(p):]
            for rs in restores:
                yield p, rs + rest, len(rs)
                if rs == '':
                    for q, _ in PREFIX_INNER:
                        if rest.startswith(q) and len(rest) > len(q):
                            yield p + q, rest[len(q):], 0

    @staticmethod
    def _heuristic(w):
        """Unknown word: final-syllable e is taling unless closed by m, n or r; other e are pepet."""
        idx = [i for i, c in enumerate(w) if c in VOWEL_LETTERS]
        if not idx:
            return w
        last = idx[-1]
        chars = list(w)
        if w[last] == 'e':
            coda = w[last + 1:]
            if coda == '' or (coda[-1] not in 'mnr' and coda != 'ng'):
                chars[last] = 'é'
        return ''.join(chars)

    def respell(self, w):
        if 'e' not in w:
            return w
        if w in self.lex_e:
            return self.lex_e[w]
        d = self._decompose(w)
        if d is not None:
            return d
        return self._heuristic(w)

    # ---------- letters to phonemes ----------
    def _letters(self, s):
        out = []
        n = len(s)
        i = 0
        while i < n:
            c = s[i]
            two = s[i:i + 2]
            if c in 'aoé' and two in DIPHTHONGS and s[i + 2:] in SUFFIX_STRINGS:
                ph = DIPHTHONGS[two][self.diph]
                self._push_vowel(out, ph[0])
                out.extend(ph[1:])
                i += 2
                continue
            if two in DIGRAPHS:
                out.append(DIGRAPHS[two])
                i += 2
                continue
            if c == "'":
                if out and i + 1 < n:
                    out.append('ʔ')
                i += 1
                continue
            if c == 'k':
                prev_vowel = i > 0 and s[i - 1] in VOWEL_LETTERS
                nxt = s[i + 1] if i + 1 < n else ''
                if prev_vowel and (nxt == '' or (nxt not in VOWEL_LETTERS and nxt not in "srl'")):
                    out.append('ʔ')
                else:
                    out.append('k')
                i += 1
                continue
            if c == 'x':
                out.extend(['z'] if i == 0 else ['k', 's'])
                i += 1
                continue
            if c == 'r':
                out.append(self.r)
                i += 1
                continue
            ph = SINGLE.get(c)
            if ph is None:
                i += 1
                continue
            if c in VOWEL_LETTERS:
                self._push_vowel(out, ph)
            else:
                out.append(ph)
            i += 1
        return out

    @staticmethod
    def _push_vowel(out, v):
        # Hiatus: after a pepet, or two identical vowels -> glottal stop between.
        if out and out[-1] in NUCLEI and (out[-1] == 'ə' or out[-1] == v):
            out.append('ʔ')
        out.append(v)

    def _stress(self, ph, w):
        if not self.stress or w in FUNCTION_WORDS:
            return ph
        nuc = [i for i, p in enumerate(ph) if p in NUCLEI]
        if not nuc:
            return ph
        k = nuc[-2] if len(nuc) >= 2 else nuc[-1]
        if len(nuc) >= 2 and ph[k] == 'ə':
            k = nuc[-1]
        return ph[:k] + ['ˈ'] + ph[k:]

    def word(self, tok, first=True):
        if len(tok) >= 2 and tok.isupper() and tok.isalpha() and tok.lower() not in self.lex_x:
            return ' '.join(LETTERS[c] for c in tok.lower() if c in LETTERS)
        w = tok.lower().strip("'")
        if not w:
            return ''
        if w in self.lex_x:
            return self.lex_x[w]
        if w in self.lex_en:
            return self.lex_en[w]
        ph = self._letters(self.respell(w))
        if (self.final_a == 'ə' and ph and ph[-1] == 'a' and sum(p in NUCLEI for p in ph) >= 2
                and w not in KEEP_FINAL_A and not (tok[0].isupper() and not first)):
            ph[-1] = 'ə'
        return ''.join(self._stress(ph, w))

    # ---------- sentence ----------
    def __call__(self, text):
        text = strip_accents(text).replace('-', ' ')
        out = ''
        first = True
        for tok in TOKEN_RE.findall(text):
            if tok in PUNCT:
                out = out.rstrip(' ') + tok + ' '
                if tok in '.!?…':
                    first = True
                continue
            ph = self.word(tok, first)
            first = False
            if ph:
                out += ph + ' '
        return out.strip()

    @staticmethod
    def tokens(phonemes):
        return [0] + [VOCAB[c] for c in phonemes if c in VOCAB] + [0]


def chunk(phonemes, limit=510):
    """Split a phoneme string into pieces of at most `limit` symbols, at the best boundary."""
    out = []
    rest = phonemes.strip()
    while len(rest) > limit:
        cut = -1
        for marks in ('.!?…', ';:—', ',', ' '):
            for j in range(limit - 1, limit // 3, -1):
                if rest[j] in marks:
                    cut = j + 1
                    break
            if cut > 0:
                break
        if cut <= 0:
            cut = limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    g = G2P()
    for line in sys.stdin:
        print(g(line.rstrip('\n')))
