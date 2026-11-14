"""Gate 1 checks for the Malay front end. Run: python -m pytest g2p/tests -q"""
import bz2
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
from ms_norm import normalize  # noqa: E402
from ms_g2p import G2P, VOCAB, chunk  # noqa: E402

G = G2P()
# Symbols that are in Kokoro's vocab but must never come out of the Malay path:
# ASCII g (silently dropped by Kokoro), and c/q/y which are English-only conventions.
# ('x' is allowed: it is the velar fricative for 'kh'.)
FORBIDDEN_MALAY = set('gcqy')


def tatoeba(n=None):
    lines = bz2.open(os.path.join(ROOT, 'data', 'text', 'src', 'zsm_sentences.tsv.bz2'), 'rt', encoding='utf-8').read().splitlines()
    out = [l.split('\t')[2] for l in lines if l.count('\t') >= 2]
    return out[:n] if n else out


def test_every_symbol_in_vocab_and_no_forbidden():
    bad, forb = set(), set()
    for t in tatoeba():
        n = normalize(t)
        for tok in n.replace('-', ' ').split():
            w = tok.strip(',.!?;:…—"()\'').lower()
            ph = G(tok)
            bad |= {c for c in ph if c not in VOCAB}
            if w and w not in G.lex_en and w not in G.lex_x and not tok.strip(',.!?;:…—"()').isupper():
                forb |= {c for c in ph if c in FORBIDDEN_MALAY}
    assert not bad, f'out-of-vocab symbols: {bad}'
    assert not forb, f'forbidden symbols from Malay rules: {forb}'


# Hand-verified e-vowel answers (Malaysian standard). 'é' = taling, 'e' = pepet.
E_GOLD = """
besar meja kereta emak enam empat teman merah boleh pendek leher sewa beza elok esok ekor sate
selesai mereka telefon kedai negeri negara bersedia perempuan sekolah dewan lewat tengok nenek
cerita kecil berapa sebelum selepas semua sekarang pergi kerja betul terima kenapa tempat dengar
keluarga perlu selamat seorang menyelesaikan memerlukan pejabat pekerja bekerja hebat memang
bebas tembak mesin serius idea video hotel projek beg teh restoran teknologi elektrik kerana
""".split()
E_EXPECT = {
    'besar': 'besar', 'meja': 'méja', 'kereta': 'keréta', 'emak': 'emak', 'enam': 'enam', 'empat': 'empat',
    'teman': 'teman', 'merah': 'mérah', 'boleh': 'boléh', 'pendek': 'pendék', 'leher': 'léhér', 'sewa': 'séwa',
    'beza': 'béza', 'elok': 'élok', 'esok': 'ésok', 'ekor': 'ékor', 'sate': 'saté', 'selesai': 'selésai',
    'mereka': 'meréka', 'telefon': 'téléfon', 'kedai': 'kedai', 'negeri': 'negeri', 'negara': 'negara',
    'bersedia': 'bersedia', 'perempuan': 'perempuan', 'sekolah': 'sekolah', 'dewan': 'déwan', 'lewat': 'léwat',
    'tengok': 'téngok', 'nenek': 'nénék', 'cerita': 'cerita', 'kecil': 'kecil', 'berapa': 'berapa',
    'sebelum': 'sebelum', 'selepas': 'selepas', 'semua': 'semua', 'sekarang': 'sekarang', 'pergi': 'pergi',
    'kerja': 'kerja', 'betul': 'betul', 'terima': 'terima', 'kenapa': 'kenapa', 'tempat': 'tempat',
    'dengar': 'dengar', 'keluarga': 'keluarga', 'perlu': 'perlu', 'selamat': 'selamat', 'seorang': 'seorang',
    'menyelesaikan': 'menyelésaikan', 'memerlukan': 'memerlukan', 'pejabat': 'pejabat', 'pekerja': 'pekerja',
    'bekerja': 'bekerja', 'hebat': 'hébat', 'memang': 'mémang', 'bebas': 'bébas', 'tembak': 'témbak',
    'mesin': 'mésin', 'serius': 'sérius', 'idea': 'idéa', 'video': 'vidéo', 'hotel': 'hotél',
    'projek': 'projék', 'beg': 'bég', 'teh': 'téh', 'restoran': 'réstoran', 'teknologi': 'téknologi',
    'elektrik': 'éléktrik', 'kerana': 'kerana',
}


def test_e_vowels():
    wrong = [(w, G.respell(w), E_EXPECT[w]) for w in E_GOLD if G.respell(w) != E_EXPECT[w]]
    acc = 1 - len(wrong) / len(E_GOLD)
    assert acc >= 0.97, f'e accuracy {acc:.3f}; wrong: {wrong}'


NORM_CASES = [
    ('Harga RM12.50 sahaja.', 'Harga dua belas ringgit lima puluh sen sahaja.'),
    ('RM0.50', 'lima puluh sen'),
    ('RM5', 'lima ringgit'),
    ('RM1.5 juta', 'satu perpuluhan lima juta ringgit'),
    ('RM1,250,000', 'satu juta dua ratus lima puluh ribu ringgit'),
    ('25%', 'dua puluh lima peratus'),
    ('ke-3', 'ketiga'), ('ke-1', 'pertama'), ('ke-21', 'kedua puluh satu'),
    ('13/11/2026', 'tiga belas November dua ribu dua puluh enam'),
    ('5 Mac 2025', 'lima Mac dua ribu dua puluh lima'),
    ('3:30 petang', 'pukul tiga tiga puluh petang'),
    ('pukul 9.15 pagi', 'pukul sembilan lima belas pagi'),
    ('2pm', 'pukul dua petang'), ('12pm', 'pukul dua belas tengah hari'), ('8pm', 'pukul lapan malam'),
    ('012-345 6789', 'kosong satu dua tiga empat lima enam tujuh lapan sembilan'),
    ('kod OTP 482913', 'kod OTP empat lapan dua sembilan satu tiga'),
    ('Baki akaun anda ialah RM87.20', 'Baki akaun anda ialah lapan puluh tujuh ringgit dua puluh sen'),
    ('15km', 'lima belas kilometer'), ('2.5kg', 'dua perpuluhan lima kilogram'),
    ('30°C', 'tiga puluh darjah Celsius'),
    ('Dr. Ahmad', 'doktor Ahmad'), ('ABC Sdn. Bhd.', 'ABC sendirian berhad.'),
    ('1998', 'seribu sembilan ratus sembilan puluh lapan'), ('2026', 'dua ribu dua puluh enam'),
    ('11', 'sebelas'), ('10', 'sepuluh'), ('100', 'seratus'), ('1000', 'seribu'),
    ('21000', 'dua puluh satu ribu'), ('100000', 'seratus ribu'), ('1000000', 'satu juta'),
    ('3.14', 'tiga perpuluhan satu empat'), ('10-20 tahun', 'sepuluh hingga dua puluh tahun'),
    ('yg ni x boleh, tq!', 'yang ni tak boleh, terima kasih!'),
    ('Ali & Abu', 'Ali dan Abu'),
]


def test_normaliser():
    bad = [(i, normalize(i), o) for i, o in NORM_CASES if normalize(i) != o]
    assert not bad, bad


def test_chunking_limit():
    longest = 0
    texts = tatoeba()
    blob = ' '.join(texts[:400])
    for ph in [G(normalize(t)) for t in texts[:2000]] + [G(normalize(blob))]:
        for c in chunk(ph):
            longest = max(longest, len(c))
            assert len(c) <= 510
    assert longest > 0


def test_golden_matches_python():
    rows = [json.loads(l) for l in open(os.path.join(HERE, 'golden.jsonl'), encoding='utf-8')]
    cache = {}
    bad = 0
    for r in rows:
        key = json.dumps(r['opts'], sort_keys=True)
        if key not in cache:
            cache[key] = G2P(**r['opts'])
        n = normalize(r['text'])
        if n != r['norm'] or cache[key](n) != r['ph']:
            bad += 1
    assert bad == 0, f'{bad} golden rows changed; regenerate with scripts/make_golden.py and re-run parity.mjs'
