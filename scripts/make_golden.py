"""Write g2p/tests/golden.jsonl: (text, options, normalised, phonemes) from the Python reference.

The JS port must reproduce every row exactly (g2p/tests/parity.mjs).
Rows: Tatoeba Malay sentences + normaliser edge cases + 200 rows under non-default options.
"""
import bz2
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_norm import normalize  # noqa: E402
from ms_g2p import G2P  # noqa: E402

EDGE = [
    'Harga RM12.50 sahaja.', 'Jumlah RM1,250,000 atau RM1.5 juta.', 'Diskaun 25% untuk ahli ke-3.',
    'Temujanji pada 13/11/2026 pukul 3.30 petang.', 'Mesyuarat 10:00 am, 2pm dan 8.15 malam.',
    'Hubungi 012-345 6789 atau +6012 345 6789.', 'No. IC 900101-14-5678, kod OTP 482913.',
    'Jarak 15km, berat 2.5kg, suhu 30°C.', 'Dr. Ahmad dan En. Ali dari ABC Sdn. Bhd. dll.',
    'Tahun 1998 dan 2026, 100 orang, 21000 ringgit.', 'Tarikh 5 Mac 2025 dan 2 Disember.',
    'Umur 10-20 tahun, harga 0.75 sen, 1000000 unit, 3.14159.', 'yg ni x boleh, tq!',
    'Kereta api akan tiba pukul 3.30 petang di Stesen Sentral.', 'Bayaran RM12.50 melalui akaun KWSP.',
    'Maaf, boleh saya check booking awak untuk meeting esok?', "Qur'an, ta'at dan Jum'at.",
    'Kanak-kanak bermain di taman; berlari-lari — sangat gembira…', 'OK, LHDN dan SST naik 8%.',
    'Café résumé naïve — “petikan” ‘tunggal’.', 'Pukul 12pm dan 9pm, jam 7.45 pagi.',
    '$5 atau USD 1.2 bilion.', 'Nombor akaun 1234 5678 9012.', 'Tingkat ke-21, unit 0.5ml.',
    'Saiz 1,000,000,000 dan 2,500,000,000,000.', 'Hari ke-1 dan ke-100.',
]


def main():
    lines = bz2.open(os.path.join(ROOT, 'data', 'text', 'src', 'zsm_sentences.tsv.bz2'), 'rt', encoding='utf-8').read().splitlines()
    sents = [l.split('\t')[2] for l in lines if l.count('\t') >= 2]
    rnd = random.Random(1234)
    rnd.shuffle(sents)
    texts = EDGE + sents[:1800 - len(EDGE)]
    variants = [dict(stress=False), dict(r='ɾ'), dict(diph='kokoro'), dict(stress=False, r='ɾ', diph='kokoro')]
    g_default = G2P()
    g_var = [G2P(**v) for v in variants]
    out = []
    for t in texts:
        n = normalize(t)
        out.append({'text': t, 'opts': {}, 'norm': n, 'ph': g_default(n)})
    for k, t in enumerate(sents[1800:2000]):
        v = k % len(variants)
        n = normalize(t)
        out.append({'text': t, 'opts': variants[v], 'norm': n, 'ph': g_var[v](n)})
    path = os.path.join(ROOT, 'g2p', 'tests', 'golden.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('wrote', len(out), 'rows to', path)


if __name__ == '__main__':
    main()
