"""Malay text normaliser for kokoro-malay-bella.

Turns written Malay (digits, currency, dates, abbreviations) into the words a
speaker would say. Ordinary words keep their casing so the G2P can still see
ALL-CAPS acronyms.

ms_norm.js is a line-for-line port. Every regex runs in ASCII mode so \\d, \\w
and \\b mean the same thing in both languages. Change both files together and
regenerate tests/golden.jsonl.
"""
import re
import unicodedata

ONES = ['kosong', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan', 'sembilan']
MONTHS = ['Januari', 'Februari', 'Mac', 'April', 'Mei', 'Jun', 'Julai', 'Ogos',
          'September', 'Oktober', 'November', 'Disember']
MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mac': 3, 'mar': 3, 'apr': 4, 'mei': 5, 'may': 5, 'jun': 6,
    'jul': 7, 'ogo': 8, 'aug': 8, 'sep': 9, 'okt': 10, 'oct': 10, 'nov': 11, 'dis': 12, 'dec': 12,
}
MONTH_RE = ('januari|january|jan|februari|february|feb|mac|march|mar|april|apr|mei|may|'
            'jun|june|julai|july|jul|ogos|august|aug|september|sept|sep|oktober|october|okt|oct|'
            'november|nov|disember|december|dis|dec')

ABBR = [
    ('sdn. bhd.', 'sendirian berhad'), ('sdn bhd', 'sendirian berhad'),
    ('bhd.', 'berhad'), ('sdn.', 'sendirian'),
    ('dr.', 'doktor'), ('en.', 'encik'), ('pn.', 'puan'), ('cik.', 'cik'),
    ('tn.', 'tuan'), ('hj.', 'haji'), ('hjh.', 'hajah'), ('prof.', 'profesor'),
    ('jln.', 'jalan'), ('jln', 'jalan'), ('tmn.', 'taman'), ('tmn', 'taman'),
    ('kg.', 'kampung'), ('lrg.', 'lorong'),
    ('dll.', 'dan lain-lain'), ('dll', 'dan lain-lain'), ('dsb.', 'dan sebagainya'),
    ('spt.', 'seperti'), ('spt', 'seperti'), ('yg', 'yang'), ('dgn', 'dengan'),
    ('utk', 'untuk'), ('kpd', 'kepada'), ('drp', 'daripada'), ('sbb', 'sebab'),
    ('tlg', 'tolong'), ('tq', 'terima kasih'), ('tkasih', 'terima kasih'),
    ('lg', 'lagi'), ('jgn', 'jangan'), ('sy', 'saya'), ('x', 'tak'),
    ('no.', 'nombor'), ('tel.', 'telefon'), ('ptg', 'petang'), ('pg', 'pagi'),
]

UNITS = [
    ('km/j', 'kilometer sejam'), ('km/h', 'kilometer sejam'),
    ('°c', 'darjah Celsius'), ('km', 'kilometer'), ('kg', 'kilogram'),
    ('cm', 'sentimeter'), ('mm', 'milimeter'), ('ml', 'mililiter'),
    ('mg', 'miligram'), ('g', 'gram'), ('m', 'meter'), ('l', 'liter'),
]

DIGIT_CONTEXT = r'(?:kod|otp|pin|nombor|no\.?|akaun|telefon|tel\.?|ic|tac)'
NUM = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?'
TIME_SFX = r'(am|pm|a\.m\.|p\.m\.|pagi|petang|tengah hari|malam)'
DIGITS = '0123456789'


def sub(pattern, repl, text, ci=False):
    return re.sub(pattern, repl, text, flags=re.A | (re.I if ci else 0))


def g(m, i):
    """Group i or '' (JS returns undefined for unmatched groups)."""
    return m.group(i) or ''


def cardinal(n):
    if n < 10:
        return ONES[n]
    if n == 10:
        return 'sepuluh'
    if n == 11:
        return 'sebelas'
    if n < 20:
        return ONES[n - 10] + ' belas'
    if n < 100:
        t, r = n // 10, n % 10
        return ONES[t] + ' puluh' + ('' if r == 0 else ' ' + ONES[r])
    for size, word, single in ((10 ** 12, 'trilion', 'satu trilion'),
                               (10 ** 9, 'bilion', 'satu bilion'),
                               (10 ** 6, 'juta', 'satu juta'),
                               (1000, 'ribu', 'seribu'),
                               (100, 'ratus', 'seratus')):
        if n >= size:
            q, r = n // size, n % size
            head = single if q == 1 else cardinal(q) + ' ' + word
            return head + ('' if r == 0 else ' ' + cardinal(r))
    raise ValueError(n)


def digits(s):
    out = []
    for ch in s:
        if ch in DIGITS:
            out.append(ONES[int(ch)])
        elif ch == '+':
            out.append('tambah')
    return ' '.join(out)


def number(s):
    s = s.replace(',', '')
    if '.' in s:
        whole, frac = s.split('.', 1)
        return cardinal(int(whole or '0')) + ' perpuluhan ' + digits(frac)
    return cardinal(int(s))


def ordinal(n):
    return 'pertama' if n == 1 else 'ke' + cardinal(n)


def time_words(h, m, suffix):
    h, m = int(h), int(m)
    part = ''
    if suffix:
        sfx = suffix.lower().replace('.', '')
        if sfx == 'am':
            part = 'pagi'
        elif sfx == 'pm':
            part = 'tengah hari' if h == 12 else ('petang' if h < 7 else 'malam')
        else:
            part = suffix
    words = cardinal(h) + ('' if m == 0 else ' ' + cardinal(m))
    return words + ((' ' + part) if part else '')


def strip_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', text)


def money(m):
    amount, scale = g(m, 2), g(m, 3)
    unit = 'ringgit' if g(m, 1).lower() == 'rm' else 'dolar'
    if scale:
        return number(amount) + ' ' + scale.lower() + ' ' + unit
    a = amount.replace(',', '')
    dot = a.find('.')
    whole = a if dot < 0 else a[:dot]
    frac = '' if dot < 0 else a[dot + 1:]
    whole_n = int(whole or '0')
    sen = int((frac + '00')[:2]) if frac else 0
    parts = []
    if whole_n or not sen:
        parts.append(cardinal(whole_n) + ' ' + unit)
    if sen:
        parts.append(cardinal(sen) + ' sen')
    return ' '.join(parts)


def date_num(m):
    d, mo, y = int(g(m, 1)), int(g(m, 2)), g(m, 3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    yy = int(y) if len(y) == 4 else 2000 + int(y)
    return cardinal(d) + ' ' + MONTHS[mo - 1] + ' ' + cardinal(yy)


def date_word(m):
    mo = MONTH_ABBR[g(m, 2).lower()[:3]]
    out = cardinal(int(g(m, 1))) + ' ' + MONTHS[mo - 1]
    if g(m, 3):
        out += ' ' + cardinal(int(g(m, 3)))
    return out


def time_m(m):
    lead = g(m, 1)
    return (lead if lead else 'pukul ') + time_words(g(m, 2), g(m, 3), g(m, 4))


def num_m(m):
    s = m.group(0)
    plain = s.replace(',', '')
    is_int = all(c in DIGITS for c in plain)
    if (len(plain) > 1 and plain[0] == '0' and '.' not in plain) or (is_int and len(plain) >= 8):
        return digits(plain)
    return number(s)


def esc(s):
    return re.sub(r'([.*+?^${}()|\[\]\\/])', r'\\\1', s)


def normalize(text):
    t = strip_accents(text.replace('°', '\u0001'))
    t = t.replace('\u0001', '°')
    t = t.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    t = t.replace('–', '-').replace('—', ' — ')
    t = re.sub(r'[\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff\t\r\n\f\v]', ' ', t)

    # 1. Phone, IC and digit-context numbers: digit by digit.
    t = sub(r'\+?\b6?0\d{1,2}[- ]?\d{3,4}[- ]?\d{3,4}\b', lambda m: ' ' + digits(m.group(0)) + ' ', t)
    t = sub(r'\b\d{6}-\d{2}-\d{4}\b', lambda m: ' ' + digits(m.group(0)) + ' ', t)
    t = sub(r'\b(' + DIGIT_CONTEXT + r')(\s*:?\s*)(\d[\d\- ]*\d|\d)\b',
            lambda m: g(m, 1) + g(m, 2) + digits(g(m, 3)), t, ci=True)

    # 2. Currency.
    t = sub(r'\b(RM|USD|US\$)\s?(' + NUM + r')(?:\s?(juta|bilion|ribu|trilion)\b)?', money, t, ci=True)
    t = sub(r'(\$)\s?(' + NUM + r')(?:\s?(juta|bilion|ribu|trilion)\b)?', money, t, ci=True)

    # 3. Percent.
    t = sub(r'(' + NUM + r')\s?%', lambda m: number(g(m, 1)) + ' peratus', t)

    # 4. Dates dd/mm/yyyy, dd-mm-yyyy, and 13 Nov 2026.
    t = sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b', date_num, t)
    t = sub(r'\b(\d{1,2})\s+(' + MONTH_RE + r')\.?(?:\s+(\d{4}))?\b', date_word, t, ci=True)

    # 5. Times.
    t = sub(r'(\b(?:pukul|jam)\s+)?\b(\d{1,2}):(\d{2})\s?' + TIME_SFX + r'?', time_m, t, ci=True)
    t = sub(r'(\b(?:pukul|jam)\s+)\b(\d{1,2})\.(\d{2})\s?' + TIME_SFX + r'?', time_m, t, ci=True)
    t = sub(r'()\b(\d{1,2})\.(\d{2})\s?' + TIME_SFX, time_m, t, ci=True)
    t = sub(r'(\b(?:pukul|jam)\s+)?\b(\d{1,2})\s?(am|pm|a\.m\.|p\.m\.)(?=\W|$)',
            lambda m: (g(m, 1) or 'pukul ') + time_words(g(m, 2), 0, g(m, 3)), t, ci=True)

    # 6. Ordinals and ranges.
    t = sub(r'\bke-?(\d+)\b', lambda m: ordinal(int(g(m, 1))), t, ci=True)
    t = sub(r'\b(' + NUM + r')\s?-\s?(' + NUM + r')\b',
            lambda m: number(g(m, 1)) + ' hingga ' + number(g(m, 2)), t)

    # 7. Units after a number.
    for u, w in UNITS:
        t = sub(r'(\d)\s?' + esc(u) + r'(?![a-zA-Z])', lambda m, w=w: g(m, 1) + ' ' + w, t, ci=True)

    # 8. Remaining numbers.
    t = sub(NUM, num_m, t)

    # 9. Abbreviations, longest first.
    for a, w in sorted(ABBR, key=lambda x: -len(x[0])):
        pat = r'(?<![\w.])' + esc(a) + (r'(?!\w)' if a.endswith('.') else r'(?![\w.])')
        t = sub(pat, w, t, ci=True)

    # 10. Symbols.
    t = t.replace('&', ' dan ').replace('+', ' tambah ').replace('=', ' sama dengan ').replace('%', ' peratus ')

    # 11. Spacing.
    t = sub(r'\s+', ' ', t).strip()
    t = sub(r'\s+([,.!?;:…])', r'\1', t)
    if text.rstrip().endswith('.') and t and t[-1] not in '.!?…':
        t += '.'
    return t


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        print(normalize(line.rstrip('\n')))
