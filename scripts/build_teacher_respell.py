"""Build g2p/teacher_respell.json: English word -> Malay-orthography spelling.

Used ONLY for the text sent to the Chatterbox teacher. Chatterbox reads Malay text with
Malay letter rules, so 'confirm' comes out 'chon-firm' and 'celsius' 'chel-sius'
(Malay c = /tʃ/). Respelled as Malaysians say them ('konfem', 'selsius') it reads them
right. Kokoro's own transcript keeps the original word; its G2P handles English itself.

Automatic: misaki phonemes from lexicon_en.json -> Malay letters. Hand overrides win.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Malaysian everyday spellings (as used in casual Malay writing / speech).
OVERRIDES = {
    'confirm': 'konfem', 'confirmed': 'konfem', 'celsius': 'selsius', 'cancel': 'kensel',
    'cancelled': 'kensel', 'check': 'cek', 'booking': 'buking', 'book': 'buk',
    'meeting': 'miting', 'update': 'apdet', 'password': 'paswed', 'app': 'ep', 'apps': 'eps',
    'send': 'sen', 'link': 'link', 'whatsapp': 'wasap', 'problem': 'problem',
    'call': 'kol', 'customer': 'kastemer', 'service': 'servis', 'payment': 'pemen',
    'online': 'onlain', 'offline': 'oflain', 'email': 'imel', 'wifi': 'waifai',
    'delivery': 'deliveri', 'order': 'oder', 'okay': 'okei', 'ok': 'okei', 'sorry': 'sori',
    'please': 'plis', 'thank': 'tenk', 'thanks': 'tenks', 'phone': 'fon', 'computer': 'komputer',
    'internet': 'internet', 'website': 'websait', 'login': 'login', 'account': 'akaun',
    'feedback': 'fidbek', 'schedule': 'syedul', 'free': 'fri', 'sale': 'sel', 'deal': 'dil',
    'cash': 'kesy', 'card': 'kad', 'credit': 'kredit', 'debit': 'debit', 'transfer': 'transfer',
    'code': 'kod', 'video': 'video', 'chat': 'cet', 'share': 'syer', 'save': 'sev',
    'download': 'daunlod', 'upload': 'aplod', 'install': 'instol', 'manager': 'menejer',
    'staff': 'staf', 'boss': 'bos', 'office': 'ofis', 'claim': 'klem', 'refund': 'rifan',
    'discount': 'diskaun', 'voucher': 'vaucer', 'promo': 'promo', 'register': 'rejister',
    'appointment': 'apointmen', 'available': 'avelabel', 'confirmation': 'konfemesyen',
    'reminder': 'rimainder', 'slot': 'slot', 'status': 'status', 'member': 'member', 'report': 'ripot',
    'group': 'grup', 'router': 'rauter', 'restart': 'ristat', 'laptop': 'leptop', 'renew': 'rinyu', 'actually': 'ekcueli', 'sales': 'sels',
    'expired': 'ekspaiad', 'think': 'tink', 'of': 'of', 'for': 'for', 'colour': 'kaler', 'follow': 'folo', 'support': 'sapot', 'team': 'tim',
}

PH = [  # longest first
    ('ʤ', 'j'), ('ʧ', 'c'), ('ʃ', 'sy'), ('ʒ', 'sy'), ('θ', 't'), ('ð', 'd'), ('ŋ', 'ng'),
    ('ɹ', 'r'), ('ɾ', 't'), ('ɡ', 'g'), ('j', 'y'),
    ('A', 'ei'), ('I', 'ai'), ('W', 'au'), ('Y', 'oi'), ('O', 'o'), ('Q', 'o'),
    ('æ', 'e'), ('ɑ', 'a'), ('ʌ', 'a'), ('ə', 'e'), ('ᵊ', 'e'), ('ɜ', 'e'), ('ɛ', 'e'),
    ('ɪ', 'i'), ('ʊ', 'u'), ('ɔ', 'o'), ('ɒ', 'o'), ('ɐ', 'a'),
]
DROP = set('ˈˌː ')


def auto(ph):
    out = []
    for ch in ph:
        if ch in DROP:
            continue
        out.append(dict(PH).get(ch, ch))
    s = ''.join(out)
    # 'er' at word end: Malaysians say -er as -e(r); keep 'er'
    return s if s.isascii() and s.isalpha() else None


def main():
    en = json.load(open(os.path.join(ROOT, 'g2p', 'lexicon_en.json'), encoding='utf-8'))
    out = {}
    for w, ph in en.items():
        r = auto(ph)
        if r:
            out[w] = r
    out.update(OVERRIDES)
    path = os.path.join(ROOT, 'g2p', 'teacher_respell.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(out.items())), f, ensure_ascii=False, indent=0)
    print('teacher_respell.json', len(out), 'entries; skipped', len(en) - len(set(en) & set(out)))


def respell(text, table):
    """Replace English words (case-insensitive, whole word) for the teacher only."""
    import re

    def rep(m):
        w = m.group(0)
        r = table.get(w.lower())
        if r is None:
            return w
        return r.capitalize() if w[0].isupper() else r
    return re.sub(r'[A-Za-z]+', rep, text)


if __name__ == '__main__':
    main()
