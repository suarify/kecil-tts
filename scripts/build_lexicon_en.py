"""Build g2p/lexicon_en.json: English words common in Malaysian code-switching -> Kokoro
(misaki US) phonemes. Words that are also Malay words are skipped, so Malay always wins.

Source: misaki us_gold.json / us_silver.json (Apache-2.0), plus a few brand names written by hand.
Usage: python scripts/build_lexicon_en.py path/to/us_gold.json path/to/us_silver.json
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'lex_src')

WORDS = """
meeting router restart group groups report reports member members renew laptop meetings booking book booked online offline check checking checked confirm confirmed confirmation
cancel cancelled delivery deliver order orders payment pay paid password okay sorry thanks thank you please
account update updated system phone call calling called internet email website link click login logout
register app apps download upload file files screenshot message messages voice customer customers service
support team manager boss office deadline schedule appointment slot available busy free urgent important
problem issue issues settle settled follow up reminder remind discount promo promotion offer deal price
cash card transfer insurance claim package basic total receipt invoice refund return exchange size colour
color style design quality brand product products stock item items shop shopping mall parking traffic
driver rider lunch dinner breakfast coffee weekend weekday monday tuesday wednesday thursday friday saturday
sunday birthday party holiday trip flight room staff admin director sales marketing budget report
presentation proposal contract agreement sign signature approve approval reject feedback review rating
survey form fill submit apply application status pending done finish start stop wait sure yes no nope yeah
hi hello bye welcome congrats congratulations best nice good great cool awesome so actually basically
already maybe really very just also sir madam bro sis guys everyone anyway the on in at for to of and i
we they it is my your new old next last first second one two three time today tomorrow yesterday tonight
week month year minute minutes hour hours late early fast slow easy hard happy sad love like want need
know think feel see go come back home work job meet talk chat reply send sent receive received share
open close closed full empty high low big small more less again later soon now still yet only even
anything something nothing everything someone anyone everybody nobody way true false right wrong left
same different simple fine safe risk plan plans goal target result results data? save saved delete
deleted edit change changed add added remove setting settings user users member members subscription
trial demo test testing launch event events training course class session lesson online-banking
banking charge charges fee fees deposit balance limit loan rate rates interest bonus reward rewards
voucher code points point level upgrade downgrade renew renewal expired expire expiry valid invalid
error bug crash slow loading network signal battery charger laptop computer printer screen camera
mic speaker headphone headphones mouse keyboard password username profile photo picture selfie post
story stories like likes follow follower followers comment comments live streaming stream content
creator influencer viral trending
""".split()

BRANDS = {
    'wifi': 'wˈIfˌI', 'whatsapp': 'wˈʌtsˌæp', 'facebook': 'fˈAsbˌʊk', 'instagram': 'ˈɪnstəɡɹˌæm',
    'tiktok': 'tˈɪktˌɑk', 'youtube': 'jˈutˌub', 'google': 'ɡˈuɡᵊl', 'zoom': 'zˈum', 'gmail': 'ʤˈimˌAl',
    'shopee': 'ʃˈɑpi', 'lazada': 'ləzˈɑdə', 'grab': 'ɡɹˈæb', 'telegram': 'tˈɛləɡɹˌæm',
}


def malay_words():
    out = set()
    with open(os.path.join(SRC, 'msa_latn_broad.tsv'), encoding='utf-8') as f:
        for line in f:
            out.add(line.split('\t')[0].lower())
    with open(os.path.join(SRC, 'schwa_dict.csv'), encoding='utf-8') as f:
        for row in csv.reader(f):
            if row:
                out.add(row[0].lower())
    with open(os.path.join(ROOT, 'g2p', 'lexicon_e.json'), encoding='utf-8') as f:
        out |= set(json.load(f))
    return out


def main(gold_path, silver_path):
    gold = json.load(open(gold_path, encoding='utf-8'))
    silver = json.load(open(silver_path, encoding='utf-8'))
    vocab = json.load(open(os.path.join(ROOT, 'g2p', 'vocab.json'), encoding='utf-8'))
    malay = malay_words()
    keep_even_if_malay = {'the', 'and', 'you', 'okay', 'sorry', 'please', 'thanks', 'meeting', 'booking',
                          'online', 'cancel', 'confirm', 'password', 'check', 'order', 'email', 'manager',
                          'problem', 'pending', 'open', 'edit', 'member', 'target', 'laptop', 'printer',
                          'transfer', 'register', 'item', 'we'}
    out, skipped_malay, missing = {}, [], []
    for w in WORDS:
        w = w.strip('?').lower()
        if '-' in w or not w:
            continue
        if w in malay and w not in keep_even_if_malay:
            skipped_malay.append(w)
            continue
        ph = gold.get(w, silver.get(w))
        if not ph and w.endswith('s') and w[:-1] in gold:
            base = gold[w[:-1]]
            base = base.get('DEFAULT') if isinstance(base, dict) else base
            ph = base + ('s' if base[-1] in 'ptkfθ' else 'z')
        elif not ph and w.endswith('ed') and w[:-1] in gold:
            base = gold[w[:-1]]
            base = base.get('DEFAULT') if isinstance(base, dict) else base
            ph = base + ('ᵻd' if base[-1] in 'td' else ('t' if base[-1] in 'pkfsʃʧθ' else 'd'))
        if isinstance(ph, dict):
            ph = ph.get('DEFAULT') or next(iter(ph.values()))
        if not ph:
            missing.append(w)
            continue
        if any(c not in vocab for c in ph):
            missing.append(w)
            continue
        out[w] = ph
    out.update(BRANDS)
    with open(os.path.join(ROOT, 'g2p', 'lexicon_en.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(out.items())), f, ensure_ascii=False, indent=0)
    print('lexicon_en entries:', len(out))
    print('skipped (also Malay):', ' '.join(skipped_malay))
    print('missing / out-of-vocab:', ' '.join(missing))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
