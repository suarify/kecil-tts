"""Generate 600 code-switch sentences (Malay frames + English loanwords) for Phase 4."""
import csv
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
rng = random.Random(404)

FRAMES = [
    "Booking {name} untuk hari {day} dah confirmed.",
    "Sila {verb} {noun} sebelum hari {day}.",
    "Payment untuk {noun} boleh dibuat secara {method}.",
    "Saya tak dapat {verb}, {noun} nampak salah.",
    "Jangan lupa {verb} {noun} selepas {event}.",
    "{name} dah {verb} {noun} tadi pagi.",
    "Status {noun} untuk {name} dah di-{verb}.",
    "Tolong {verb} {noun} dalam {app} sebelum {event}.",
    "Customer service kami boleh dihubungi melalui {app}.",
    "Kalau {noun} masih slow, cuba {verb} dulu.",
    "Saya akan {verb} {noun} selepas {event}.",
    "Deadline untuk {noun} ialah hari {day}.",
    "{name} minta {verb} {noun} hari ini.",
    "Aduan {noun} dah kami {verb}.",
    "Sila update {noun} dalam {app} sekarang.",
    "Meeting dengan {name} bermula pukul {time}.",
    "Laporan {noun} untuk {event} dah siap.",
    "{name} tak hadir untuk {event} hari ini.",
    "Sila confirm {noun} dengan {name} sebelum {event}.",
    "Delivery {noun} ke {place} dijadualkan hari {day}.",
]
NAMES = ['encik', 'puan', 'dato', 'cik', 'tuan', 'doktor', 'abang', 'kak']
DAYS = ['Isnin', 'Selasa', 'Rabu', 'Khamis', 'Jumaat', 'Sabtu', 'Ahad']
VERBS = ['confirm', 'check', 'update', 'send', 'reply', 'submit', 'download', 'save', 'cancel', 'approve', 'restart', 'share']
NOUNS = ['booking', 'invoice', 'resit', 'report', 'password', 'email', 'form', 'claim', 'delivery', 'message', 'profile', 'account', 'order', 'file', 'link', 'payment', 'schedule', 'appointment']
METHODS = ['online', 'cash', 'transfer', 'card']
APPS = ['WhatsApp', 'email', 'system', 'app']
EVENTS = ['meeting', 'delivery', 'payment', 'submission', 'registration', 'service', 'booking']
TIMES = ['empat', 'lima', 'enam', 'tujuh', 'lapan', 'sembilan']
PLACES = ['Shah Alam', 'Ipoh', 'Kota Bharu', 'Johor Bahru', 'Penang', 'Putrajaya']

out, seen = [], set()
while len(out) < 600:
    t = rng.choice(FRAMES)
    s = t.format(name=rng.choice(NAMES), day=rng.choice(DAYS), verb=rng.choice(VERBS),
                 noun=rng.choice(NOUNS), method=rng.choice(METHODS), app=rng.choice(APPS),
                 event=rng.choice(EVENTS), time=rng.choice(TIMES), place=rng.choice(PLACES))
    if s not in seen:
        seen.add(s)
        out.append(s)

p = 'eval/p4_codeswitch_synth.tsv'
with open(p, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['id', 'category', 'text', 'teacher', 'source'])
    for i, s in enumerate(out):
        w.writerow([f'p4x{i:05d}', 'codeswitch', s, s, 'synth_codeswitch'])
print('wrote', p, len(out))
