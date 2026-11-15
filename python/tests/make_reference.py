"""Write ids + a coarse loudness envelope from the Python runtime so the Node test can compare. usage: make_reference.py OUT.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from suaradee import Suaradee  # noqa: E402

tts = Suaradee(os.environ['SUARADEE_MODEL'])
SENT = ['Selamat pagi, boleh saya bantu encik hari ini?', 'Jumlah bayaran anda ialah RM1,250.50.',
        'Boleh tolong confirm appointment saya pada hari Isnin?', 'Terima kasih kerana menghubungi kami, sila tunggu sebentar.',
        'Kod OTP anda ialah 482913.']
out = []
for s in SENT:
    ph = tts.phonemize(s)[0]
    ids = tts.ids(ph)[0].tolist()
    y = tts.session.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]
    env = [float(np.sqrt((y[i:i + 2400] ** 2).mean())) for i in range(0, len(y) - 2400, 2400)]
    out.append(dict(text=s, ph=ph, ids=ids, samples=len(y), env=env))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False)
print('wrote', sys.argv[1], len(out))
