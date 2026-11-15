"""Which half of the k09 student causes the word errors and the damp sound?  (CPU, existing files only)

For the held-out sentences of run k09 we have three versions of the same sentence:
  target      the Chatterbox clip (what the student should sound like)
  decoder     the 8M decoder fed the MEASURED timings/pitch/loudness of that clip (text side bypassed)
  student     the full 8M model from text only (text side + decoder) — what ships
Same ASR (Whisper-small, CPU) on all three, plus brightness numbers.

Run:  python scripts/diagnose_k09.py [n]
"""
import os
import re
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from ms_norm import normalize  # noqa: E402
from transformers import pipeline  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K9 = os.path.join(ROOT, 'runs', 'k09')
asr = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=-1)


def hear(a, sr):
    w = librosa.resample(np.asarray(a, dtype=np.float32), orig_sr=sr, target_sr=16000)
    return asr({'raw': w, 'sampling_rate': 16000}, generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 120})['text'].strip()


def clean(s):
    return re.sub(r'[^a-z]', '', normalize(s).lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return min(1.0, lev(clean(ref), clean(hyp)) / max(1, len(clean(ref))))


def bright(a, sr=24000):
    S = np.abs(librosa.stft(np.asarray(a, dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / sr)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    return [float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean()), float((P[(f > 1500) & (f < 4000)][:, keep].sum(0) / tot[keep]).mean()),
            float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())]


sc = pd.read_csv(os.path.join(K9, 'e2e_scores.csv'))
held = sc[sc.split == 'held'].reset_index(drop=True).head(N)
rows = []
for k, r in held.iterrows():
    f_t = os.path.join(K9, 'e2e', 'ckpt', 'dec', f'held_{k + 1}_target.wav')
    f_d = os.path.join(K9, 'e2e', 'ckpt', 'dec', f'held_{k + 1}_student.wav')
    f_s = os.path.join(K9, 'audio', f'held_{r.id}_onnx_int8.wav')
    f_t2 = os.path.join(K9, 'audio', f'held_{r.id}_target.wav')
    t, sr = sf.read(f_t, dtype='float32')
    t2, _ = sf.read(f_t2, dtype='float32')
    assert abs(len(t) - len(t2)) < 700, (k, r.id, len(t), len(t2))        # same clip: the mapping is right
    d, _ = sf.read(f_d, dtype='float32')
    s, _ = sf.read(f_s, dtype='float32')
    row = dict(id=r.id, text=r.text)
    for name, a in (('target', t), ('decoder', d), ('student', s)):
        h = hear(a, sr)
        row[f'cer_{name}'] = cer(r.text, h)
        row[f'heard_{name}'] = h
        row[f'hi_{name}'], row[f'mid_{name}'], row[f'flat_{name}'] = bright(a)
    rows.append(row)
    print(k, r.id, {n: round(row[f'cer_{n}'], 2) for n in ('target', 'decoder', 'student')}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(K9, 'diagnose.csv'), index=False)
print('\nsame ASR on all three (Whisper-small), n =', len(df))
for n in ('target', 'decoder', 'student'):
    c = df[f'cer_{n}']
    print(f'{n:8s} CER mean {c.mean():.3f} median {c.median():.3f} | <=8% {(c <= .08).mean():.2f} | energy >4k {df[f"hi_{n}"].mean():.3f}  1.5-4k {df[f"mid_{n}"].mean():.3f}  flatness {df[f"flat_{n}"].mean():.3f}')
