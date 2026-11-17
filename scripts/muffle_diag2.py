"""Second look: average spectrum shape (dB rel. to 0-1 kHz) of Chatterbox targets vs v0.4 synth.
Also prints what fraction of frames even have ANY energy above 4 kHz."""
import csv
import os
import sys

import numpy as np
import torch
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N, SR = 25, 24000
targets = torch.load('runs/k15_v3/cb_clips2.pt', map_location='cpu', weights_only=False)
plan = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = [i for i in targets if i in plan][:N]

from suaradee.tts import Suaradee  # noqa: E402
tts = Suaradee('runs/k16_final/ms_c4_int8.onnx', threads=4)


def spec_db(a):
    a = np.asarray(a, np.float32)
    a = a / (np.abs(a).max() + 1e-9)
    S = np.abs(librosa.stft(a, n_fft=2048, hop_length=512)) ** 2
    fr = np.linspace(0, SR / 2, S.shape[0])
    lo = (fr >= 100) & (fr < 1000)
    base = 10 * np.log10(S[lo].mean(-1).mean() + 1e-12)
    out = {}
    for name, m in [('1-2k', (fr >= 1000) & (fr < 2000)), ('2-4k', (fr >= 2000) & (fr < 4000)),
                    ('4-6k', (fr >= 4000) & (fr < 6000)), ('6-8k', (fr >= 6000) & (fr < 8000)),
                    ('8-11k', (fr >= 8000) & (fr < 11000))]:
        out[name] = float(10 * np.log10(S[m].mean(-1).mean() + 1e-12) - base)
    # fraction of frames with meaningful energy above 4 kHz
    hi = (fr >= 4000) & (fr < 11000)
    out['frames_with_>4k'] = float(((S[hi].sum(0) > S[lo].sum(0) * 1e-3)).mean())
    return out


T, V = [], []
for i in ids:
    y = tts(plan[i])
    t = targets[i].numpy()
    n = min(len(y), len(t))
    T.append(spec_db(t[:n])); V.append(spec_db(y[:n]))

keys = ['1-2k', '2-4k', '4-6k', '6-8k', '8-11k', 'frames_with_>4k']
print(f'{"band":>16} {"target":>8} {"v0.4":>8}   (dB relative to 0-1 kHz)')
for k in keys:
    a = np.mean([r[k] for r in T]); b = np.mean([r[k] for r in V])
    print(f'{k:>16} {a:8.1f} {b:8.1f}')
