"""Third look: spectral shape (dB rel 0-1 kHz) of
  (a) original Kokoro Bella output  (runs/k01/baseline/s0_r_glide/*.wav)
  (b) raw Chatterbox wav for the same k15 clip ids (runs/k15_v3/raw2/<id>_1234.wav)
  (c) the chosen target clips in cb_clips2.pt
"""
import glob
import os
import sys

import numpy as np
import torch
import librosa
import soundfile as sf

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 24000


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
    return out


def agg(files, loader):
    R = []
    for f in files:
        a, sr = loader(f)
        if sr != SR:
            a = librosa.resample(a, orig_sr=sr, target_sr=SR)
        if len(a) < SR:
            continue
        R.append(spec_db(a))
    keys = ['1-2k', '2-4k', '4-6k', '6-8k', '8-11k']
    return {k: float(np.mean([r[k] for r in R])) for k in keys}, len(R)


kokoro = sorted(glob.glob('runs/k01/baseline/s0_r_glide/c0*.wav'))
raw = sorted(glob.glob('runs/k15_v3/raw2/*_1234.wav'))[:25]
clips = torch.load('runs/k15_v3/cb_clips2.pt', map_location='cpu', weights_only=False)
chosen = [clips[os.path.basename(f).split('_')[0]].numpy() for f in raw if os.path.basename(f).split('_')[0] in clips]

for name, files, loader in [
        ('Kokoro Bella (bright ref)', kokoro, lambda f: sf.read(f, dtype='float32')),
        ('Chatterbox raw (k15)', raw, lambda f: sf.read(f, dtype='float32')),
        ('cb_clips2 (training target)', None, None)]:
    if loader is None:
        R = [spec_db(a[:SR * 20]) for a in chosen]
        keys = ['1-2k', '2-4k', '4-6k', '6-8k', '8-11k']
        res = {k: float(np.mean([r[k] for r in R])) for k in keys}
        n = len(R)
    else:
        res, n = agg(files, loader)
    print(f'{name:28s} n={n:3d}  ' + '  '.join(f'{k}={v:6.1f}' for k, v in res.items()))
