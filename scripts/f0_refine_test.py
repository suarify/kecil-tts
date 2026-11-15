"""Does refining the pitch (paradee_ms/f0_refine.py) find the real pitch, or does it only fit noise? (CPU)

Honest test: fit the nudge on the 1-2.5 kHz harmonics ONLY, then measure the contrast on 2.5-4 kHz, which the
fit never saw. If the unseen band gets cleaner, the nudge is a real pitch correction.
Also reports how much frame-to-frame wobble the refined track has compared with the stored one.

    python scripts/f0_refine_test.py --n 24
"""
import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paradee_ms'))
from f0_refine import DELTAS, comb, power_frames, refine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=24)
A = ap.parse_args()
SR = 24000
Z = len(DELTAS) // 2


def measure_f0(a24, n):
    w16 = librosa.resample(a24, orig_sr=SR, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=70, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    t = np.arange(len(f0)) * 0.01
    grid = (np.arange(n) + 0.5) * 0.0125
    ok = ~np.isnan(f0)
    if ok.sum() < 5:
        return np.zeros(n, np.float32)
    v = np.interp(grid, t, ok.astype(float)) > 0.5
    return np.where(v, np.exp(np.interp(grid, t[ok], np.log(f0[ok]))), 0.0).astype(np.float32)


def steady(f0, i):
    s = f0[i - 2:i + 3]
    return s.min() >= 140 and s.max() / s.min() <= 1.12


res = {k: [] for k in ('stored', 'fit_low', 'fit_all', 'best')}
wob = {k: [] for k in ('stored', 'fit_all')}
for p in sorted(glob.glob('runs/k12/cbwav/*.wav'))[7::25][:A.n]:
    a, _ = sf.read(p, dtype='float32')
    n = len(a) // 300
    f0 = measure_f0(a, n)
    P = power_frames(a, n)
    tracks = dict(stored=f0, fit_low=refine(a, f0, 1000, 2500), fit_all=refine(a, f0))
    for i in range(2, n - 2):
        if not steady(f0, i):
            continue
        for k, t in tracks.items():
            res[k].append(comb(P[i], float(t[i]), 2500, 4000)[Z])
        res['best'].append(comb(P[i], float(f0[i]), 2500, 4000).max())
    for k in wob:
        t = tracks[k]
        v = (t[1:] > 0) & (t[:-1] > 0)
        wob[k].append(np.abs(np.diff(np.log(np.maximum(t, 1))))[v])
print(f'frames {len(res["stored"])}; contrast on 2.5-4 kHz (median dB):')
print(f'  stored pYIN pitch                         {np.median(res["stored"]):5.1f}')
print(f'  refined on 1-2.5 kHz only (band unseen)   {np.median(res["fit_low"]):5.1f}')
print(f'  refined on 1-4 kHz (what training uses)   {np.median(res["fit_all"]):5.1f}')
print(f'  best single-frame nudge (upper bound)     {np.median(res["best"]):5.1f}')
for k in wob:
    w = np.concatenate(wob[k])
    print(f'  frame-to-frame pitch change, {k:8s}: median {np.median(w) * 100:.2f}%  95th pct {np.percentile(w, 95) * 100:.2f}%')
