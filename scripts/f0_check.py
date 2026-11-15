"""Is the pitch track we feed the decoder accurate enough for the upper harmonics? (CPU, no training)

For every steady voiced 12.5 ms frame, measure how much more energy sits ON the multiples of the pitch than
halfway BETWEEN them, in the 2-4 kHz band (the "comb contrast", in dB), using
  (a) the pitch exactly as the training rows store it (pYIN, see paradee_ms/ms_rows.py measure_f0), and
  (b) the pitch nudged by -3%..+3% to wherever the contrast is highest.
If (b) is much higher than (a), or the best nudge is often above ~1%, the stored pitch puts the decoder's
sine waves between the real harmonics, and a spectral loss can only answer that with noise.

    python scripts/f0_check.py --n 30
"""
import argparse
import glob

import librosa
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=30)
ap.add_argument('--shift', type=float, default=0.0, help='shift the pitch track by this many ms before testing')
A = ap.parse_args()
SR, NFFT, WIN, HOP = 24000, 4096, 1024, 300
BW = SR / NFFT
DELTAS = np.linspace(-0.03, 0.03, 61)


def measure_f0(a24, n, frame_length=1024):
    w16 = librosa.resample(a24, orig_sr=SR, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=70, fmax=450, sr=16000, frame_length=frame_length, hop_length=160)
    t = np.arange(len(f0)) * 0.01
    grid = (np.arange(n) + 0.5) * 0.0125 + A.shift / 1000
    ok = ~np.isnan(f0)
    if ok.sum() < 5:
        return np.zeros(n, np.float32)
    v = np.interp(grid, t, ok.astype(float)) > 0.5
    c = np.exp(np.interp(grid, t[ok], np.log(f0[ok])))
    return np.where(v, c, 0.0).astype(np.float32)


def contrast(P, f):
    """P [bins], f pitch in Hz -> contrast in dB for every delta."""
    k = np.arange(np.ceil(2000 / f), np.floor(4000 / f) + 1)
    ff = f * (1 + DELTAS)[:, None]
    on = np.clip(np.round(k[None] * ff / BW).astype(int), 0, len(P) - 1)
    off = np.clip(np.round((k[None] + 0.5) * ff / BW).astype(int), 0, len(P) - 1)
    return 10 * np.log10((P[on].mean(1) + 1e-12) / (P[off].mean(1) + 1e-12))


def clip_stats(path, frame_length=1024):
    a, sr = sf.read(path, dtype='float32')
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    n = len(a) // HOP
    f0 = measure_f0(a, n, frame_length)
    P = np.abs(librosa.stft(a[HOP // 2:], n_fft=NFFT, win_length=WIN, hop_length=HOP, center=True)) ** 2
    n = min(n, P.shape[1])
    out = []
    for i in range(2, n - 2):
        seg = f0[i - 2:i + 3]
        if seg.min() < 140 or seg.max() / seg.min() > 1.12:
            continue
        c = contrast(P[:, i], float(f0[i]))
        j = int(c.argmax())
        out.append((c[30], c[j], DELTAS[j], abs(f0[i + 1] - f0[i - 1]) / f0[i]))
    return np.array(out).reshape(-1, 4)


SETS = {
    'Chatterbox clips (training target), stored pitch': sorted(glob.glob('runs/k12/cbwav/*.wav'))[::25][:A.n],
    'Kokoro audio (control: clean synthetic speech)': sorted(glob.glob('runs/listen/tp/F1/*.wav'))[:A.n],
    'Suaradee v0.2 output': sorted(glob.glob('runs/k12/audio/A_*.wav'))[:A.n],
}
print(f"{'':52s} frames  contrast(stored) contrast(best)  strong  |nudge| med   >1%   >1.7%")
for name, files in SETS.items():
    S = np.concatenate([clip_stats(f) for f in files])
    strong = S[S[:, 1] > 8]
    d = np.abs(strong[:, 2])
    print(f'{name:52s} {len(S):6d}  {np.median(S[:, 0]):8.1f} dB     {np.median(S[:, 1]):6.1f} dB    {len(strong) / len(S):5.0%}  '
          f'{np.median(d) * 100:8.2f}%   {np.mean(d > 0.01):5.0%}  {np.mean(d > 0.017):5.0%}')
    if name.startswith('Chatterbox'):
        np.save('runs/f0_check_cb.npy', S)
        # is the error a constant offset (bias), which a time shift or a scale would fix?
        print(f'    mean signed nudge {strong[:, 2].mean() * 100:+.2f}%   on strong frames: contrast stored {np.median(strong[:, 0]):.1f} dB vs best {np.median(strong[:, 1]):.1f} dB')
        fast = strong[:, 3] > np.median(strong[:, 3])
        print(f'    fast-moving pitch frames: |nudge| {np.median(np.abs(strong[fast, 2])) * 100:.2f}%   steady frames: {np.median(np.abs(strong[~fast, 2])) * 100:.2f}%')
