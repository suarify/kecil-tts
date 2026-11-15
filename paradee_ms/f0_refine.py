"""Refine a pitch track against the audio it was measured from (continue.plan.v6.md).

pYIN is good to about 1%, which is fine at the pitch itself but puts the 15th harmonic 15% of the way to the
next one. Here every voiced frame's pitch is nudged by up to 3% to where the clip's harmonics really are:
the nudge that gives the most energy ON the multiples of the pitch compared with halfway BETWEEN them.

    refine(audio24, f0)          numpy in, numpy out; f0 at 80 frames/s, frame i centred at (i + 0.5) * 12.5 ms
    python f0_refine.py IN.pt OUT.pt     rewrite a row shard: F0 becomes the refined track, F0_pyin keeps the old one
"""
import sys

import numpy as np

SR, NFFT, WIN, HOP = 24000, 4096, 1024, 300
BW = SR / NFFT
DELTAS = np.linspace(-0.03, 0.03, 61)
_W = np.hanning(WIN + 1)[:WIN].astype(np.float32)


def power_frames(a, n):
    """Power spectrum of frame i, centred at sample (i + 0.5) * HOP. [n, bins]"""
    half = WIN // 2
    x = np.pad(np.asarray(a, dtype=np.float32), (half, half + HOP * (n + 1)))
    idx = (np.arange(n) * HOP + HOP // 2)[:, None] + np.arange(WIN)[None]
    return np.abs(np.fft.rfft(x[idx] * _W, NFFT, axis=1)) ** 2


def comb(P, f, lo=1000.0, hi=4000.0):
    """Contrast in dB for every nudge in DELTAS. P [bins] power, f pitch in Hz."""
    k = np.arange(np.ceil(lo / f), np.floor(hi / f) + 1)
    if len(k) == 0:
        return np.zeros(len(DELTAS))
    ff = f * (1 + DELTAS)[:, None]
    on = np.clip(np.round(k[None] * ff / BW).astype(int), 0, len(P) - 1)
    off = np.clip(np.round((k[None] + 0.5) * ff / BW).astype(int), 0, len(P) - 1)
    return 10 * np.log10((P[on].mean(1) + 1e-12) / (P[off].mean(1) + 1e-12))


def refine(a, f0, lo=1000.0, hi=4000.0, strong=8.0, fmin=120.0, med=5, return_nudge=False):
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    P = power_frames(a, n)
    nud = np.full(n, np.nan)
    for i in np.nonzero(f0 >= fmin)[0]:
        c = comb(P[i], f0[i], lo, hi)
        j = int(c.argmax())
        if c[j] >= strong:
            nud[i] = DELTAS[j]
    out = np.zeros(n)
    voiced = f0 > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(int), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):                      # one voiced run at a time
        seg = nud[s:e]
        ok = ~np.isnan(seg)
        if ok.sum() == 0:
            continue
        x = np.arange(e - s)
        filled = np.interp(x, x[ok], seg[ok])                       # frames with no clear harmonics borrow from neighbours
        pad = np.pad(filled, med // 2, mode='edge')
        out[s:e] = np.median(np.lib.stride_tricks.sliding_window_view(pad, med), axis=1)
    f0r = (f0 * (1 + out)).astype(np.float32)
    return (f0r, out) if return_nudge else f0r


if __name__ == '__main__':
    import torch

    src, dst = sys.argv[1], sys.argv[2]
    rows = torch.load(src)
    moved, cover = [], []
    for r in rows:
        f0 = r['F0'].numpy()
        au = r['audio'].float().numpy() / 32767
        f0r, nud = refine(au, f0, return_nudge=True)
        v = f0 > 0
        moved.append(np.abs(nud[v]))
        cover.append(float((nud[v] != 0).mean()) if v.any() else 0.0)
        r['F0_pyin'] = r['F0']
        r['F0'] = torch.from_numpy(f0r)
    m = np.concatenate(moved)
    torch.save(rows, dst)
    print(f'refined {len(rows)} rows | median |nudge| {np.median(m) * 100:.2f}% | over 1%: {np.mean(m > 0.01):.0%} | '
          f'voiced frames nudged: {np.mean(cover):.0%}', flush=True)
