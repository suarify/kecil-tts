"""Why does the trained model sound muffled?  Fresh band-by-band measurement (CPU only).

Compares the Chatterbox voice-converted TARGET clips the decoder was trained on
(runs/k15_v3/cb_clips2.pt) with what the v0.4 ONNX actually synthesises for the
SAME texts.  Reports band energy shares, spectral flatness 2-6 kHz and RMS for
both sides.  Usage: python scripts/muffle_diag.py
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa  # noqa: E402

N = 25          # clips to compare
SR = 24000

# --- targets ---
targets = torch.load('runs/k15_v3/cb_clips2.pt', map_location='cpu', weights_only=False)
plan = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
ids = [i for i in targets if i in plan][:N]
print(f'{len(ids)} matched clips (of {len(targets)} targets, {len(plan)} planned)')

# --- v0.4 synthesis ---
from suaradee.tts import Suaradee  # noqa: E402  (python/ is on sys.path via package dir)

MODEL = 'runs/k16_final/ms_c4_int8.onnx'
tts = Suaradee(MODEL, threads=4)


def bands(a):
    """Energy share per band + flatness 2-6 kHz + rms."""
    a = np.asarray(a, dtype=np.float32)
    a = a / (np.abs(a).max() + 1e-9)
    S = np.abs(librosa.stft(a, n_fft=2048, hop_length=512)) ** 2
    fr = np.linspace(0, SR / 2, S.shape[0])
    tot = S.sum()
    out = {}
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11000), (11000, 12000)]:
        m = (fr >= lo) & (fr < hi)
        out[f'{lo//1000}-{hi//1000}k'] = float(S[m].sum() / tot)
    f26 = (fr >= 2000) & (fr < 6000)
    P = S[f26] + 1e-12
    out['flat2-6k'] = float(np.exp(np.log(P).mean(-1).mean()) / (P.mean(-1).mean()))  # geometric/arith mean
    out['rms'] = float(np.sqrt((a ** 2).mean()))
    return out


# synthesize same texts
tts_res, tgt_res = [], []
for i in ids:
    try:
        y = tts(plan[i])
    except Exception as e:
        print('synth fail', i, str(e)[:80]); continue
    if len(y) < SR // 2:
        continue
    t = targets[i].numpy()
    n = min(len(y), len(t))
    tts_res.append(bands(y[:n])); tgt_res.append(bands(t[:n]))

keys = list(tts_res[0])
print(f'\n{"band":>9} {"target":>8} {"v0.4":>8} {"delta":>7}   (energy share; target = what the decoder was trained on)')
for k in keys:
    a = np.mean([r[k] for r in tgt_res]); b = np.mean([r[k] for r in tts_res])
    print(f'{k:>9} {a:8.4f} {b:8.4f} {b - a:+7.4f}')
