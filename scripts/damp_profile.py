"""A1: objective damp profile of systems 1 (Chatterbox clips), 3 (v0.2 from text) and 4 (Kokoro Bella).

30 held-out rows each. Measures, median across clips per system:
  - level per third-octave band, long-term average (re 1 kHz band, dB)
  - rolloff: frequency below which 95% of the energy sits
  - energy above 4 kHz and above 8 kHz (share of total)
  - onset sharpness: median rise of the 2-8 kHz envelope at detected word onsets (dB per 10 ms)
  - decay: median time for the 2-8 kHz envelope to fall 10 dB after a peak
  - pitch range: 10th-90th percentile of voiced F0, in semitones

    PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe scripts/damp_profile.py
"""
import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from student import alignment  # noqa: E402
from vhead import VDecoder  # noqa: E402

N_CLIP = 30
SHARD = os.path.join(ROOT, 'runs/k12_rows/rows/shard_0000.pt')
DEC = os.path.join(ROOT, 'runs/k12/ckpt/dec_last.pt')
ONNX = os.path.join(ROOT, 'runs/k12/ms_A_int8_gpu.onnx')
CBWAV = os.path.join(ROOT, 'runs/k12/cbwav')
SR = 24000
torch.set_num_threads(4)

rows = torch.load(SHARD, map_location='cpu', weights_only=False)
pick = rows[:N_CLIP]                       # held-out rows, same set the k12 gates scored

# ---------------- render systems 3 and 4 ----------------
vd = VDecoder(head='v2', dim=160, blocks=4).eval()
vd.load_ckpt(DEC)
sys2, sys3, sys4 = [], [], []
with torch.no_grad():
    for r in pick:
        aln = alignment(r['pred_dur'])
        Fr = aln.shape[1]
        torch.manual_seed(0)
        y = vd((r['t_en'].float() @ aln)[None], r['F0'][None, :2 * Fr], r['N'][None, :2 * Fr]).squeeze().numpy()
        sys2.append(y.astype(np.float32))
    print('decoder renders done', flush=True)
del vd
from suaradee import Suaradee  # noqa: E402

tts = Suaradee(ONNX)
sys3 = [tts(r['text']) for r in pick]
print('v0.2 renders done', flush=True)

import types  # noqa: E402


def _stub(name):
    m = types.ModuleType(name)

    def _ga(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return type(attr, (), {})
    m.__getattr__ = _ga
    return m


sys.modules.setdefault('misaki', _stub('misaki'))
for sub in ('en', 'espeak'):
    sys.modules.setdefault('misaki.' + sub, _stub('misaki.' + sub))
    setattr(sys.modules['misaki'], sub, sys.modules['misaki.' + sub])
from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

km = KModel(repo_id='hexgrad/Kokoro-82M').eval()
bella = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_bella.pt'), weights_only=True)
with torch.no_grad():
    for r in pick:
        sys4.append(km(r['ps'][:510], bella[len(r['ps'][:510]) - 1], 1.0).float().numpy())
print('kokoro renders done', flush=True)

sys1 = [sf.read(os.path.join(CBWAV, r['id'] + '.wav'), dtype='float32')[0] for r in pick]

# ---------------- measures ----------------
def third_octave_db(a):
    S = np.abs(librosa.stft(a, n_fft=2048, hop_length=256)) ** 2
    f = np.fft.rfftfreq(2048, 1 / SR)
    tot = S.sum(1)
    centers = 1000.0 * 2 ** (np.arange(-15, 9) / 3)      # ~99 Hz ... ~16 kHz
    with np.errstate(divide='ignore'):
        out = []
        for c in centers:
            lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
            band = tot[(f >= lo) & (f < hi)].sum()
            out.append(10 * np.log10(band / len(S[0]) + 1e-12))
    return np.array(out) - 10 * np.log10(tot.sum() / len(S[0]) + 1e-12)


def energy_shares(a):
    S = np.abs(librosa.stft(a, n_fft=2048, hop_length=256)) ** 2
    f = np.fft.rfftfreq(2048, 1 / SR)
    tot = S.sum()
    acc = np.cumsum(S.sum(1))
    rolloff = float(f[np.searchsorted(acc, 0.95 * acc[-1])])
    return rolloff, float(S[f > 4000].sum() / tot), float(S[f > 8000].sum() / tot)


def env_stats(a):
    """Onset sharpness and decay of the 2-8 kHz envelope."""
    S = np.abs(librosa.stft(a, n_fft=512, hop_length=int(0.010 * SR))) ** 2      # 10 ms frames
    f = np.fft.rfftfreq(512, 1 / SR)
    env = 10 * np.log10(S[(f > 2000) & (f < 8000)].sum(0) + 1e-12)
    rise = env[1:] - env[:-1]
    # onset = a rise frame, at least 200 ms after the previous onset, that is followed by a local max
    idx, last = [], -10 ** 9
    for i in np.argsort(rise)[::-1]:
        if rise[i] < 2.0:
            break
        if i - last >= 20 and i + 1 < len(env) and env[i + 1] >= env[max(0, i - 2):i + 4].max() - 0.5:
            idx.append(i)
            last = i
        if len(idx) >= 12:
            break
    onset = float(np.median(rise[idx])) if len(idx) >= 3 else np.nan
    # decay: time for the envelope to fall 10 dB after a peak
    decays = []
    for p in idx:
        tgt = env[p + 1] - 10.0
        after = env[p + 1:p + 51]          # 500 ms window
        hit = np.where(after <= tgt)[0]
        if len(hit):
            decays.append(hit[0] * 0.010)
    decay = float(np.median(decays)) if len(decays) >= 3 else np.nan
    return onset, decay


def pitch_range(a):
    f0, vflag, _ = librosa.pyin(a, fmin=60, fmax=450, sr=SR, frame_length=1024)
    v = f0[vflag == 1]
    if len(v) < 10:
        return np.nan
    st = 12 * np.log2(v / np.median(v))
    return float(np.percentile(st, 90) - np.percentile(st, 10))


def profile(audios):
    return dict(
        rolloff=float(np.median([energy_shares(a)[0] for a in audios])),
        over_4k=float(np.median([energy_shares(a)[1] for a in audios])),
        over_8k=float(np.median([energy_shares(a)[2] for a in audios])),
        onset_db_10ms=float(np.nanmedian([env_stats(a)[0] for a in audios])),
        decay_s=float(np.nanmedian([env_stats(a)[1] for a in audios])),
        pitch_span_st=float(np.nanmedian([pitch_range(a) for a in audios])),
        bands=np.median([third_octave_db(a) for a in audios], axis=0).round(1).tolist(),
    )


centers = (1000.0 * 2 ** (np.arange(-15, 9) / 3)).round(0).astype(int)
res = {name: profile(a) for name, a in
       (('1_chatterbox', sys1), ('2_decoder_measured', sys2), ('3_v02_text', sys3), ('4_kokoro_bella', sys4))}
lines = [f'| Third-octave centre {c} Hz | ' + ' | '.join(f"{res[n]['bands'][i]:+.1f}" for n in res) + ' |'
         for i, c in enumerate(centers)]
table = ['| Measure | ' + ' | '.join(res) + ' |', '|---|' + '---|' * len(res)]
for k in ('rolloff', 'over_4k', 'over_8k', 'onset_db_10ms', 'decay_s', 'pitch_span_st'):
    table.append('| ' + k + ' | ' + ' | '.join(f"{res[n][k]:.2f}" for n in res) + ' |')
out = os.path.join(ROOT, 'runs/listen/damp_profile')
os.makedirs(out, exist_ok=True)
import json  # noqa: E402

json.dump(dict(centers=centers.tolist(), **{k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}),
          open(os.path.join(out, 'profile.json'), 'w'), indent=1)
print('| Measure | 1 Chatterbox | 2 decoder | 3 v0.2 | 4 Kokoro |')
for row in table:
    print(row)
print('third-octave bands (re total, dB):')
print('| Band Hz | ' + ' | '.join(res) + ' |')
for ln in lines:
    print(ln)
