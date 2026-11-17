"""How close is the student (v0.3.0) to its own teacher clips? Same 15 texts:
teacher (old, 2 conversions) vs teacher (new, 1 conversion) vs student v0.3.0 output."""
import csv
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

SR = 24000
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')


def periodicity(a):
    a = np.asarray(a, np.float32); a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480; vals = []
    for i in range(0, len(a) - win, hop):
        fr_ = a[i:i + win]
        if np.sqrt((fr_ ** 2).mean()) < 0.02:
            continue
        fr_ = fr_ - fr_.mean()
        ac = np.correlate(fr_, fr_, 'full')[win - 1:]
        ac /= ac[0] + 1e-12
        vals.append(float(ac[int(SR / 200):int(SR / 80)].max()))
    return float(np.median(vals)) if vals else 0.0


def texture(a):
    import librosa
    S = np.abs(librosa.stft(np.asarray(a, np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2; tot = P.sum(0); keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    o4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    o7 = float((P[f > 7000][:, keep].sum(0) / tot[keep]).mean())
    return flat, o4, o7


def mcd(a, b):
    """crude spectral distance: mean |log mel spectra| (0 = identical)."""
    import librosa
    M = lambda x: np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR, n_mels=64) + 1e-8)  # noqa: E731
    A, B = M(a), M(b)
    T = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :T] - B[:, :T]).mean())


texts = {r['id']: r['text'] for r in csv.DictReader(open('runs/sm_vc15b_flat/texts.csv', encoding='utf-8'))}
sess = ort.InferenceSession('runs/michael_v03_export/paradee_int8_webgpu.onnx', providers=['CPUExecutionProvider'])
ids = [i for i in texts if os.path.exists(f'runs/listen/vc15b/old/{i}.wav') and os.path.exists(f'runs/listen/vc15b/single/{i}.wav')]

rows = {'teacher old (2 VC)': [], 'teacher new (1 VC)': [], 'student v0.3.0': []}
dist_old, dist_new = [], []
os.makedirs('runs/listen/fidelity', exist_ok=True)
for i in ids:
    t_old, _ = sf.read(f'runs/listen/vc15b/old/{i}.wav')
    t_new, _ = sf.read(f'runs/listen/vc15b/single/{i}.wav')
    ids_t = np.array([g2p.tokens(g2p(normalize(texts[i])))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f'runs/listen/fidelity/{i}_student.wav', y, SR)
    for k, a in (('teacher old (2 VC)', t_old), ('teacher new (1 VC)', t_new), ('student v0.3.0', y)):
        rows[k].append([periodicity(a), *texture(a)])
    dist_old.append(mcd(y, t_old))
    dist_new.append(mcd(y, t_new))

print(f'{"source":20s} periodic  flat   >4k    >7k')
for k, v in rows.items():
    m = np.mean(v, 0)
    print(f'{k:20s} {m[0]:.3f}    {m[1]:.3f}  {m[2]:.3f}  {m[3]:.3f}')
print(f'\nspectral distance student vs teacher(old): {np.mean(dist_old):.3f}')
print(f'spectral distance student vs teacher(new): {np.mean(dist_new):.3f}')
print(f'spectral distance teacher old vs new     : {np.mean([mcd(sf.read(f"runs/listen/vc15b/old/{i}.wav")[0], sf.read(f"runs/listen/vc15b/single/{i}.wav")[0]) for i in ids]):.3f}')
