"""Head-to-head: Yati v0.4.1 vs Michael v0.2.2 (latest published exports) on the same sentences.
Reports acoustic texture, consonant energy, speech rate and timing spread."""
import csv
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
    a = np.asarray(a, np.float32)
    a = a / (np.abs(a).max() + 1e-9)
    win, hop = 960, 480
    vals = []
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
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    flat = float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())
    o4 = float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean())
    o7 = float((P[f > 7000][:, keep].sum(0) / tot[keep]).mean())
    return flat, o4, o7


def cons(y):
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    fq = np.fft.rfftfreq(1024, 1 / SR)
    band = S[(fq > 2000) & (fq < 8000)].sum(0)
    flat = np.exp(np.log(S + 1e-9).mean(0)) / (S.mean(0) + 1e-9)
    return float(band[flat > np.percentile(flat, 70)].mean())


MODELS = {
    'yati v0.4.1': 'runs/k16_final/fixh_export/paradee_int8_webgpu.onnx',
    'michael v0.2.2': 'runs/michael_v022_export/paradee_int8_webgpu.onnx',
}
plan = {r['id']: (r['text'], r['category']) for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
# a spread across categories
ids = ['c0313', 'c0551', 'c0571', 'c0726', 'c0746', 'c0747']

sess = {k: ort.InferenceSession(v, providers=['CPUExecutionProvider']) for k, v in MODELS.items()}
res = {k: [] for k in MODELS}
os.makedirs('runs/listen/latest_cmp', exist_ok=True)
for i in ids:
    text, cat = plan[i]
    ids_t = np.array([g2p.tokens(g2p(normalize(text)))], dtype=np.int64)
    for k, s in sess.items():
        y = s.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        sf.write(f'runs/listen/latest_cmp/{i}_{k.split()[0]}.wav', y, SR)
        dur = len(y) / SR
        res[k].append((periodicity(y), *texture(y), cons(y), dur, len(text)))

print(f'{"model":16s} period  flat  >4k   >7k   cons   audio(s)  chars/s')
for k, rows in res.items():
    a = np.array([r[:6] for r in rows], np.float32)
    m = a.mean(0)
    rate = float(np.sum([r[6] for r in rows]) / np.sum([r[5] for r in rows]))
    print(f'{k:16s} {m[0]:.3f}  {m[1]:.3f} {m[2]:.3f} {m[3]:.3f} {m[4]:6.1f}  {m[5]:7.2f}   {rate:.2f}')
