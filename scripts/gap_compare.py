"""Is Michael's student worse than Yati's, relative to its own teacher?
Same measure for both: log-mel distance student vs teacher (lower = closer)."""
import csv
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

SR = 24000
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
import librosa  # noqa: E402


def mcd(a, b):
    M = lambda x: np.log(librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=SR, n_mels=64) + 1e-8)  # noqa: E731
    A, B = M(a), M(b)
    T = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :T] - B[:, :T]).mean())


def gap(student_onnx, teacher_getter, ids, texts):
    sess = ort.InferenceSession(student_onnx, providers=['CPUExecutionProvider'])
    d = []
    for i in ids:
        t = teacher_getter(i)
        if t is None:
            continue
        ids_t = np.array([g2p.tokens(g2p(normalize(texts[i])))], dtype=np.int64)
        y = sess.run(None, {'input_ids': ids_t, 'speed': np.ones(1, np.float32)})[0][0]
        d.append(mcd(y, t))
    return float(np.mean(d)), len(d)


# ---- Yati: teacher = cb_clips2.pt (her converted Chatterbox clips) ----
y_clips = torch.load('runs/k15_v3/cb_clips2.pt', map_location='cpu', weights_only=False)
plan = {r['id']: r['text'] for r in csv.DictReader(open('runs/k15_v3/gen_plan.csv', encoding='utf-8'))}
y_ids = [i for i in ['c0313', 'c0551', 'c0571', 'c0726', 'c0746', 'c0747'] if i in y_clips]
y_teacher = lambda i: y_clips[i].numpy().astype(np.float32) / 32767.0  # noqa: E731
gy, ny = gap('runs/k16_final/fixh_export/paradee_int8_webgpu.onnx', y_teacher, y_ids, plan)
print(f'yati   student vs teacher: {gy:.2f}  ({ny} clips)')

# ---- Michael: teacher = runs/listen/vc10/b (his converted clips) ----
m_texts = {}
for r in csv.DictReader(open('runs/sm_vc10_flat/texts.csv', encoding='utf-8')):
    m_texts[r['id']] = r['text']
m_ids = sorted(os.path.basename(p)[:-4] for p in glob.glob('runs/listen/vc10/b/*.wav'))


def m_teacher(i):
    p = f'runs/listen/vc10/b/{i}.wav'
    return sf.read(p)[0] if os.path.exists(p) else None


gm, nm = gap('runs/michael_v03_export/paradee_int8_webgpu.onnx', m_teacher, m_ids, m_texts)
print(f'michael student vs teacher: {gm:.2f}  ({nm} clips)')

# ---- reference: how far the two Michael conversions are from each other ----
pairs = [mcd(sf.read(f'runs/listen/vc10/b/{i}.wav')[0], sf.read(f'runs/listen/vc10/a/{i}.wav')[0]) for i in m_ids]
print(f'michael teacher-vs-teacher (2 refs): {np.mean(pairs):.2f}')
