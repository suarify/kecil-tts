# kmb-18: score v0.2, the k14 clean model and the C4 model on the same rows (C4 gate).
# The C4 ONNX was exported by kmb-16 v3 and patched locally (webgpu_patch), so no training here:
# pure inference + Whisper large-v3 scoring, ~20 min on one T4.
import glob
import json
import os
import re
import subprocess
import sys

subprocess.run(f'{sys.executable} -m pip install -q onnxruntime onnx', shell=True, check=True)

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import pipeline

OUT = '/kaggle/working'
DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
ROWS = torch.load(glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0],
                  map_location='cpu', weights_only=False)
MODELS = {'c4': glob.glob('/kaggle/input/**/ms_c4_int8.onnx', recursive=True)[0]}
v02 = glob.glob('/kaggle/input/**/ms_A_int8_gpu.onnx', recursive=True)
if v02:
    MODELS['v02'] = v02[0]
clean = glob.glob('/kaggle/input/**/ms_clean_int8.onnx', recursive=True)
if clean:
    MODELS['clean'] = clean[0]
print('models:', {k: os.path.basename(v) for k, v in MODELS.items()}, flush=True)

sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
t200 = pd.read_csv(f'{DS}/test_200.tsv', sep='\t', dtype=str, keep_default_na=False)
asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)


def hear(a, sr=24000):
    w = torchaudio.functional.resample(torch.from_numpy(np.asarray(a, dtype=np.float32)), sr, 16000).numpy()
    return asr({'raw': w[:16000 * 29], 'sampling_rate': 16000}, batch_size=1,
               generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})['text'].strip()


def clean_txt(s):
    return re.sub(r'[^a-z]', '', normalize(s).lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return min(1.0, lev(clean_txt(ref), clean_txt(hyp)) / max(1, len(clean_txt(ref))))


def stats(v):
    v = np.array(v)
    return dict(n=len(v), median=round(float(np.median(v)), 3), capped_mean=round(float(v.mean()), 3),
                share_le_8=round(float((v <= .08).mean()), 2))


test_items = [(r.id, r.text, r.category, g2p.tokens(g2p(normalize(r.text)))) for r in t200.itertuples()]
held_items = [(r['id'], r['text'], 'held', r['ids'].numpy().tolist()) for r in ROWS[:60]]
rows = []
so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = {k: ort.InferenceSession(p, so, providers=['CPUExecutionProvider']) for k, p in MODELS.items()}
for tag, s in sess.items():
    for iid, text, cat, ids in test_items + held_items:
        y = s.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]
        h = hear(y)
        rows.append(dict(model=tag, id=iid, category=cat, text=text, cer=cer(text, h), heard=h))
    print('scored', tag, flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/test_scores_c4.csv', index=False)

res = {}
for tag in sess:
    d = pd.DataFrame([r for r in rows if r['model'] == tag])
    res[tag] = dict(overall=stats(d[d.category != 'held'].cer),
                    per_category={c: stats(g.cer) for c, g in d[d.category != 'held'].groupby('category')},
                    held=stats(d[d.category == 'held'].cer))

weak = ['general', 'codeswitch']
g = {}
for c in weak:
    b = res['v02']['per_category'].get(c, {}).get('median') if 'v02' in res else None
    n4 = res['c4']['per_category'].get(c, {}).get('median')
    g[c] = dict(v02=b, c4=n4, rel=round(1 - n4 / b, 3) if b else None,
                pass_=bool(b and n4 is not None and 1 - n4 / b >= 0.25))
g['overall_better'] = bool('v02' in res and res['c4']['overall']['capped_mean'] <= res['v02']['overall']['capped_mean'])
g['pass'] = bool(g['overall_better'] and all(g[c]['pass_'] for c in weak))
json.dump(dict(models=res, gates=g), open(f'{OUT}/c4_report.json', 'w'), indent=1)
print(json.dumps(dict(models=res, gates=g), indent=1), flush=True)
