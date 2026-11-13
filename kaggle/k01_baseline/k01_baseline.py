# kokoro-malay-bella / Phase 2: zero-training baseline (Gate 2a).
# Stock Kokoro-82M + af_bella reads Malay phonemes from our G2P under 8 option variants.
# Scored by Whisper large-v3 round-trip CER. Also renders the English reference clips
# (Bella, Sarah) that Phase 3 uses for cloning and speaker-similarity calibration.
import json
import os
import re
import subprocess
import sys
import time
import types

T0 = time.time()


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"')
sh('pip install -q loguru')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


def _stub(name):
    m = types.ModuleType(name)

    def _ga(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return type(attr, (), {})
    m.__getattr__ = _ga
    return m


# KModel does not need misaki; the package imports it eagerly, so stub it.
sys.modules['misaki'] = _stub('misaki')
for sub in ('en', 'espeak'):
    sys.modules['misaki.' + sub] = _stub('misaki.' + sub)
    setattr(sys.modules['misaki'], sub, sys.modules['misaki.' + sub])

from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

DS = '/kaggle/input/kokoro-malay-g2p'
sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

OUT = '/kaggle/working'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SR = 24000
print('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '')

model = KModel(repo_id='hexgrad/Kokoro-82M').to(DEV).eval()


def voice(name):
    return torch.load(hf_hub_download('hexgrad/Kokoro-82M', f'voices/{name}.pt'), weights_only=True)


BELLA, SARAH = voice('af_bella'), voice('af_sarah')


def synth(ph, pack, speed=1.0):
    ph = ph[:510]
    with torch.no_grad():
        audio = model(ph, pack[len(ph) - 1].to(DEV), speed)
    return audio.float().cpu().numpy()


def read_tsv(name):
    return pd.read_csv(os.path.join(DS, name), sep='\t', dtype=str, keep_default_na=False)


test = read_tsv('test_40.tsv')
etest = read_tsv('e_test.tsv')
eng = read_tsv('en_ref.tsv')
items = [(r.id, r.category, r.text) for r in test.itertuples()] + \
        [('e_' + r.word, 'etest', r.text) for r in etest.itertuples()]

# ---------------- 1. English reference clips (true Bella / Sarah) ----------------
for name, pack in (('bella_en', BELLA), ('sarah_en', SARAH)):
    os.makedirs(f'{OUT}/refs/{name}', exist_ok=True)
    rows = eng if name == 'bella_en' else eng.iloc[:6]
    for r in rows.itertuples():
        sf.write(f'{OUT}/refs/{name}/{r.id}.wav', synth(r.phonemes, pack), SR)

gap = np.zeros(int(0.25 * SR), dtype=np.float32)
parts = []
for r in eng.itertuples():
    a, _ = sf.read(f'{OUT}/refs/bella_en/{r.id}.wav', dtype='float32')
    parts += [a, gap]
    if sum(len(p) for p in parts) > 14 * SR:
        break
ref_en = np.concatenate(parts)[:20 * SR]
sf.write(f'{OUT}/refs/ref_en.wav', ref_en, SR)
print('ref_en seconds', round(len(ref_en) / SR, 1))

# ---------------- 2. Baseline: 8 G2P variants ----------------
VARIANTS = {}
for st in (True, False):
    for r in ('r', 'ɾ'):
        for d in ('glide', 'kokoro'):
            VARIANTS[f"s{int(st)}_{'r' if r == 'r' else 'tap'}_{d}"] = dict(stress=st, r=r, diph=d)

rows = []
for vname, opts in VARIANTS.items():
    g = G2P(**opts)
    os.makedirs(f'{OUT}/baseline/{vname}', exist_ok=True)
    for iid, cat, text in items:
        norm = normalize(text)
        ph = g(norm)
        audio = synth(ph, BELLA)
        path = f'{OUT}/baseline/{vname}/{iid}.wav'
        sf.write(path, audio, SR)
        rows.append(dict(variant=vname, id=iid, category=cat, text=text, norm=norm, ph=ph,
                         path=path, seconds=len(audio) / SR, n_ph=len(ph)))
df = pd.DataFrame(rows)
print('synthesised', len(df), 'clips in', round(time.time() - T0), 's')

# ---------------- 3. Whisper large-v3 round trip ----------------
del model
torch.cuda.empty_cache()
from transformers import pipeline  # noqa: E402

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0 if DEV == 'cuda' else -1)


def transcribe(paths):
    batch = []
    for p in paths:
        a, sr = sf.read(p, dtype='float32')
        batch.append({'raw': a, 'sampling_rate': sr})
    out = asr(batch, batch_size=16, generate_kwargs={'language': 'malay', 'task': 'transcribe'})
    return [o['text'].strip() for o in out]


df['hyp'] = transcribe(df.path.tolist())


def clean(s):
    s = normalize(s).lower()
    return re.sub(r'[^a-z]', '', s)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


df['cer'] = [lev(clean(r), clean(h)) / max(1, len(clean(r))) for r, h in zip(df.norm, df.hyp)]
df['ph_per_s'] = df.n_ph / df.seconds
df.drop(columns=['path']).to_csv(f'{OUT}/baseline_scores.csv', index=False)

main = df[df.category != 'etest']
summary = main.groupby('variant').agg(cer=('cer', 'mean'), cer_median=('cer', 'median'),
                                      ph_per_s=('ph_per_s', 'mean')).sort_values('cer')
by_cat = main.pivot_table(index='variant', columns='category', values='cer', aggfunc='mean')
summary = summary.join(by_cat)
print(summary.round(4).to_string())

# Best variant: lowest mean CER; within 0.002 of the best, prefer the plan default.
DEFAULT = 's1_r_glide'
best = summary.index[0]
if DEFAULT in summary.index and summary.loc[DEFAULT, 'cer'] - summary.loc[best, 'cer'] <= 0.002:
    best = DEFAULT
print('BEST VARIANT', best)

# e-test: did Whisper hear the word?
et = df[(df.category == 'etest') & (df.variant == best)].copy()
et['word'] = et.id.str[2:]
et['heard'] = [w in clean(h) for w, h in zip(et.word, et.hyp)]
print('e-test words recognised (best variant):', int(et.heard.sum()), '/', len(et))

# Ref-MS: cleanest general sentences of the best variant, ~15 s.
pick = main[(main.variant == best) & (main.category == 'general')].sort_values('cer')
parts = []
for r in pick.itertuples():
    a, _ = sf.read(f'{OUT}/baseline/{best}/{r.id}.wav', dtype='float32')
    parts += [a, gap]
    if sum(len(p) for p in parts) > 14 * SR:
        break
ref_ms = np.concatenate(parts)[:20 * SR]
sf.write(f'{OUT}/refs/ref_ms.wav', ref_ms, SR)

report = {
    'best_variant': best, 'options': VARIANTS[best],
    'summary': json.loads(summary.round(4).to_json(orient='index')),
    'etest_recognised': int(et.heard.sum()), 'etest_total': len(et),
    'ref_en_seconds': round(len(ref_en) / SR, 1), 'ref_ms_seconds': round(len(ref_ms) / SR, 1),
    'runtime_s': round(time.time() - T0),
}
json.dump(report, open(f'{OUT}/baseline_report.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
