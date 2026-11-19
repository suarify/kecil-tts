# sm24-cer-verify: re-score the SAME 15 generated wavs with the proven k18 pipeline scorer
# vs the direct WhisperForConditionalGeneration scorer used in sm23.
import csv
import glob
import json
import os
import re
import subprocess
import sys


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    if p.returncode:
        raise SystemExit(f'FAILED {cmd}: {p.stderr[-1500:]}')


sh('pip install -q soundfile')
sh('pip uninstall -y -q torchvision torchao torchtext || true')
sh('pip install -q "transformers==4.46.3"')
sh(f'{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchaudio==2.11.0 '
   '--index-url https://download.pytorch.org/whl/cu130')
sh(f'{sys.executable} -c "import torch, torchaudio; print(\'PAIR\', torch.__version__, torchaudio.__version__, torch.cuda.is_available())"')

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

T0 = os.environ.get('T0', '')
OUT = '/kaggle/working'
REFS = glob.glob('/kaggle/input/**/refs.tsv', recursive=True)[0]
WAVS = sorted(glob.glob('/kaggle/input/**/wav_*.wav', recursive=True))
refs = {r['id']: r for r in csv.DictReader(open(REFS), delimiter='\t')}
print('refs', len(refs), 'wavs', len(WAVS), flush=True)

# ---- scorer 1: the project-validated k18 pipeline ----
asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               device=0, torch_dtype=torch.float16)


def hear_pipe(path, sr=24000):
    a, s = sf.read(path, dtype='float32')
    w = torchaudio.functional.resample(torch.from_numpy(a), s, 16000).numpy()
    return asr({'raw': w[:16000 * 29], 'sampling_rate': 16000}, batch_size=1,
               generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})['text'].strip()


# ---- scorer 2: direct model (as used in sm23) ----
proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).to('cuda').eval()


def hear_direct(path, sr=24000):
    a, s = sf.read(path, dtype='float32')
    w = torchaudio.functional.resample(torch.from_numpy(a), s, 16000).numpy()[:16000 * 29]
    f = proc.feature_extractor(w, sampling_rate=16000, return_tensors='pt').input_features.to('cuda', torch.float16)
    with torch.inference_mode():
        ids = wm.generate(f, language='malay', task='transcribe', max_new_tokens=200)
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def clean(s):
    return re.sub(r'[^a-z]', '', s.lower())


def cer(ref, hyp):
    a, b = clean(ref), clean(hyp)
    prev = list(range(len(b) + 1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(1.0, prev[-1] / max(1, len(a)))


rows = []
for p in WAVS:
    i = os.path.basename(p).replace('wav_', '').replace('.wav', '')
    r = refs[i]
    h1 = hear_pipe(p)
    h2 = hear_direct(p)
    c1, c2 = cer(r['teacher'], h1), cer(r['teacher'], h2)
    rows.append(dict(id=i, ref=r['teacher'][:90], pipe_hyp=h1[:90], direct_hyp=h2[:90],
                     pipe_cer=round(c1, 4), direct_cer=round(c2, 4)))
    print(i, 'PIPE', round(c1, 3), '|', h1[:90], flush=True)
    print(i, 'DIR ', round(c2, 3), '|', h2[:90], flush=True)

res = dict(pipe_median=round(float(np.median([r['pipe_cer'] for r in rows])), 4),
           pipe_mean=round(float(np.mean([r['pipe_cer'] for r in rows])), 4),
           direct_median=round(float(np.median([r['direct_cer'] for r in rows])), 4),
           direct_mean=round(float(np.mean([r['direct_cer'] for r in rows])), 4),
           rows=rows)
json.dump(res, open(f'{OUT}/sm24_report.json', 'w'), indent=1)
print('RESULT', {k: v for k, v in res.items() if k != 'rows'}, flush=True)
print('SM24 DONE', flush=True)
