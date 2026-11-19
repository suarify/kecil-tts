# sm25-cer50: representative 50-sentence CER + speed retest with the baseline Chatterbox config.
# 20 general / 10 conversational / 10 codeswitch / 10 numbers from the 3,400 Phase 4 sentences.
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np


def sh(cmd, name=None):
    t = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    if p.stdout.strip():
        print(p.stdout[-1200:], flush=True)
    if p.returncode:
        print(p.stderr[-2500:], flush=True)
        json.dump(dict(failed=name, tail=p.stderr[-2500:]), open('/kaggle/working/sm25_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    if name:
        print(f'=== {name}: {time.time() - t:.1f}s', flush=True)


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
sh('pip install -q chatterbox-tts soundfile', 'pip_cb')
sh('pip uninstall -y -q torchvision torchao torchtext || true', 'rm_extras')
sh('pip install -q "transformers==4.46.3"', 'pip_tr')
sh(f'{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchaudio==2.11.0 '
   '--index-url https://download.pytorch.org/whl/cu130', 'torch_pair')
sh(f'{sys.executable} -c "import torch, torchaudio; print(\'PAIR\', torch.__version__, torchaudio.__version__, torch.cuda.is_available())"',
   'verify_pair')

TEXTS = glob.glob('/kaggle/input/**/texts.tsv', recursive=True)[0]
REF = glob.glob('/kaggle/input/**/ref_ms.wav', recursive=True)[0]
plan = list(csv.DictReader(open(TEXTS), delimiter='\t'))
print('texts', len(plan), flush=True)
assert len(plan) == 50

import torch  # noqa: E402
import soundfile as sf  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402
from transformers import pipeline  # noqa: E402

tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')
asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               device=0, torch_dtype=torch.float16)
import torchaudio  # noqa: E402


def hear(path, sr=24000):
    a, s = sf.read(path, dtype='float32')
    w = torchaudio.functional.resample(torch.from_numpy(a), s, 16000).numpy()
    return asr({'raw': w[:16000 * 29], 'sampling_rate': 16000}, batch_size=1,
               generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})['text'].strip()


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


OUT = '/kaggle/working/p4_cer50'
os.makedirs(OUT, exist_ok=True)
rows, t = [], time.time()
for i, r in enumerate(plan):
    torch.manual_seed(1234)
    try:
        wav = tts.generate(r['text'], language_id='ms', audio_prompt_path=REF,
                           exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
    except Exception as e:
        print('GEN FAIL', i, type(e).__name__, str(e)[:120], flush=True)
        continue
    p = f'{OUT}/{i}.wav'
    sf.write(p, wav.squeeze(0).float().cpu().numpy(), tts.sr)
    try:
        hyp = hear(p)
        c = cer(r['teacher'], hyp)
    except Exception as e:
        print('CER FAIL', i, type(e).__name__, str(e)[:100], flush=True)
        continue
    rows.append(dict(i=i, cat=r['category'], cer=round(c, 4), ref=r['teacher'][:110], hyp=hyp[:110]))
    print(i, r['category'], 'CER', round(c, 3), '|', hyp[:80], flush=True)

elapsed = time.time() - t
bycat = {}
for r in rows:
    bycat.setdefault(r['cat'], []).append(r['cer'])
res = dict(n=len(rows), elapsed_s=round(elapsed, 1),
           s_per_clip=round(elapsed / max(1, len(rows)), 2),
           clips_per_hour=round(3600 * len(rows) / max(1, elapsed)),
           gpu_hours_3400=round(3400 * elapsed / max(1, len(rows)) / 3600, 2),
           cer_median=round(float(np.median([r['cer'] for r in rows])), 4),
           cer_mean=round(float(np.mean([r['cer'] for r in rows])), 4),
           by_category={k: dict(n=len(v), median=round(float(np.median(v)), 4)) for k, v in bycat.items()},
           rows=rows)
json.dump(res, open('/kaggle/working/sm25_report.json', 'w'), indent=1)
print('RESULT', {k: v for k, v in res.items() if k != 'rows'}, flush=True)
print('SM25 DONE', flush=True)
