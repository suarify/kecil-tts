# sm23-p4-timing: Phase 4 step 2 - find the fastest Chatterbox config with CER <= 2.5%.
# Configs: A = baseline (audio_prompt_path per call), B = conditionals prepared once,
# C = conditionals once + top_p 0.9. Reports s/clip, clips/hour, projected GPU-h for 3,400, and CER.
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

T0 = time.time()
OUT = '/kaggle/working'
W = f'{OUT}/p4'
os.makedirs(W, exist_ok=True)


def sh(cmd, name=None):
    t = time.time()
    print(f'=== [{time.time() - T0:6.0f}s] $ {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    if p.stdout.strip():
        print(p.stdout[-1500:], flush=True)
    if p.returncode:
        print(p.stderr[-2500:], flush=True)
        json.dump(dict(failed=name, tail=p.stderr[-3000:]), open(f'{OUT}/sm23_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED {cmd}')
    if name:
        print(f'=== stage {name}: {time.time() - t:.1f}s', flush=True)
    return p.stdout


print('python', sys.version.split()[0], flush=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
for d in sorted(glob.glob('/kaggle/input/*')):
    print('input:', d, len(os.listdir(d)), 'entries', flush=True)

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
print('texts', len(plan), '| ref', REF, flush=True)
assert len(plan) >= 45

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402
import soundfile as sf  # noqa: E402

tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')

# ---- CER scorer (Whisper large-v3, language=malay, capped 1.0) ----
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402
import torchaudio  # noqa: E402
proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).to('cuda').eval()


def cer(ref, hyp):
    def clean(s):
        return re.sub(r'[^a-z]', '', s.lower())
    a, b = clean(ref), clean(hyp)
    prev = list(range(len(b) + 1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return min(1.0, prev[-1] / max(1, len(a)))


def hear(path):
    a, sr = sf.read(path, dtype='float32')
    w = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()[:16000 * 29]
    f = proc.feature_extractor(w, sampling_rate=16000, return_tensors='pt').input_features.to('cuda', torch.float16)
    with torch.inference_mode():
        ids = wm.generate(f, language='malay', task='transcribe', max_new_tokens=200)
    return proc.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


CONFIGS = [('A_baseline', dict(per_call_ref=True)),
           ('B_cond_once', dict(per_call_ref=False, top_p=1.0)),
           ('C_cond_top9', dict(per_call_ref=False, top_p=0.9))]
N = 15
results = {}
for name, cfg in CONFIGS:
    d = f'{W}/{name}'
    os.makedirs(d, exist_ok=True)
    if not cfg['per_call_ref']:
        tts.prepare_conditionals(REF, exaggeration=0.5)
    secs, cers = [], []
    t = time.time()
    for i, r in enumerate(plan[:N]):
        torch.manual_seed(1234)
        try:
            if cfg['per_call_ref']:
                wav = tts.generate(r['text'], language_id='ms', audio_prompt_path=REF,
                                   exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
            else:
                wav = tts.generate(r['text'], language_id='ms', exaggeration=0.5,
                                   cfg_weight=0.3, temperature=0.8, top_p=cfg['top_p'])
        except Exception as e:
            print('FAIL', name, i, type(e).__name__, str(e)[:120], flush=True)
            continue
        p = f'{d}/{i}.wav'
        sf.write(p, wav.squeeze(0).float().cpu().numpy(), tts.sr)
        secs.append(time.time() - t if i == 0 else 0)
        t = time.time()
        try:
            cers.append(cer(r['teacher'], hear(p)))
        except Exception as e:
            print('CER FAIL', name, i, type(e).__name__, str(e)[:100], flush=True)
    # recompute per-clip timing from file timestamps is messy; measure generation-only separately below
    results[name] = dict(n=len(cers), cer_median=round(float(np.median(cers)), 4) if cers else None,
                         cer_capped_mean=round(float(np.mean(cers)), 4) if cers else None,
                         pass_cer=bool(cers and np.median(cers) <= 0.025))
    print('CER', name, results[name], flush=True)

# ---- clean generation-only timing (no Whisper) ----
print('--- generation-only timing ---', flush=True)
for name, cfg in CONFIGS:
    if not cfg['per_call_ref']:
        tts.prepare_conditionals(REF, exaggeration=0.5)
    t = time.time()
    for i, r in enumerate(plan[:N]):
        torch.manual_seed(1234)
        if cfg['per_call_ref']:
            tts.generate(r['text'], language_id='ms', audio_prompt_path=REF,
                         exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
        else:
            tts.generate(r['text'], language_id='ms', exaggeration=0.5,
                         cfg_weight=0.3, temperature=0.8, top_p=cfg['top_p'])
    dt = (time.time() - t) / N
    results[name]['s_per_clip'] = round(dt, 2)
    results[name]['clips_per_hour'] = round(3600 / dt)
    results[name]['gpu_hours_3400'] = round(3400 * dt / 3600, 2)
    print('SPEED', name, dt, results[name]['gpu_hours_3400'], 'GPU-h', flush=True)

json.dump(results, open(f'{OUT}/sm23_report.json', 'w'), indent=1)
print('SM23 DONE', flush=True)
