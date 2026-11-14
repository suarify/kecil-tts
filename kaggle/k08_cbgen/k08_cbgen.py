# kokoro-malay-bella / continue.plan.v4.md step C1: generate the Chatterbox training clips.
# 1,140 sentences -> Chatterbox Multilingual (the approved setting) -> ChatterboxVC toward Bella.
# Output: cb_clips.pt = {id: int16 array at 24 kHz} (voice-converted), saved every 150 clips so a
# time-out never loses work. No Whisper here; ms_rows.py aligns and trims the clips later.
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()
OUT = '/kaggle/working'
GEN_CAP_MIN = float(os.environ.get('GEN_CAP_MIN', 125))
VC_CAP_MIN = float(os.environ.get('VC_CAP_MIN', 40))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q chatterbox-tts')
sh('pip uninstall -y -q torchvision || true')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DS = os.path.dirname(glob.glob('/kaggle/input/**/cb_1100.tsv', recursive=True)[0])
K01 = os.path.dirname(glob.glob('/kaggle/input/**/baseline_report.json', recursive=True)[0])
REF_MS, REF_EN = f'{K01}/refs/ref_ms.wav', f'{K01}/refs/ref_en.wav'
df = pd.read_csv(f'{DS}/cb_1100.tsv', sep='\t', dtype=str, keep_default_na=False)
print('rows', len(df), 'torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

os.makedirs('/tmp/raw', exist_ok=True)
tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')
SR = tts.sr
raw = {}
fails = 0
t_gen = time.time()
for k, r in enumerate(df.itertuples()):
    if (time.time() - t_gen) / 60 > GEN_CAP_MIN:
        print('generation cap reached at', k, flush=True)
        break
    torch.manual_seed(1234)
    try:
        wav = tts.generate(r.teacher, language_id='ms', audio_prompt_path=REF_MS, exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
    except Exception as e:
        fails += 1
        print('FAIL', r.id, repr(e)[:120], flush=True)
        continue
    a = wav.squeeze(0).float().cpu().numpy()
    p = f'/tmp/raw/{r.id}.wav'
    sf.write(p, a, SR)
    raw[r.id] = len(a) / SR
    if k % 50 == 0:
        el = time.time() - t_gen
        print(f'gen {k}/{len(df)} | {sum(raw.values()) / 60:.1f} min audio in {el / 60:.1f} min | RTF {el / max(sum(raw.values()), 1):.2f}', flush=True)
secs = sum(raw.values())
print(f'generated {len(raw)} clips, {secs / 60:.1f} min, {fails} failures, {(time.time() - t_gen) / 60:.1f} min', flush=True)
del tts
gc.collect()
torch.cuda.empty_cache()

from chatterbox.vc import ChatterboxVC  # noqa: E402

vc = ChatterboxVC.from_pretrained('cuda')
clips = {}
t_vc = time.time()
for k, i in enumerate(raw):
    if (time.time() - t_vc) / 60 > VC_CAP_MIN:
        print('vc cap reached at', k, flush=True)
        break
    try:
        w = vc.generate(audio=f'/tmp/raw/{i}.wav', target_voice_path=REF_EN)
    except Exception as e:
        print('VC FAIL', i, repr(e)[:120], flush=True)
        continue
    a = w.squeeze(0).float().cpu().numpy()
    clips[i] = torch.from_numpy((np.clip(a, -1, 1) * 32767).astype(np.int16))
    if k % 150 == 149:
        torch.save(clips, f'{OUT}/cb_clips.pt')
        print(f'vc {k + 1}/{len(raw)} saved | {(time.time() - t_vc) / 60:.1f} min', flush=True)
torch.save(clips, f'{OUT}/cb_clips.pt')
rep = dict(sentences=len(df), generated=len(raw), converted=len(clips), failures=fails, audio_min=round(secs / 60, 1),
           gen_min=round((time.time() - t_gen) / 60, 1), sr=int(vc.sr), total_min=round((time.time() - T0) / 60, 1))
json.dump(rep, open(f'{OUT}/cb_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1), flush=True)
