# kokoro-malay-bella / Phase 3: teacher bake-off (Gate 2).
# Generates Malay audio in the Bella voice with Chatterbox Multilingual three ways,
# then scores every candidate on Whisper CER, ECAPA similarity to true Bella, and roll-off.
#   T1  Chatterbox TTS, ref = Bella English (ref_en.wav), cfg_weight=0
#   T2  Chatterbox TTS, ref = Kokoro-Bella Malay baseline (ref_ms.wav), cfg_weight=0.3
#   T3  Real Malay speech (FLEURS ms_my, female) converted to Bella with ChatterboxVC
#   B0  Kokoro baseline best variant (from notebook 01), for comparison
import glob
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q chatterbox-tts')
sh('pip install -q speechbrain')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

DS = '/kaggle/input/kokoro-malay-g2p'
K01 = '/kaggle/input/kmb-01-baseline'
sys.path.insert(0, DS)
from ms_norm import normalize  # noqa: E402

OUT = '/kaggle/working'
DEV = 'cuda'
print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)

rep01 = json.load(open(f'{K01}/baseline_report.json'))
BEST = rep01['best_variant']
REF_EN = f'{K01}/refs/ref_en.wav'
REF_MS = f'{K01}/refs/ref_ms.wav'


def read_tsv(name):
    return pd.read_csv(os.path.join(DS, name), sep='\t', dtype=str, keep_default_na=False)


test = read_tsv('test_40.tsv')
etest = read_tsv('e_test.tsv')
items = [(r.id, r.category, normalize(r.text)) for r in test.itertuples()] + \
        [('e_' + r.word, 'etest', normalize(r.text)) for r in etest.itertuples()]

rows = []

# ---------------- T1 / T2: Chatterbox multilingual TTS ----------------
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

tts = ChatterboxMultilingualTTS.from_pretrained(device=DEV)
SR_CB = tts.sr
for tname, ref, cfg in (('T1', REF_EN, 0.0), ('T2', REF_MS, 0.3)):
    os.makedirs(f'{OUT}/teacher/{tname}', exist_ok=True)
    for iid, cat, text in items:
        torch.manual_seed(1234)
        try:
            wav = tts.generate(text, language_id='ms', audio_prompt_path=ref,
                               exaggeration=0.5, cfg_weight=cfg, temperature=0.8)
            a = wav.squeeze(0).float().cpu().numpy()
        except Exception as e:  # keep going; record the failure
            print('FAIL', tname, iid, e, flush=True)
            continue
        path = f'{OUT}/teacher/{tname}/{iid}.wav'
        sf.write(path, a, SR_CB)
        rows.append(dict(cand=tname, id=iid, category=cat, ref_text=text, path=path))
    print(tname, 'done', round(time.time() - T0), 's', flush=True)
del tts
torch.cuda.empty_cache()

# ---------------- T3: FLEURS ms_my female speech -> Bella via ChatterboxVC ----------------
try:
    import tarfile
    from huggingface_hub import hf_hub_download
    tsv = hf_hub_download('google/fleurs', 'data/ms_my/test.tsv', repo_type='dataset')
    tar = hf_hub_download('google/fleurs', 'data/ms_my/audio/test.tar.gz', repo_type='dataset')
    cols = ['id', 'file', 'raw', 'transcription', 'phonemes', 'num_samples', 'gender']
    fl = pd.read_csv(tsv, sep='\t', names=cols, dtype=str, quoting=3, keep_default_na=False)
    fl['sec'] = fl.num_samples.astype(int) / 16000
    fl = fl[(fl.gender.str.upper() == 'FEMALE') & fl.sec.between(4, 12)].drop_duplicates('id').head(20)
    want = set(fl.file)
    os.makedirs('/tmp/fleurs', exist_ok=True)
    with tarfile.open(tar) as tf:
        for m in tf.getmembers():
            if os.path.basename(m.name) in want:
                m.name = os.path.basename(m.name)
                tf.extract(m, '/tmp/fleurs')
    from chatterbox.vc import ChatterboxVC
    vc = ChatterboxVC.from_pretrained(DEV)
    os.makedirs(f'{OUT}/teacher/T3', exist_ok=True)
    os.makedirs(f'{OUT}/teacher/T3_src', exist_ok=True)
    for r in fl.itertuples():
        src = f'/tmp/fleurs/{r.file}'
        if not os.path.exists(src):
            continue
        wav = vc.generate(audio=src, target_voice_path=REF_EN)
        iid = 'fl_' + r.id
        path = f'{OUT}/teacher/T3/{iid}.wav'
        sf.write(path, wav.squeeze(0).float().cpu().numpy(), vc.sr)
        a, sr = sf.read(src, dtype='float32')
        sf.write(f'{OUT}/teacher/T3_src/{iid}.wav', a, sr)
        rows.append(dict(cand='T3', id=iid, category='fleurs', ref_text=normalize(r.raw), path=path))
        rows.append(dict(cand='T3src', id=iid, category='fleurs', ref_text=normalize(r.raw),
                         path=f'{OUT}/teacher/T3_src/{iid}.wav'))
    del vc
    torch.cuda.empty_cache()
    print('T3 done', round(time.time() - T0), 's', flush=True)
except Exception as e:
    print('T3 FAILED', repr(e), flush=True)

# Kokoro baseline (best variant) for comparison
for iid, cat, text in items:
    p = f'{K01}/baseline/{BEST}/{iid}.wav'
    if os.path.exists(p):
        rows.append(dict(cand='B0', id=iid, category=cat, ref_text=text, path=p))

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/teacher_index.csv', index=False)
print('candidates', df.cand.value_counts().to_dict(), flush=True)

# ---------------- Scoring ----------------
def load16(p):
    a, sr = sf.read(p, dtype='float32')
    if a.ndim > 1:
        a = a.mean(1)
    t = torch.from_numpy(a)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t


def clean(s):
    return re.sub(r'[^a-z]', '', normalize(s).lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


try:
    from transformers import pipeline
    asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
                   torch_dtype=torch.float16, device=0)
    batch = [{'raw': load16(p).numpy(), 'sampling_rate': 16000} for p in df.path]
    out = asr(batch, batch_size=16, generate_kwargs={'language': 'malay', 'task': 'transcribe'})
    df['hyp'] = [o['text'].strip() for o in out]
    df['cer'] = [lev(clean(r), clean(h)) / max(1, len(clean(r))) for r, h in zip(df.ref_text, df.hyp)]
    del asr
    torch.cuda.empty_cache()
except Exception as e:
    print('ASR FAILED', repr(e), flush=True)

try:
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb',
                                         savedir='/tmp/ecapa', run_opts={'device': DEV})

    def emb(p):
        with torch.no_grad():
            e = enc.encode_batch(load16(p).unsqueeze(0).to(DEV)).squeeze()
        return torch.nn.functional.normalize(e, dim=0).cpu()

    bella = sorted(glob.glob(f'{K01}/refs/bella_en/*.wav'))
    sarah = sorted(glob.glob(f'{K01}/refs/sarah_en/*.wav'))
    E_b = [emb(p) for p in bella]
    centroid = torch.nn.functional.normalize(torch.stack(E_b[6:] or E_b).mean(0), dim=0)
    df['sim'] = [float(emb(p) @ centroid) for p in df.path]
    calib = {
        'bella_hi': float(np.mean([float(e @ centroid) for e in E_b[:6]])),
        'sarah_lo': float(np.mean([float(emb(p) @ centroid) for p in sarah])),
    }
except Exception as e:
    print('ECAPA FAILED', repr(e), flush=True)
    calib = {}

try:
    import librosa

    def rolloff(p):
        a, sr = librosa.load(p, sr=24000)
        return float(np.median(librosa.feature.spectral_rolloff(y=a, sr=sr, roll_percent=0.85)))
    df['rolloff_hz'] = [rolloff(p) for p in df.path]
except Exception as e:
    print('ROLLOFF FAILED', repr(e), flush=True)

df['seconds'] = [sf.info(p).duration for p in df.path]
df['rel'] = [os.path.relpath(p, OUT) if p.startswith(OUT) else 'k01/' + os.path.relpath(p, K01) for p in df.path]
df.drop(columns=['path']).to_csv(f'{OUT}/teacher_scores.csv', index=False)

agg = {c: (c, 'mean') for c in ('cer', 'sim', 'rolloff_hz', 'seconds') if c in df}
summary = df[df.category != 'etest'].groupby('cand').agg(**agg).round(4)
print(summary.to_string())
report = {'best_kokoro_variant': BEST, 'calibration': calib,
          'summary': json.loads(summary.to_json(orient='index')),
          'runtime_s': round(time.time() - T0)}
json.dump(report, open(f'{OUT}/teacher_report.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
