# kokoro-malay-bella / Phase 4 pilot: 150 prompts through the chosen teacher (T2).
# T2 = Chatterbox Multilingual, ref = Kokoro-Bella Malay (ref_ms.wav), cfg_weight=0.3.
# Fixes from the listening test:
#   1. English words are respelled Malay-style for the teacher only (confirm -> konfem).
#   2. Trailing babble is cut at the end of the last reference word (Whisper word timestamps).
#   3. A clip still failing after trimming is regenerated once with another seed.
# Also makes T2v = T2 pushed toward true Bella with ChatterboxVC, and measures speed.
import glob
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q chatterbox-tts')
sh('pip install -q speechbrain')
sh('pip uninstall -y -q torchvision || true')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
K01 = os.path.dirname(glob.glob('/kaggle/input/**/baseline_report.json', recursive=True)[0])
sys.path.insert(0, DS)
from ms_norm import normalize  # noqa: E402
print('DS', DS, 'K01', K01, flush=True)

OUT = '/kaggle/working'
DEV = 'cuda'
REF_MS = f'{K01}/refs/ref_ms.wav'
REF_EN = f'{K01}/refs/ref_en.wav'
RESPELL = json.load(open(f'{DS}/teacher_respell.json', encoding='utf-8'))


def respell(text):
    def rep(m):
        w = m.group(0)
        r = RESPELL.get(w.lower())
        if r is None:
            return w
        return r.capitalize() if w[0].isupper() else r
    return re.sub(r'[A-Za-z]+', rep, text)


pilot = pd.read_csv(f'{DS}/pilot_150.tsv', sep='\t', dtype=str, keep_default_na=False)
pilot['norm'] = [normalize(t) for t in pilot.text]
pilot['teacher'] = [respell(n) for n in pilot.norm]
for d in ('raw', 'T2', 'T2v'):
    os.makedirs(f'{OUT}/pilot/{d}', exist_ok=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402


import gc  # noqa: E402
import shutil  # noqa: E402

PREV = glob.glob('/kaggle/input/**/raw/p000_s1234.wav', recursive=True)
if PREV:
    for f in glob.glob(os.path.join(os.path.dirname(PREV[0]), '*.wav')):
        shutil.copy(f, f'{OUT}/pilot/raw/')
    print('reused', len(os.listdir(f'{OUT}/pilot/raw')), 'raw clips', flush=True)


def generate(ids, seed, temp):
    ids = [i for i in ids if not os.path.exists(f'{OUT}/pilot/raw/{i}_s{seed}.wav')]
    if not ids:
        return 24000, 1.0, 0.0
    tts = ChatterboxMultilingualTTS.from_pretrained(device=DEV)
    t = time.time()
    secs = 0.0
    for iid in ids:
        r = pilot.set_index('id').loc[iid]
        torch.manual_seed(seed)
        try:
            wav = tts.generate(r.teacher, language_id='ms', audio_prompt_path=REF_MS,
                               exaggeration=0.5, cfg_weight=0.3, temperature=temp)
        except Exception as e:
            print('FAIL', iid, e, flush=True)
            continue
        a = wav.squeeze(0).float().cpu().numpy()
        sf.write(f'{OUT}/pilot/raw/{iid}_s{seed}.wav', a, tts.sr)
        secs += len(a) / tts.sr
    wall = time.time() - t
    sr = tts.sr
    del tts
    gc.collect()
    torch.cuda.empty_cache()
    return sr, wall, secs


SR, wall1, secs1 = generate(pilot.id.tolist(), 1234, 0.8)
GEN_RTF = 1.42 if secs1 == 0 else round(wall1 / secs1, 3)  # 1.42 measured in v1
print(f'gen pass 1: {wall1:.0f}s wall for {secs1:.0f}s audio, RTF {wall1 / max(secs1, 1):.2f}', flush=True)

# ---------------- Whisper with word timestamps ----------------
from transformers import pipeline  # noqa: E402

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)


def load16(p):
    a, sr = sf.read(p, dtype='float32')
    if a.ndim > 1:
        a = a.mean(1)
    t = torch.from_numpy(a)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.numpy()


def transcribe(paths, words=True):
    batch = [{'raw': load16(p)[:16000 * 29], 'sampling_rate': 16000} for p in paths]
    kw = dict(batch_size=1 if words else 4, generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})
    if words:
        kw['return_timestamps'] = 'word'
    return asr(batch, **kw)


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


def cer(refs, hyp):
    h = clean(hyp)
    return min(lev(clean(r), h) / max(1, len(clean(r))) for r in refs)


def best_prefix(refs, chunks):
    """Shortest prefix of Whisper words that best matches the reference."""
    words = [c['text'] for c in chunks]
    best = (cer(refs, ' '.join(words)), len(words))
    for k in range(1, len(words)):
        c = cer(refs, ' '.join(words[:k]))
        if c < best[0] - 1e-9:
            best = (c, k)
    return best


def trim(src, dst, chunks, k):
    a, sr = sf.read(src, dtype='float32')
    n = len(a)
    end = n
    if chunks and k >= 1:
        e = chunks[k - 1]['timestamp'][1]
        if e is not None:
            cut = e + 0.25
            if k < len(chunks) and chunks[k]['timestamp'][0] is not None:
                cut = min(cut, chunks[k]['timestamp'][0] + 0.02)
            end = min(n, int(cut * sr))
    a = a[:end].copy()
    # trailing silence: keep 0.12 s after last frame above -40 dB of peak
    fr = int(0.02 * sr)
    if len(a) > fr:
        peak = np.abs(a).max() + 1e-9
        rms = np.sqrt(np.convolve(a ** 2, np.ones(fr) / fr, mode='same'))
        loud = np.where(rms > peak * 0.01)[0]
        if len(loud):
            a = a[:min(len(a), loud[-1] + int(0.12 * sr))]
    f = min(len(a), int(0.03 * sr))
    a[len(a) - f:] *= np.linspace(1, 0, f)
    sf.write(dst, a, sr)
    return n / sr, len(a) / sr


def score_and_trim(ids, seed):
    paths = [f'{OUT}/pilot/raw/{i}_s{seed}.wav' for i in ids]
    keep = [(i, p) for i, p in zip(ids, paths) if os.path.exists(p)]
    outs = transcribe([p for _, p in keep])
    res = {}
    P = pilot.set_index('id')
    for (iid, p), o in zip(keep, outs):
        refs = [P.loc[iid, 'norm'], P.loc[iid, 'teacher']]
        chunks = o.get('chunks') or []
        raw_cer = cer(refs, o['text'])
        c, k = best_prefix(refs, chunks)
        dst = f'{OUT}/pilot/T2/{iid}_s{seed}.wav'
        before, after = trim(p, dst, chunks, k)
        res[iid] = dict(seed=seed, raw_hyp=o['text'].strip(), raw_cer=raw_cer,
                        hyp=' '.join(ch['text'] for ch in chunks[:k]).strip(), cer=c,
                        sec_raw=before, sec=after, trimmed=round(before - after, 2), path=dst)
    return res


R = score_and_trim(pilot.id.tolist(), 1234)
bad = [i for i, r in R.items() if r['cer'] > 0.10] + [i for i in pilot.id if i not in R]
print('after trim, failing:', len(bad), flush=True)

if bad:
    del asr
    torch.cuda.empty_cache()
    _, wall2, secs2 = generate(bad, 77, 0.7)
    asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
                   torch_dtype=torch.float16, device=0)
    R2 = score_and_trim(bad, 77)
    for i, r in R2.items():
        if i not in R or r['cer'] < R[i]['cer']:
            r['retried'] = True
            R[i] = r
    print('retry pass:', round(wall2), 's', flush=True)

for i, r in R.items():
    final = f'{OUT}/pilot/T2/{i}.wav'
    os.replace(r['path'], final)
    r['path'] = final
for p in glob.glob(f'{OUT}/pilot/T2/*_s*.wav'):
    os.remove(p)

# ---------------- T2v: voice-convert toward true Bella ----------------
del asr
torch.cuda.empty_cache()
from chatterbox.vc import ChatterboxVC  # noqa: E402

vc = ChatterboxVC.from_pretrained(DEV)
t = time.time()
for i, r in R.items():
    wav = vc.generate(audio=r['path'], target_voice_path=REF_EN)
    sf.write(f'{OUT}/pilot/T2v/{i}.wav', wav.squeeze(0).float().cpu().numpy(), vc.sr)
vc_wall = time.time() - t
del vc
torch.cuda.empty_cache()

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)
ids = list(R)
outs = transcribe([f'{OUT}/pilot/T2v/{i}.wav' for i in ids], words=False)
P = pilot.set_index('id')
for i, o in zip(ids, outs):
    R[i]['v_hyp'] = o['text'].strip()
    R[i]['v_cer'] = cer([P.loc[i, 'norm'], P.loc[i, 'teacher']], o['text'])
del asr
torch.cuda.empty_cache()

# ---------------- speaker similarity ----------------
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

enc = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir='/tmp/ecapa',
                                     run_opts={'device': 'cuda:0'})


def emb(p):
    with torch.no_grad():
        e = enc.encode_batch(torch.from_numpy(load16(p)).unsqueeze(0).to('cuda:0')).squeeze()
    return torch.nn.functional.normalize(e, dim=0).cpu()


bella = sorted(glob.glob(f'{K01}/refs/bella_en/*.wav'))
centroid = torch.nn.functional.normalize(torch.stack([emb(p) for p in bella]).mean(0), dim=0)
for i in ids:
    R[i]['sim'] = float(emb(R[i]['path']) @ centroid)
    R[i]['v_sim'] = float(emb(f'{OUT}/pilot/T2v/{i}.wav') @ centroid)

df = pilot.set_index('id').join(pd.DataFrame.from_dict(R, orient='index'))
df['rel'] = [f'pilot/T2/{i}.wav' for i in df.index]
df['rel_v'] = [f'pilot/T2v/{i}.wav' for i in df.index]
df['rel_raw'] = [f"pilot/raw/{i}_s{int(s)}.wav" if s == s else '' for i, s in zip(df.index, df.seed)]
df.drop(columns=['path']).to_csv(f'{OUT}/pilot_scores.csv')

good = df[df.cer <= 0.10]
report = {
    'n': len(df), 'kept_cer_le_0.10': int(len(good)), 'kept_hours': round(good.sec.sum() / 3600, 3),
    'cer_mean_trimmed': round(df.cer.mean(), 4), 'cer_mean_raw': round(df.raw_cer.mean(), 4),
    'clips_trimmed_over_0.3s': int((df.trimmed > 0.3).sum()), 'retried': int(df.get('retried', pd.Series()).fillna(False).sum()),
    'sim_T2': round(df.sim.mean(), 3), 'sim_T2v': round(df.v_sim.mean(), 3),
    'cer_T2v': round(df.v_cer.mean(), 4),
    'by_category_cer': df.groupby('category').cer.mean().round(4).to_dict(),
    'gen_rtf': GEN_RTF, 'vc_rtf': round(vc_wall / max(df.sec.sum(), 1), 3),
    'est_gpu_hours_for_11h_audio': round(11 * GEN_RTF * 1.15, 1),
    'runtime_s': round(time.time() - T0),
}
json.dump(report, open(f'{OUT}/pilot_report.json', 'w'), indent=1)
print(json.dumps(report, indent=1), flush=True)
