# kokoro-malay-bella / continue.plan.md section 12.3: prosody-transplant proof (32 sentences).
# Idea: keep Kokoro's DECODER (clean sound, no Chatterbox muffling or babble) but drive it with the
# RHYTHM (word timings) and MELODY (F0) of the Chatterbox clip the owner liked.
#   F1   = teacher finalist F1 as it is (ef_dora 50% Bella, Kokoro's own durations and pitch)
#   TPd  = F1 + word durations taken from the Chatterbox clip (Whisper word timestamps)
#   TPf  = TPd + pitch contour (torchcrepe) taken from the Chatterbox clip
#   CB   = the Chatterbox voice-converted clip (reference, from kmb-ref)
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import types
import zipfile

T0 = time.time()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"')
sh('pip install -q loguru torchcrepe')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _stub(name):
    m = types.ModuleType(name)

    def _ga(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return type(attr, (), {})
    m.__getattr__ = _ga
    return m


sys.modules['misaki'] = _stub('misaki')
for sub in ('en', 'espeak'):
    sys.modules['misaki.' + sub] = _stub('misaki.' + sub)
    setattr(sys.modules['misaki'], sub, sys.modules['misaki.' + sub])

from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
REF = os.path.dirname(glob.glob('/kaggle/input/**/ref_32.tsv', recursive=True)[0])
sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

OUT = '/kaggle/working'
SR = 24000
FPS = 40.0                      # Kokoro duration frames per second (600 samples each)
DEV = 'cuda'
PUNCT = set(',.!?;:…—')
m = KModel(repo_id='hexgrad/Kokoro-82M').to(DEV).eval()
VOCAB = m.vocab


def voice(name):
    return torch.load(hf_hub_download('hexgrad/Kokoro-82M', f'voices/{name}.pt'), weights_only=True)


BELLA, DORA = voice('af_bella'), voice('ef_dora')
PACK = torch.cat([0.5 * BELLA[..., :128] + 0.5 * DORA[..., :128], DORA[..., 128:]], -1)   # F1 = ef_dora a0.5 p0.0
G = G2P(stress=False, r='r', diph='glide', final_a='a')


@torch.no_grad()
def internals(ps):
    ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]]).to(DEV)
    T = ids.shape[1]
    r = PACK[min(max(0, T - 3), PACK.shape[0] - 1)].to(DEV)
    L = torch.full((1,), T, device=DEV)
    mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    bert = m.bert(ids, attention_mask=(~mask).int())
    d_en = m.bert_encoder(bert).transpose(-1, -2)
    s = r[:, 128:]
    d = m.predictor.text_encoder(d_en, s, L, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0)
    t_en = m.text_encoder(ids, L, mask)
    return dict(ids=ids, T=T, r=r, s=s, d=d, t_en=t_en, dur=dur)


@torch.no_grad()
def render(I, frames, f0=None):
    """frames: int tensor [T] durations. f0: optional numpy array (Hz, length 2*sum(frames)) replacing Kokoro's pitch."""
    frames = frames.long().to(DEV)
    idx = torch.repeat_interleave(torch.arange(I['T'], device=DEV), frames)
    aln = torch.zeros(I['T'], idx.shape[0], device=DEV)
    aln[idx, torch.arange(idx.shape[0], device=DEV)] = 1
    aln = aln[None]
    en = I['d'].transpose(-1, -2) @ aln
    F0, N = m.predictor.F0Ntrain(en, I['s'])
    kf0 = F0.clone()
    if f0 is not None:
        F0 = torch.from_numpy(f0.astype(np.float32))[None].to(DEV)
    asr = I['t_en'] @ aln
    torch.manual_seed(0)
    a = m.decoder(asr, F0, N, I['r'][:, :128]).squeeze()
    return a.float().cpu().numpy(), kf0.squeeze(0).cpu().numpy()


# ---------------- models for timestamps, scoring ----------------
from transformers import pipeline  # noqa: E402
import torchcrepe  # noqa: E402

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)
FE, ENC = asr.feature_extractor, asr.model.get_encoder()
GK = {'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200}


def to16(a, sr):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32))
    if t.ndim > 1:
        t = t.mean(-1)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.numpy()


def load16(p):
    a, sr = sf.read(p, dtype='float32')
    return to16(a, sr)


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


def words_of(w16):
    o = asr({'raw': w16[:16000 * 29], 'sampling_rate': 16000}, batch_size=1, return_timestamps='word', generate_kwargs=GK)
    out = []
    for c in o.get('chunks', []):
        t = re.sub(r'[^\w]', '', c['text'])
        if t and c['timestamp'][0] is not None and c['timestamp'][1] is not None:
            out.append((t, float(c['timestamp'][0]), float(c['timestamp'][1])))
    return out, o['text']


def transcribe(w16):
    return asr({'raw': w16[:16000 * 29], 'sampling_rate': 16000}, batch_size=1, generate_kwargs=GK)['text'].strip()


@torch.no_grad()
def enc_feat(w16):
    f = FE([w16[:16000 * 29]], sampling_rate=16000, return_tensors='pt').input_features
    h = ENC(f.to(DEV, torch.float16)).last_hidden_state.float()[0]
    n = min(1500, math.ceil(min(len(w16), 16000 * 29) / 320))
    return torch.nn.functional.normalize(h[:n:2], dim=-1).cpu().numpy()


def dtw_cos(a, b):
    c = 1.0 - a @ b.T
    n, k = c.shape
    D = np.full((n + 1, k + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], c[i - 1]
        for j in range(1, k + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return float(D[n, k] / (n + k))


def pitch_track(w16):
    a = torch.from_numpy(w16)[None]
    pitch, per = torchcrepe.predict(a, 16000, 160, 60, 450, 'full', decoder=torchcrepe.decode.viterbi,
                                    return_periodicity=True, batch_size=1024, device=DEV)
    per = torchcrepe.filter.median(per, 3)
    pitch = torchcrepe.threshold.At(0.21)(pitch, per)
    return pitch[0].cpu().numpy()                      # 10 ms hop, NaN = unvoiced


def f0_curve(track, n):
    """Resample a 10 ms pitch track (NaN = unvoiced) to Kokoro's 12.5 ms grid of length n."""
    t = (np.arange(n) + 0.5) * 0.0125
    tc = np.arange(len(track)) * 0.01
    voiced = ~np.isnan(track)
    if voiced.sum() < 5:
        return np.zeros(n, dtype=np.float32)
    v = np.interp(t, tc, voiced.astype(float)) > 0.5
    lf = np.interp(t, tc[voiced], np.log(track[voiced]))
    return np.where(v, np.exp(lf), 0.0).astype(np.float32)


def segments_of(ps):
    """Word segments of the phoneme string: letter-token range, punctuation tokens, following space token."""
    tok = []                                           # token index (1-based) for each char, or None
    n = 0
    for ch in ps:
        if ch in VOCAB:
            n += 1
            tok.append(n)
        else:
            tok.append(None)
    segs, pos = [], 0
    for part in ps.split(' '):
        c0, c1 = pos, pos + len(part)
        pos = c1 + 1
        if not any(ch.isalpha() for ch in part):
            continue
        letters = [tok[i] for i in range(c0, c1) if ps[i] not in PUNCT and tok[i] is not None]
        puncts = [tok[i] for i in range(c0, c1) if ps[i] in PUNCT and tok[i] is not None]
        sp = tok[c1] if c1 < len(ps) and ps[c1] == ' ' else None
        if letters:
            segs.append(dict(a=letters[0], b=letters[-1] + 1, punct=puncts, space=sp))
    return segs


def new_durations(I, segs, words):
    base = torch.clamp(torch.round(I['dur']), min=1).cpu().numpy().astype(int)
    kd = I['dur'].cpu().numpy()
    new = base.copy()
    new[0] = max(1, int(round(words[0][1] * FPS)))
    for j, (sg, (_, st, en)) in enumerate(zip(segs, words)):
        a, b = sg['a'], sg['b']
        scale = float(np.clip((en - st) * FPS / max(kd[a:b].sum(), 1e-3), 0.4, 3.0))
        new[a:b] = np.maximum(1, np.round(kd[a:b] * scale)).astype(int)
        if j + 1 < len(words):
            gap = max(1, int(round((words[j + 1][1] - en) * FPS)))
            pun = sum(new[p] for p in sg['punct'])
            if sg['space'] is not None:
                new[sg['space']] = max(1, gap - pun)
    return new


# ---------------- run ----------------
ref32 = pd.read_csv(os.path.join(REF, 'ref_32.tsv'), sep='\t', dtype=str, keep_default_na=False)
rows, clips = [], {}
for r in ref32.itertuples():
    p2 = glob.glob(os.path.join(REF, '**', 'T2', f'{r.id}.wav'), recursive=True)
    pv = glob.glob(os.path.join(REF, '**', 'T2v', f'{r.id}.wav'), recursive=True)
    if not p2:
        continue
    cb16 = load16(p2[0])
    norm = normalize(r.text)
    ps = G(norm)
    segs = segments_of(ps)
    words, _ = words_of(cb16)
    row = dict(id=r.id, category=r.category, norm=norm, n_words_ps=len(segs), n_words_asr=len(words))
    if len(segs) != len(words) or len(words) == 0:
        row['skipped'] = 'word count mismatch'
        rows.append(row)
        print('skip', r.id, len(segs), len(words), flush=True)
        continue
    I = internals(ps)
    base_frames = torch.clamp(torch.round(I['dur']), min=1)
    a_f1, kf0 = render(I, base_frames)
    new = new_durations(I, segs, words)
    a_d, _ = render(I, torch.from_numpy(new))
    track = pitch_track(cb16)
    f0c = f0_curve(track, 2 * int(new.sum()))
    kv = kf0[kf0 > 50]
    cv = f0c[f0c > 50]
    ratio = float(np.clip(np.median(kv) / np.median(cv), 0.6, 1.6)) if len(kv) > 5 and len(cv) > 5 else 1.0
    a_f, _ = render(I, torch.from_numpy(new), f0c * ratio)
    row.update(f0_ratio=ratio, sec_f1=len(a_f1) / SR, sec_cb=len(cb16) / 16000, sec_tp=len(a_f) / SR)
    for name, a in (('F1', a_f1), ('TPd', a_d), ('TPf', a_f)):
        clips[(name, r.id)] = a
    rows.append(row)
    print('done', r.id, round(time.time() - T0), 's', flush=True)

# ---------------- scoring ----------------
df = pd.DataFrame(rows)
ok = df[df.get('skipped').isna()] if 'skipped' in df else df
res = {}
for r in ok.itertuples():
    p2 = glob.glob(os.path.join(REF, '**', 'T2', f'{r.id}.wav'), recursive=True)[0]
    pv = glob.glob(os.path.join(REF, '**', 'T2v', f'{r.id}.wav'), recursive=True)
    ref_feat = enc_feat(load16(p2))
    for name in ('F1', 'TPd', 'TPf'):
        w16 = to16(clips[(name, r.id)], SR)
        h = transcribe(w16)
        res[(name, r.id)] = dict(cer=lev(clean(r.norm), clean(h)) / max(1, len(clean(r.norm))),
                                 acc=dtw_cos(enc_feat(w16), ref_feat), hyp=h)
    if pv:
        res[('CB', r.id)] = dict(cer=float('nan'), acc=dtw_cos(enc_feat(load16(pv[0])), ref_feat), hyp='')
sc = pd.DataFrame([dict(config=k[0], id=k[1], **v) for k, v in res.items()])
sc.to_csv(f'{OUT}/transplant_scores.csv', index=False)
df.to_csv(f'{OUT}/transplant_rows.csv', index=False)
with zipfile.ZipFile(f'{OUT}/clips_transplant.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for (name, i), a in clips.items():
        p = f'/tmp/{name}_{i}.wav'
        sf.write(p, a, SR)
        z.write(p, f'{name}/{i}.wav')
summary = sc.groupby('config').agg(cer=('cer', 'mean'), acc=('acc', 'mean')).round(4)
print(summary.to_string(), flush=True)
rep = dict(n_total=len(df), n_used=len(ok), n_skipped=int(len(df) - len(ok)),
           summary=json.loads(summary.to_json(orient='index')), runtime_s=round(time.time() - T0))
json.dump(rep, open(f'{OUT}/transplant_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1), flush=True)
