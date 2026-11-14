"""CPU check for the Chatterbox-direct route: can inputs measured from a Chatterbox clip drive a
Paradee-style decoder? No training. If the pretrained 8M decoder (Paradee v1.3, af_heart) speaks the
sentence intelligibly from (Kokoro text features + timings, pitch and loudness measured on the
Chatterbox clip), then the measured inputs are in the right format and scale to train on.

Also calibrates the two measured signals against Kokoro's own values on Kokoro audio:
  N  = log of the L2 norm of the mel spectrum (StyleTTS2 definition), 80 values per second
  F0 = pitch in Hz, 0 where unvoiced, 80 values per second

Run:  .venv/Scripts/python scripts/direct_local.py <paradee/training/scripts> [n_clips]
"""
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSCRIPTS = sys.argv[1]
NCLIP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
stubs = os.path.join(ROOT, '.stubs', 'misaki')
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    with open(os.path.join(stubs, name + '.py'), 'w') as f:
        f.write('def __getattr__(name):\n    return type(name, (), {})\n')
with open(os.path.join(stubs, '__init__.py'), 'w') as f:
    f.write('from . import en, espeak\n')
sys.path.insert(0, os.path.dirname(stubs))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
sys.path.insert(0, PSCRIPTS)

import librosa  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402
from ms_g2p import G2P, TOKEN_RE, PUNCT, strip_accents  # noqa: E402
from ms_norm import normalize  # noqa: E402
from vhead import VDecoder  # noqa: E402

T0 = time.time()
SR, FPS = 24000, 40.0
OUT = os.path.join(ROOT, 'runs', 'k06_direct')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() - 1))
m = KModel(repo_id='hexgrad/Kokoro-82M').eval()
VOCAB = m.vocab
G = G2P(stress=False, r='r', diph='glide', final_a='a')
BELLA = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_bella.pt'), weights_only=True)
DORA = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/ef_dora.pt'), weights_only=True)
PACK = torch.cat([0.5 * BELLA[..., :128] + 0.5 * DORA[..., :128], DORA[..., 128:]], -1)

st = dict(torch.load(hf_hub_download('sahilmahendrakar/Paradee-8M-v1.3', 'pytorch/decoder.pt'), map_location='cpu')['model'])
st['style'] = st['style'][0:1].clone()
VD = VDecoder(head='v2', dim=160, blocks=4).eval()
VD.load_state_dict(st)

bundle = torchaudio.pipelines.MMS_FA
fa = bundle.get_model().eval()
LAB = {c: i for i, c in enumerate(bundle.get_labels())}
# StyleTTS2's mel (it leaves torchaudio's default sample_rate=16000 in place, so we do too)
MEL = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def ctc_viterbi(lp, tgt, blank=0):
    T, L = lp.shape[0], len(tgt)
    S = 2 * L + 1
    ext = np.full(S, blank)
    ext[1::2] = tgt
    NEG = -1e9
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = lp[0, blank]
    dp[0, 1] = lp[0, ext[1]]
    allow2 = np.zeros(S, bool)
    allow2[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(allow2, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stk = np.stack([prev, c1, c2])
        k = stk.argmax(0)
        dp[t] = stk[k, np.arange(S)] + lp[t, ext]
        bp[t] = k
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def word_times(w16, words):
    with torch.no_grad():
        em = torch.log_softmax(fa(torch.from_numpy(w16)[None])[0][0], -1).numpy()
    tgt, first, last = [], [], []
    for w in words:
        first.append(len(tgt))
        tgt += [LAB[c] for c in w if c in LAB]
        last.append(len(tgt) - 1)
    if any(l < f for f, l in zip(first, last)) or len(tgt) * 2 + 1 > em.shape[0]:
        return None
    path = ctc_viterbi(em, np.array(tgt))
    sec = len(w16) / 16000 / em.shape[0]
    out = []
    for f, l in zip(first, last):
        idx = np.where((path >= 2 * f + 1) & (path <= 2 * l + 1))[0]
        if len(idx) == 0:
            return None
        out.append((idx[0] * sec, (idx[-1] + 1) * sec))
    return out


def segments_of(ps):
    tok, n = [], 0
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


@torch.no_grad()
def kokoro_text(ps):
    ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]])
    T = ids.shape[1]
    r = PACK[min(max(0, T - 3), PACK.shape[0] - 1)]
    L = torch.full((1,), T)
    mask = torch.zeros(1, T, dtype=torch.bool)
    d = m.predictor.text_encoder(m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2), r[:, 128:], L, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0)
    return dict(T=T, r=r, d=d, dur=dur, t_en=m.text_encoder(ids, L, mask))


def align_matrix(T, frames):
    frames = torch.as_tensor(frames).long()
    idx = torch.repeat_interleave(torch.arange(T), frames)
    aln = torch.zeros(T, idx.shape[0])
    aln[idx, torch.arange(idx.shape[0])] = 1
    return aln[None]


def frames_from_clip(I, segs, times, total_sec):
    """Token durations (25 ms frames) that add up to the clip length, from word times."""
    kd = I['dur'].numpy()
    new = np.maximum(1, np.round(kd)).astype(int)
    new[0] = max(1, int(round(times[0][0] * FPS)))
    for j, (sg, (st_, en)) in enumerate(zip(segs, times)):
        a, b = sg['a'], sg['b']
        pause = 0.0
        if j + 1 < len(times):
            gap = times[j + 1][0] - en
            if gap < 0.15:
                en_eff = times[j + 1][0] - 1.0 / FPS
            else:
                en_eff, pause = en + 0.05, gap - 0.05
        else:
            en_eff = en + 0.05
        scale = float(np.clip((en_eff - st_) * FPS / max(kd[a:b].sum(), 1e-3), 0.3, 4.0))
        new[a:b] = np.maximum(1, np.round(kd[a:b] * scale)).astype(int)
        if sg['space'] is not None:
            new[sg['space']] = max(1, int(round(pause * FPS)) - sum(new[p] for p in sg['punct']))
    want = int(round(total_sec * FPS))
    new[-1] = max(1, new[-1] + want - int(new.sum()))      # trailing silence goes to the end pad
    return new


def measure_f0(a24, n):
    w16 = librosa.resample(a24, orig_sr=SR, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=70, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    t = np.arange(len(f0)) * 0.01
    grid = (np.arange(n) + 0.5) * 0.0125
    v = np.interp(grid, t, (~np.isnan(f0)).astype(float)) > 0.5
    ok = ~np.isnan(f0)
    if ok.sum() < 5:
        return np.zeros(n, np.float32)
    c = np.exp(np.interp(grid, t[ok], np.log(f0[ok])))
    return np.where(v, c, 0.0).astype(np.float32)


def measure_n(a24, n):
    mel = MEL(torch.from_numpy(a24))                       # [80, frames]
    x = torch.log((1e-5 + mel).norm(dim=0)).numpy()
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)), mode='edge')
    return x[:n].astype(np.float32)


# ---------------- 1. calibration on Kokoro's own audio ----------------
from transformers import pipeline  # noqa: E402

asr = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=-1)


def hear(a24):
    w16 = librosa.resample(np.asarray(a24, dtype=np.float32), orig_sr=SR, target_sr=16000)
    return asr({'raw': w16, 'sampling_rate': 16000}, generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 120})['text'].strip()


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


def cer(ref, hyp):
    return lev(clean(ref), clean(hyp)) / max(1, len(clean(ref)))


scores = pd.read_csv(os.path.join(ROOT, 'runs', 'k03', 'pilot_scores.csv'), dtype={'id': str}).set_index('id')
ref32 = pd.read_csv(os.path.join(ROOT, 'eval', 'ref_32.tsv'), sep='\t', dtype=str, keep_default_na=False).head(NCLIP)
rows = []
for r in ref32.itertuples():
    norm = normalize(r.text)
    ps = G(norm)
    I = kokoro_text(ps)
    segs = segments_of(ps)

    # Kokoro's own render: predicted F0 / N versus the same things measured on its audio
    base = np.maximum(1, np.round(I['dur'].numpy())).astype(int)
    aln = align_matrix(I['T'], base)
    with torch.no_grad():
        F0k, Nk = m.predictor.F0Ntrain(I['d'].transpose(-1, -2) @ aln, I['r'][:, 128:])
        torch.manual_seed(0)
        ak = m.decoder(I['t_en'] @ aln, F0k, Nk, I['r'][:, :128]).squeeze().numpy()
    n = F0k.shape[1]
    f0m, nm = measure_f0(ak, n), measure_n(ak, n)
    F0k_, Nk_ = F0k[0].numpy(), Nk[0].numpy()
    both = (f0m > 50) & (F0k_ > 50)
    row = dict(id=r.id, norm=norm,
               n_corr=float(np.corrcoef(Nk_, nm)[0, 1]), n_offset=float((nm - Nk_).mean()), n_std_ratio=float(nm.std() / Nk_.std()),
               f0_ratio=float(np.median(f0m[both] / F0k_[both])) if both.sum() > 5 else float('nan'),
               voiced_agree=float(((f0m > 50) == (F0k_ > 50)).mean()),
               f0_unvoiced_kokoro=float(np.median(F0k_[F0k_ <= 50])) if (F0k_ <= 50).any() else float('nan'))

    # the Chatterbox (voice-converted) clip -> measured inputs -> pretrained 8M decoder
    a, sr = sf.read(os.path.join(ROOT, 'runs', 'k03', 'pilot', 'T2v', f'{r.id}.wav'), dtype='float32')
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    w16 = librosa.resample(a, orig_sr=SR, target_sr=16000)
    spoken = str(scores.loc[r.id, 'teacher'])
    words = [re.sub(r"[^a-z']", '', w.lower()) for w in TOKEN_RE.findall(strip_accents(spoken).replace('-', ' ')) if w not in PUNCT]
    words = [w for w in words if w]
    times = word_times(w16, words) if len(words) == len(segs) else None
    if times is None:
        row['skipped'] = True
        rows.append(row)
        continue
    total = min(len(a) / SR, times[-1][1] + 0.2)
    fr = frames_from_clip(I, segs, times, total)
    F = int(fr.sum())
    aln2 = align_matrix(I['T'], fr)
    f0c, nc = measure_f0(a[:F * 600], 2 * F), measure_n(a[:F * 600], 2 * F)
    with torch.no_grad():
        torch.manual_seed(0)
        y_cb = VD(I['t_en'] @ aln2, torch.from_numpy(f0c)[None], torch.from_numpy(nc - row['n_offset'])[None]).squeeze().numpy()
        torch.manual_seed(0)
        y_ko = VD(I['t_en'] @ aln, F0k, Nk).squeeze().numpy()
    for name, x in (('cb_ref', a), ('dec_from_cb', y_cb), ('dec_from_kokoro', y_ko)):
        sf.write(os.path.join(OUT, f'{r.id}_{name}.wav'), x, SR)
    row.update(cer_cb_ref=cer(norm, hear(a)), cer_dec_from_cb=cer(norm, hear(y_cb)), cer_dec_from_kokoro=cer(norm, hear(y_ko)),
               frames=F, sec=F / FPS)
    rows.append(row)
    print(r.id, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ('norm', 'id')}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'direct_check.csv'), index=False)
num = df.select_dtypes('number')
print('MEAN', num.mean().round(3).to_dict())
print(f'{time.time() - T0:.0f}s')
