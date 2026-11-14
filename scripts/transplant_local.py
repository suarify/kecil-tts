"""Prosody-transplant proof on CPU (no Kaggle): 32 reference sentences.

Keep Kokoro's decoder (teacher F1 = af_bella/ef_dora blend) but drive it with the word timing and the
pitch contour of the Chatterbox clip the owner approved.
  F1   teacher as it is
  TPd  + Chatterbox word durations (MMS forced alignment, as in the owner's kokoro-malay-voicetune notebook)
  TPf  + Chatterbox pitch contour (pYIN), kept on Kokoro's own voiced frames
Run:  .venv/Scripts/python scripts/transplant_local.py
Writes runs/k05/{clips_transplant.zip, transplant_rows.csv, transplant_scores.csv}.
"""
import os
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stubs = os.path.join(ROOT, '.stubs', 'misaki')
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    with open(os.path.join(stubs, name + '.py'), 'w') as f:
        f.write('def __getattr__(name):\n    return type(name, (), {})\n')
with open(os.path.join(stubs, '__init__.py'), 'w') as f:
    f.write('from . import en, espeak\n')
sys.path.insert(0, os.path.dirname(stubs))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))

import librosa  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402
from ms_g2p import G2P, TOKEN_RE, PUNCT, strip_accents  # noqa: E402
from ms_norm import normalize  # noqa: E402

T0 = time.time()
SR, FPS = 24000, 40.0
OUT = os.path.join(ROOT, 'runs', 'k05')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, os.cpu_count() - 1))
m = KModel(repo_id='hexgrad/Kokoro-82M').eval()
VOCAB = m.vocab


def voice(name):
    return torch.load(hf_hub_download('hexgrad/Kokoro-82M', f'voices/{name}.pt'), weights_only=True)


BELLA, DORA = voice('af_bella'), voice('ef_dora')
PACK = torch.cat([0.5 * BELLA[..., :128] + 0.5 * DORA[..., :128], DORA[..., 128:]], -1)
G = G2P(stress=False, r='r', diph='glide', final_a='a')

# ---------------- forced alignment (MMS CTC) ----------------
bundle = torchaudio.pipelines.MMS_FA
fa = bundle.get_model().eval()
LAB = {c: i for i, c in enumerate(bundle.get_labels())}


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
        st = np.stack([prev, c1, c2])
        k = st.argmax(0)
        dp[t] = st[k, np.arange(S)] + lp[t, ext]
        bp[t] = k
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.zeros(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def word_times(w16, words):
    """(start, end) seconds of each word, or None."""
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


# ---------------- Kokoro pieces ----------------
@torch.no_grad()
def internals(ps):
    ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]])
    T = ids.shape[1]
    r = PACK[min(max(0, T - 3), PACK.shape[0] - 1)]
    L = torch.full((1,), T)
    mask = torch.zeros(1, T, dtype=torch.bool)
    bert = m.bert(ids, attention_mask=(~mask).int())
    d_en = m.bert_encoder(bert).transpose(-1, -2)
    s = r[:, 128:]
    d = m.predictor.text_encoder(d_en, s, L, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0)
    return dict(T=T, r=r, s=s, d=d, t_en=m.text_encoder(ids, L, mask), dur=dur)


@torch.no_grad()
def render(I, frames, contour=None):
    frames = torch.as_tensor(frames).long()
    idx = torch.repeat_interleave(torch.arange(I['T']), frames)
    aln = torch.zeros(I['T'], idx.shape[0])
    aln[idx, torch.arange(idx.shape[0])] = 1
    aln = aln[None]
    F0, N = m.predictor.F0Ntrain(I['d'].transpose(-1, -2) @ aln, I['s'])
    kf0 = F0.squeeze(0).numpy().copy()
    if contour is not None:
        n = min(len(contour), len(kf0))
        new = kf0.copy()
        voiced = kf0[:n] > 50                       # Kokoro decides where there is voice
        new[:n][voiced] = contour[:n][voiced]
        F0 = torch.from_numpy(new)[None]
    torch.manual_seed(0)
    a = m.decoder(I['t_en'] @ aln, F0, N, I['r'][:, :128]).squeeze()
    return a.float().numpy(), kf0


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


def new_durations(I, segs, times):
    kd = I['dur'].numpy()
    new = np.maximum(1, np.round(kd)).astype(int)
    new[0] = max(1, int(round(times[0][0] * FPS)))
    # CTC word spans are tight (the tail of a word lands in blank frames). So a word runs until the
    # next word starts, unless the gap is a real pause (>= 150 ms); then the word gets a 50 ms tail
    # and the rest of the gap goes to the space / punctuation tokens.
    for j, (sg, (st, en)) in enumerate(zip(segs, times)):
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
        scale = float(np.clip((en_eff - st) * FPS / max(kd[a:b].sum(), 1e-3), 0.4, 3.0))
        new[a:b] = np.maximum(1, np.round(kd[a:b] * scale)).astype(int)
        if sg['space'] is not None:
            new[sg['space']] = max(1, int(round(pause * FPS)) - sum(new[p] for p in sg['punct']))
    return new


def donor_contour(w16, n, target_median):
    """Continuous pitch contour of the donor clip on Kokoro's 12.5 ms grid, moved to the teacher's median pitch."""
    f0, _, _ = librosa.pyin(w16, fmin=70, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    t = np.arange(len(f0)) * 0.01
    v = ~np.isnan(f0)
    if v.sum() < 8:
        return None, 1.0
    lf = np.interp(t, t[v], np.log(f0[v]))          # bridge unvoiced gaps
    k = 5
    lf = np.convolve(np.pad(lf, k // 2, mode='edge'), np.ones(k) / k, mode='valid')
    grid = (np.arange(n) + 0.5) * 0.0125
    c = np.exp(np.interp(grid, t, lf))
    ratio = float(np.clip(target_median / np.median(f0[v]), 0.6, 1.7))
    return (c * ratio).astype(np.float32), ratio


scores = pd.read_csv(os.path.join(ROOT, 'runs', 'k03', 'pilot_scores.csv'), dtype={'id': str}).set_index('id')
ref32 = pd.read_csv(os.path.join(ROOT, 'eval', 'ref_32.tsv'), sep='\t', dtype=str, keep_default_na=False)
rows, clips = [], {}
for r in ref32.itertuples():
    a, sr = sf.read(os.path.join(ROOT, 'runs', 'k03', 'pilot', 'T2', f'{r.id}.wav'), dtype='float32')
    w16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
    norm = normalize(r.text)
    ps = G(norm)
    segs = segments_of(ps)
    spoken = str(scores.loc[r.id, 'teacher'])       # what Chatterbox was given (English words respelled)
    words = [re.sub(r"[^a-z']", '', w.lower()) for w in TOKEN_RE.findall(strip_accents(spoken).replace('-', ' ')) if w not in PUNCT]
    words = [w for w in words if w]
    row = dict(id=r.id, category=r.category, norm=norm, n_seg=len(segs), n_words=len(words))
    times = word_times(w16, words) if len(words) == len(segs) else None
    if times is None:
        row['skipped'] = 'no alignment'
        rows.append(row)
        print('skip', r.id, len(segs), len(words), flush=True)
        continue
    I = internals(ps)
    base = np.maximum(1, np.round(I['dur'].numpy())).astype(int)
    a_f1, kf0 = render(I, base)
    new = new_durations(I, segs, times)
    a_d, kf0_new = render(I, new)
    kv = kf0_new[kf0_new > 50]
    contour, ratio = donor_contour(w16, len(kf0_new), float(np.median(kv)) if len(kv) else 200.0)
    a_f, _ = render(I, new, contour) if contour is not None else (a_d, None)
    # word onset-to-onset intervals: Kokoro's own prediction vs the Chatterbox clip
    kd = I['dur'].numpy()
    on_k = np.array([kd[:s['a']].sum() / FPS for s in segs])
    on_c = np.array([s for s, _ in times])
    if len(segs) > 2:
        e = np.log(np.diff(on_k) / np.maximum(np.diff(on_c), 0.02))
    else:
        e = np.zeros(1)
    row.update(sec_f1=len(a_f1) / SR, sec_cb=len(w16) / 16000, sec_tp=len(a_f) / SR, f0_ratio=ratio,
               pace_err=float(e.mean()), rhythm_err=float(e.std()), unvoiced_frac=float((kf0_new <= 50).mean()))
    for name, x in (('F1', a_f1), ('TPd', a_d), ('TPf', a_f)):
        clips[(name, r.id)] = x
    rows.append(row)
    print('done', r.id, round(time.time() - T0), 's', f"pace {row['pace_err']:+.2f} rhythm {row['rhythm_err']:.2f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'transplant_rows.csv'), index=False)
pd.DataFrame([dict(config=k[0], id=k[1], cer=None, acc=None) for k in clips]).to_csv(os.path.join(OUT, 'transplant_scores.csv'), index=False)
with zipfile.ZipFile(os.path.join(OUT, 'clips_transplant.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for (name, i), x in clips.items():
        p = os.path.join(OUT, f'_{name}_{i}.wav')
        sf.write(p, x, SR)
        z.write(p, f'{name}/{i}.wav')
        os.remove(p)
ok = df[df['skipped'].isna()] if 'skipped' in df else df
print(f"used {len(ok)}/{len(df)} | pace err {ok.pace_err.mean():+.3f} (<0 = Kokoro faster than Chatterbox) | "
      f"rhythm err {ok.rhythm_err.mean():.3f} | Kokoro s {ok.sec_f1.mean():.2f} vs Chatterbox s {ok.sec_cb.mean():.2f} | "
      f"{time.time() - T0:.0f}s")
