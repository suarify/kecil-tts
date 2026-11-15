"""How wrong are the phoneme timings in the training rows?  (CPU, no training)

The rows place words by forced alignment, but split each word among its phonemes in proportion to
Kokoro's predicted durations (a guess). The aligner also gives a time for every LETTER. For Malay
words, each phoneme comes from a known letter, so we can compare:
    guessed phoneme boundaries   vs   letter-aligned phoneme boundaries.
A decoder frame is 25 ms; boundaries off by more than that feed the decoder the wrong phoneme.

Run:  .venv/Scripts/python scripts/align_check.py [n_clips]
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
from kokoro import KModel  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ms_g2p import G2P, TOKEN_RE, PUNCT, strip_accents, VOCAB  # noqa: E402
from ms_norm import normalize  # noqa: E402

NCLIP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
FPS = 40.0
torch.set_num_threads(max(1, os.cpu_count() - 1))
m = KModel(repo_id='hexgrad/Kokoro-82M').eval()
BELLA = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_bella.pt'), weights_only=True)
G = G2P(stress=False, r='r', diph='glide', final_a='a')
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


@torch.no_grad()
def kokoro_dur(ps):
    ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]])
    T = ids.shape[1]
    r = BELLA[min(max(0, T - 3), BELLA.shape[0] - 1)]
    L = torch.full((1,), T)
    mask = torch.zeros(1, T, dtype=torch.bool)
    d = m.predictor.text_encoder(m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2), r[:, 128:], L, mask)
    x, _ = m.predictor.lstm(d)
    return torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0).numpy()


scores = pd.read_csv(os.path.join(ROOT, 'runs', 'k03', 'pilot_scores.csv'), dtype={'id': str})
scores = scores[scores.cer <= 0.03].head(NCLIP)
diffs, n_words, n_ph, onset_spread = [], 0, 0, []
for r in scores.itertuples():
    a, sr = sf.read(os.path.join(ROOT, 'runs', 'k03', 'pilot', 'T2v', f'{r.id}.wav'), dtype='float32')
    w16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000)
    norm = normalize(r.text)
    toks = [t for t in TOKEN_RE.findall(strip_accents(norm).replace('-', ' ')) if t not in PUNCT]
    spoken = [t for t in TOKEN_RE.findall(strip_accents(str(r.teacher)).replace('-', ' ')) if t not in PUNCT]
    if len(toks) != len(spoken):
        continue
    words = [re.sub(r"[^a-z']", '', w.lower()) for w in spoken]
    if any(not w for w in words):
        continue
    with torch.no_grad():
        em = torch.log_softmax(fa(w16[None])[0][0], -1).numpy()
    tgt, first = [], []
    for w in words:
        first.append(len(tgt))
        tgt += [LAB[c] for c in w if c in LAB]
    if len(tgt) * 2 + 1 > em.shape[0]:
        continue
    path = ctc_viterbi(em, np.array(tgt))
    sec = len(w16) / 16000 / em.shape[0]
    char_start = np.array([np.where(path == 2 * c + 1)[0][0] * sec if (path == 2 * c + 1).any() else np.nan for c in range(len(tgt))])
    ps = G(norm)
    kd = kokoro_dur(ps)
    # token index of each phoneme-word start
    tok_i, pos, starts = 1, 0, []
    for part in ps.split(' '):
        letters = [ch for ch in part if ch not in PUNCT and ch in VOCAB]
        starts.append((tok_i, tok_i + len(letters)))
        tok_i += sum(1 for ch in part if ch in VOCAB) + 1          # + the space token
    if len(starts) != len(toks):
        continue
    for j, tok in enumerate(toks):
        w = tok.lower().strip("'")
        malay = not (len(tok) >= 2 and tok.isupper()) and w not in G.lex_x and w not in G.lex_en and words[j] == w
        if not malay or len(w) < 3:
            continue
        spans = []
        ph = G._letters(G.respell(w), spans)
        a0, b0 = starts[j]
        if len(ph) != b0 - a0 or len(ph) < 2:
            continue
        src = [max(i for i, k in spans if k <= p) for p in range(len(ph))]         # letter index that produced phoneme p
        cs = char_start[first[j]:first[j] + len(w)]
        if np.isnan(cs).any():
            continue
        st = cs[0]
        en = char_start[first[j + 1]] if j + 1 < len(toks) and not np.isnan(char_start[first[j + 1]]) else cs[-1] + 0.08
        if en - st > 1.2 or en <= st:
            continue
        aligned = np.array([cs[i] for i in src])                                   # phoneme starts from letters
        prop = st + np.concatenate([[0], np.cumsum(kd[a0:b0])[:-1]]) / kd[a0:b0].sum() * (en - st)   # the guess used in the rows
        d = (aligned[1:] - prop[1:]) * 1000
        diffs += list(d)
        n_words += 1
        n_ph += len(ph)
diffs = np.abs(np.array(diffs))
print(f'clips {len(scores)} | Malay words {n_words} | phonemes {n_ph} | inner boundaries {len(diffs)}')
print(f'|guess - letter-aligned| boundary error: median {np.median(diffs):.0f} ms, mean {diffs.mean():.0f} ms, 90th pct {np.percentile(diffs, 90):.0f} ms')
print(f'share off by more than one decoder frame (25 ms): {(diffs > 25).mean():.0%} | more than two frames (50 ms): {(diffs > 50).mean():.0%} | more than 100 ms: {(diffs > 100).mean():.0%}')
