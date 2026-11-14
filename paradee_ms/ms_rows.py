"""Turn Chatterbox clips into Paradee-style training rows (continue.plan.v4.md step C1, measuring part).

For every clip <id>.wav (24 kHz, voice-converted Chatterbox audio) it produces one row with the same
keys Paradee's trainers read from Kokoro teacher shards:
  ids, dur, pred_dur, t_en, F0, N, audio, text, ps
where durations come from forced alignment of the words, F0 from pYIN, N (loudness) from the mel
spectrum minus a constant measured on Kokoro's own renders, t_en from Kokoro's text encoder (no audio needed).

usage: ms_rows.py --clips DIR --tsv ref_32.tsv --g2p DIR --out data/teacher/shard_0000.pt [--device cuda]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ap = argparse.ArgumentParser()
ap.add_argument('--clips', required=True)
ap.add_argument('--tsv', required=True)
ap.add_argument('--g2p', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--max', type=int, default=10 ** 9)
A = ap.parse_args()
sys.path.insert(0, A.g2p)

import librosa  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402
from ms_g2p import G2P, TOKEN_RE, PUNCT, strip_accents  # noqa: E402
from ms_norm import normalize  # noqa: E402

T0 = time.time()
SR, FPS, DEV = 24000, 40.0, A.device
m = KModel(repo_id='hexgrad/Kokoro-82M').to(DEV).eval()
VOCAB = m.vocab
BELLA = torch.load(hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_bella.pt'), weights_only=True)
G = G2P(stress=False, r='r', diph='glide', final_a='a')
RESPELL = json.load(open(os.path.join(A.g2p, 'teacher_respell.json'), encoding='utf-8'))

bundle = torchaudio.pipelines.MMS_FA
fa = bundle.get_model().to(DEV).eval()
LAB = {c: i for i, c in enumerate(bundle.get_labels())}
MEL = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def respell(text):
    def rep(mm):
        w = mm.group(0)
        r = RESPELL.get(w.lower())
        if r is None:
            return w
        return r.capitalize() if w[0].isupper() else r
    return re.sub(r'[A-Za-z]+', rep, text)


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
        em = torch.log_softmax(fa(torch.from_numpy(w16)[None].to(DEV))[0][0], -1).cpu().numpy()
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
    ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]]).to(DEV)
    T = ids.shape[1]
    r = BELLA[min(max(0, T - 3), BELLA.shape[0] - 1)].to(DEV)
    L = torch.full((1,), T, device=DEV)
    mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
    d = m.predictor.text_encoder(d_en, r[:, 128:], L, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0)
    return dict(ids=ids, T=T, r=r, d=d, dur=dur.cpu(), t_en=m.text_encoder(ids, L, mask))


def align_matrix(T, frames):
    idx = torch.repeat_interleave(torch.arange(T), torch.as_tensor(frames).long())
    aln = torch.zeros(T, idx.shape[0])
    aln[idx, torch.arange(idx.shape[0])] = 1
    return aln[None]


def frames_from_clip(kd, segs, times, total_sec):
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
    new[-1] = max(1, new[-1] + want - int(new.sum()))
    return new


def measure_f0(a24, n):
    w16 = librosa.resample(a24, orig_sr=SR, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=70, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    t = np.arange(len(f0)) * 0.01
    grid = (np.arange(n) + 0.5) * 0.0125
    ok = ~np.isnan(f0)
    if ok.sum() < 5:
        return np.zeros(n, np.float32)
    v = np.interp(grid, t, ok.astype(float)) > 0.5
    c = np.exp(np.interp(grid, t[ok], np.log(f0[ok])))
    return np.where(v, c, 0.0).astype(np.float32)


def measure_n(a24, n):
    mel = MEL(torch.from_numpy(a24))
    x = torch.log((1e-5 + mel).norm(dim=0)).numpy()
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)), mode='edge')
    return x[:n].astype(np.float32)


# ---- constant offset between audio-measured N and Kokoro's own N (measured on Kokoro renders) ----
tsv = pd.read_csv(A.tsv, sep='\t', dtype=str, keep_default_na=False)
offs, nvoiced = [], []
for r in tsv.head(8).itertuples():
    I = kokoro_text(G(normalize(r.text)))
    frames = np.maximum(1, np.round(I['dur'].numpy())).astype(int)
    aln = align_matrix(I['T'], frames).to(DEV)
    with torch.no_grad():
        F0k, Nk = m.predictor.F0Ntrain(I['d'].transpose(-1, -2) @ aln, I['r'][:, 128:])
        ak = m.decoder(I['t_en'] @ aln, F0k, Nk, I['r'][:, :128]).squeeze().cpu().numpy()
    offs.append(float((measure_n(ak, Nk.shape[1]) - Nk[0].cpu().numpy()).mean()))
    vm = (F0k[0].cpu().numpy() > 50)
    nvoiced.append(float(Nk[0].cpu().numpy()[vm].mean()) if vm.any() else float('nan'))
N_OFF = float(np.mean(offs))
N_TARGET = float(np.nanmean(nvoiced))   # mean N on voiced frames of Kokoro renders
print(f'N offset {N_OFF:.3f} (per-sentence std {np.std(offs):.3f}); Kokoro voiced N mean {N_TARGET:.2f}', flush=True)

rows, stats = [], []
for r in tsv.itertuples():
    if len(rows) >= A.max:
        break
    p = os.path.join(A.clips, f'{r.id}.wav')
    if not os.path.exists(p):
        continue
    a, sr = sf.read(p, dtype='float32')
    if a.ndim > 1:
        a = a.mean(1)
    if sr != SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=SR)
    w16 = librosa.resample(a, orig_sr=SR, target_sr=16000)
    norm = normalize(r.text)
    ps = G(norm)
    segs = segments_of(ps)
    spoken = respell(norm)
    words = [re.sub(r"[^a-z']", '', w.lower()) for w in TOKEN_RE.findall(strip_accents(spoken).replace('-', ' ')) if w not in PUNCT]
    words = [w for w in words if w]
    if len(words) != len(segs) or not words:
        print('skip (word count)', r.id, len(segs), len(words), flush=True)
        continue
    times = word_times(w16, words)
    if times is None:
        print('skip (alignment)', r.id, flush=True)
        continue
    I = kokoro_text(ps)
    total = min(len(a) / SR, times[-1][1] + 0.15)
    fr = frames_from_clip(I['dur'].numpy(), segs, times, total)
    F = int(fr.sum())
    au = a[:F * 600]
    if len(au) < F * 600:
        au = np.pad(au, (0, F * 600 - len(au)))
    f0c = measure_f0(au, 2 * F)
    n_raw = measure_n(au, 2 * F) - N_OFF
    vmask = f0c > 50
    # bring the clip to Kokoro's loudness: scaling audio by g adds ln(g) to N (N is a log norm)
    shift = N_TARGET - float(n_raw[vmask].mean()) if vmask.any() else 0.0
    peak0 = float(np.abs(au).max()) + 1e-9
    gain = float(min(np.exp(shift), 0.99 / peak0))
    au = au * gain
    nc = n_raw + np.log(gain)
    rows.append(dict(ids=I['ids'].squeeze(0).cpu(), dur=torch.from_numpy(fr.astype(np.float32)),
                     pred_dur=torch.from_numpy(fr).long(), t_en=I['t_en'].squeeze(0).half().cpu(),
                     F0=torch.from_numpy(f0c), N=torch.from_numpy(nc.astype(np.float32)),
                     audio=torch.from_numpy((au * 32767).astype(np.int16)), text=r.text, ps=ps, id=r.id))
    kd = I['dur'].numpy()
    stats.append(dict(id=r.id, sec=F / FPS, pace=float(np.log(kd.sum() / fr.sum())), voiced=float((f0c > 50).mean()),
                      n_mean=float(nc.mean()), gain=gain))

os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
torch.save(rows, A.out)
st = pd.DataFrame(stats)
print(f'rows {len(rows)}/{len(tsv)} | {st.sec.sum() / 60:.1f} min audio | voiced {st.voiced.mean():.2f} | '
      f'kokoro/cb pace {st.pace.mean():+.3f} | N mean {st.n_mean.mean():.2f} | gain {st.gain.mean():.2f} | {time.time() - T0:.0f}s', flush=True)
json.dump(dict(rows=len(rows), n_offset=N_OFF, minutes=float(st.sec.sum() / 60), seconds_run=time.time() - T0),
          open(os.path.splitext(A.out)[0] + '_report.json', 'w'))
