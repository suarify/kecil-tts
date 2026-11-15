"""Kokoro-read rows for the text side's Malay prior (continue.plan.v5.md step 3.2).

For each sentence, Kokoro's own predicted durations, pitch and loudness plus its text features.
No audio is stored: the text trainer never reads it, and Kokoro's sound never reaches the decoder.

usage: ms_kokoro_rows.py --tsv corpus_train.tsv --exclude cb_1100.tsv --n 3000 --out data/kokoro/shard_0000.pt
(tsv columns: id, text, phonemes — the phonemes come from the frozen G2P options)
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--tsv', required=True)
ap.add_argument('--exclude', default='')
ap.add_argument('--n', type=int, default=3000)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
A = ap.parse_args()
DEV = A.device

from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

T0 = time.time()
m = KModel(repo_id='hexgrad/Kokoro-82M').to(DEV).eval()
VOCAB = m.vocab


def voice(name):
    return torch.load(hf_hub_download('hexgrad/Kokoro-82M', f'voices/{name}.pt'), weights_only=True)


B, D = voice('af_bella'), voice('ef_dora')
PACK = torch.cat([0.5 * B[..., :128] + 0.5 * D[..., :128], D[..., 128:]], -1)      # the search's F1: dora prosody

d = pd.read_csv(A.tsv, sep='\t', dtype=str, keep_default_na=False)
if A.exclude:
    ex = set(pd.read_csv(A.exclude, sep='\t', dtype=str, keep_default_na=False).text)
    d = d[~d.text.isin(ex)]
d = d[d.phonemes.str.len().between(8, 400)].sample(frac=1, random_state=11).head(A.n)
rows = []
for r in d.itertuples():
    ps = r.phonemes
    with torch.no_grad():
        ids = torch.LongTensor([[0, *[VOCAB[c] for c in ps if c in VOCAB], 0]]).to(DEV)
        T = ids.shape[1]
        ref = PACK[min(max(0, T - 3), PACK.shape[0] - 1)].to(DEV)
        L = torch.full((1,), T, device=DEV)
        mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
        d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
        s = ref[:, 128:]
        dd = m.predictor.text_encoder(d_en, s, L, mask)
        x, _ = m.predictor.lstm(dd)
        dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1).squeeze(0)
        pred = torch.round(dur).clamp(min=1).long()
        idx = torch.repeat_interleave(torch.arange(T, device=DEV), pred)
        aln = torch.zeros(T, idx.shape[0], device=DEV)
        aln[idx, torch.arange(idx.shape[0], device=DEV)] = 1
        aln = aln[None]
        F0, N = m.predictor.F0Ntrain(dd.transpose(-1, -2) @ aln, s)
        t_en = m.text_encoder(ids, L, mask)
    rows.append(dict(ids=ids.squeeze(0).cpu(), dur=dur.cpu(), pred_dur=pred.cpu(), t_en=t_en.squeeze(0).half().cpu(),
                     F0=F0.squeeze(0).cpu(), N=N.squeeze(0).cpu(), text=r.text, ps=ps, id=r.id))
os.makedirs(os.path.dirname(A.out) or '.', exist_ok=True)
torch.save(rows, A.out)
print(f'kokoro rows {len(rows)} | {time.time() - T0:.0f}s', flush=True)
