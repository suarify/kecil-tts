"""Phase 0 SOTA eval harness.

usage: python scripts/sota_eval.py ONNX_PATH --tag NAME [--text CKPT] [--rows SHARD]
       python scripts/sota_eval.py dummy --tag teacher --utmos-only --wavdir DIR

- renders the 80 fixed eval sentences with onnxruntime CPU -> runs/sota/<tag>/<id>.wav
- F0 health: pYIN on outputs; predicted-vs-measured F0 on held rows (if --text given)
- UTMOS22-strong at 16 kHz (mean + 10th pct)
- writes runs/sota/<tag>/metrics.json and appends runs/sota/leaderboard.tsv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument('onnx')
ap.add_argument('--tag', required=True)
ap.add_argument('--text', default=None, help='text checkpoint for predicted-F0 health')
ap.add_argument('--rows', default='data/teacher/shard_0000.pt')
ap.add_argument('--utmos-only', action='store_true')
ap.add_argument('--wavdir', default=None)
A = ap.parse_args()

OUT = f'runs/sota/{A.tag}'
os.makedirs(OUT, exist_ok=True)
SR = 24000

import torch

_utmos = None


def utmos(wavs):
    global _utmos
    if _utmos is None:
        import torchaudio
        _utmos = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True, verbose=False).eval()
        _utmos._sr = SR
    vals = []
    import torchaudio
    with torch.no_grad():
        for a in wavs:
            x = torch.from_numpy(np.asarray(a, np.float32)).unsqueeze(0)
            if x.shape[-1] != 16000:
                x = torchaudio.functional.resample(x, SR, 16000)
            vals.append(float(_utmos(x, 16000).item()))
    return dict(utmos_mean=round(float(np.mean(vals)), 4), utmos_p10=round(float(np.percentile(vals, 10)), 4))


def pyin_f0(a):
    import librosa
    f0, voiced, _ = librosa.pyin(np.asarray(a, np.float32), fmin=70, fmax=500, sr=SR,
                                 frame_length=1024, hop_length=240)
    return np.nan_to_num(f0)


def f0_health(f0):
    """F0 statistics on a pYIN contour (Hz, 0 = unvoiced)."""
    v = f0 > 50
    voiced_f0 = f0[v]
    if len(voiced_f0) < 10:
        return dict(voiced_frames=int(v.sum()))
    jumps = np.abs(np.diff(voiced_f0)) > 25
    span = np.percentile(voiced_f0, 90) - np.percentile(voiced_f0, 10)
    return dict(voiced_frac=round(float(v.mean()), 3),
                jumps_gt25=round(float(jumps.mean()), 4),
                span=int(span))


if A.utmos_only:
    files = sorted(glob.glob(f'{A.wavdir}/*.wav'))[:20]
    m = utmos([sf.read(f)[0] for f in files])
    m['n'] = len(files)
    json.dump(m, open(f'{OUT}/metrics.json', 'w'), indent=1)
    print('UTMOS-only', m)
    sys.exit(0)

# ---------- render ----------
import onnxruntime as ort

from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
sess = ort.InferenceSession(A.onnx, providers=['CPUExecutionProvider'],
                            sess_options=ort.SessionOptions())
eval_rows = list(csv.DictReader(open('eval/sota_eval.tsv', encoding='utf-8'), delimiter='\t'))
wavs = []
f0s = []
for r in eval_rows:
    ids = np.array([g2p.tokens(g2p(normalize(r['text'])))], dtype=np.int64)
    y = sess.run(None, {'input_ids': ids, 'speed': np.ones(1, np.float32)})[0][0]
    sf.write(f"{OUT}/{r['id']}.wav", y, SR)
    wavs.append(y)
    f0s.append(pyin_f0(y))

metrics = dict(tag=A.tag, n=len(eval_rows), **utmos(wavs))
f0h = [f0_health(f) for f in f0s]
for key in ('voiced_frac', 'jumps_gt25', 'span'):
    vals = [h[key] for h in f0h if key in h]
    if vals:
        metrics['f0_' + key] = round(float(np.mean(vals)), 4)

# ---------- predicted vs measured F0 on held rows ----------
if A.text:
    import torch.nn as nn
    from student import TextStudent
    ts = TextStudent('s+mlp').eval()
    ts.load_state_dict(torch.load(A.text, map_location='cpu', weights_only=False)['model'])
    rows = torch.load(A.rows, map_location='cpu', weights_only=False)
    # held = rows not in the eval set by id/text; use the last 60 to avoid train overlap
    held = [r for r in rows if r['id'] not in {x['id'] for x in eval_rows}][-60:]
    dropouts, uv_err, jumps, spans = [], [], [], []
    with torch.no_grad():
        for r in held:
            ids = torch.tensor([g2p.tokens(g2p(normalize(r['text'])))], dtype=torch.long)
            t, p, s = ts, ts.predictor, ts.style
            d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
            x, T = d_en, ids.shape[1]
            sx = s[:, :, None].expand(-1, -1, T)
            x = torch.cat([x, sx], 1)
            for b in p.text_encoder.lstms:
                x = (b(x.transpose(-1, -2))[0].transpose(-1, -2) if isinstance(b, nn.LSTM)
                     else torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1))
            d = x.transpose(-1, -2)
            dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
            n = torch.clamp(torch.round(dur), min=1).long()
            end = torch.cumsum(n, 0); start = end - n
            f = torch.arange(int(end[-1]))[None]
            aln = ((f >= start[:, None]) & (f < end[:, None])).float()[None]
            F0p, _ = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
            F0p = F0p[0].numpy()
            F0m = r['F0'].numpy()
            L = min(len(F0p), len(F0m))
            a, b = F0p[:L], F0m[:L]
            vm = b > 50
            vp = a > 50
            if vm.sum() < 10:
                continue
            dropouts.append(float((a[vm] < 50).mean()))
            uv_err.append(float((vm != vp).mean()))
            av = a[vm]
            jumps.append(float((np.abs(np.diff(av)) > 25).mean()))
            sm = np.percentile(b[vm], 90) - np.percentile(b[vm], 10)
            sp = np.percentile(av, 90) - np.percentile(av, 10)
            spans.append(float(sp / (sm + 1e-6)))
    metrics.update(held_n=len(dropouts),
                   dropout=round(float(np.mean(dropouts)), 4) if dropouts else None,
                   uv_err=round(float(np.mean(uv_err)), 4) if uv_err else None,
                   jumps_gt25=round(float(np.mean(jumps)), 4) if jumps else None,
                   span_ratio=round(float(np.mean(spans)), 3) if spans else None)

json.dump(metrics, open(f'{OUT}/metrics.json', 'w'), indent=1)
board = 'runs/sota/leaderboard.tsv'
new = not os.path.exists(board)
with open(board, 'a', encoding='utf-8') as f:
    if new:
        f.write('tag\tn\tutmos_mean\tutmos_p10\tf0_voiced_frac\tf0_jumps_gt25\tdropout\tuv_err\tjumps_gt25\tspan_ratio\n')
    f.write('\t'.join(str(metrics.get(k, '')) for k in
                      ('tag', 'n', 'utmos_mean', 'utmos_p10', 'f0_voiced_frac', 'f0_jumps_gt25',
                       'dropout', 'uv_err', 'jumps_gt25', 'span_ratio')) + '\n')
print(json.dumps(metrics, indent=1))
