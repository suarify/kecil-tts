"""Does letter-level phoneme timing help?  Training-free test on CPU.

Build rows for the same Chatterbox clips twice — phoneme timings guessed from Kokoro's proportions,
and phoneme timings from the letter alignment — and feed both to the SAME pretrained 8M decoder
(Paradee v1.3, never trained on these clips). If the letter timings line the phoneme features up with
the audio better, the decoder's speech should be more intelligible (lower Whisper CER).

Run:  .venv/Scripts/python scripts/timing_test.py <paradee scripts dir> [n_clips]
"""
import os
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PS = sys.argv[1]
N = sys.argv[2] if len(sys.argv) > 2 else '24'
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
sys.path.insert(0, PS)
OUT = os.path.join(ROOT, 'runs', 'timing')
env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, '.stubs'), PYTHONUTF8='1')
for name, extra in (('prop', []), ('letters', ['--letters'])):
    cmd = [sys.executable, os.path.join(ROOT, 'paradee_ms', 'ms_rows.py'), '--clips', os.path.join(ROOT, 'runs', 'k03', 'pilot', 'T2v'),
           '--tsv', os.path.join(ROOT, 'eval', 'pilot_150.tsv'), '--g2p', os.path.join(ROOT, 'g2p'),
           '--out', os.path.join(OUT, name, 'shard_0000.pt'), '--max', N, '--device', 'cpu'] + extra
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, errors='replace')
    print(name, '|', ' | '.join(l for l in p.stdout.splitlines() if l.startswith(('rows', 'words', 'skip'))), flush=True)
    if p.returncode:
        print(p.stderr[-1500:])
        sys.exit(1)

import librosa  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import alignment  # noqa: E402
from transformers import pipeline  # noqa: E402
from vhead import VDecoder  # noqa: E402

st = dict(torch.load(hf_hub_download('sahilmahendrakar/Paradee-8M-v1.3', 'pytorch/decoder.pt'), map_location='cpu')['model'])
st['style'] = st['style'][0:1].clone()
VD = VDecoder(head='v2', dim=160, blocks=4).eval()
VD.load_state_dict(st)
asr = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=-1)


def hear(a):
    w = librosa.resample(np.asarray(a, dtype=np.float32), orig_sr=24000, target_sr=16000)
    return asr({'raw': w, 'sampling_rate': 16000}, generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 120})['text'].strip()


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
    return min(1.0, lev(clean(ref), clean(hyp)) / max(1, len(clean(ref))))


res = {}
for name in ('prop', 'letters'):
    rows = torch.load(os.path.join(OUT, name, 'shard_0000.pt'))
    for r in rows:
        aln = alignment(r['pred_dur'])
        Fr = aln.shape[1]
        with torch.no_grad():
            torch.manual_seed(0)
            y = VD((r['t_en'].float() @ aln)[None], r['F0'][None, :2 * Fr], r['N'][None, :2 * Fr]).squeeze().numpy()
        res.setdefault(r['id'], dict(text=r['text']))[name] = cer(r['text'], hear(y))
        if name == 'prop':
            res[r['id']]['clip'] = cer(r['text'], hear(r['audio'].float().numpy() / 32767))
            res[r['id']]['moved'] = None
        else:
            prev = torch.load(os.path.join(OUT, 'prop', 'shard_0000.pt'))
            pr = next(x for x in prev if x['id'] == r['id'])
            res[r['id']]['moved'] = float((pr['pred_dur'] != r['pred_dur']).float().mean())
both = [v for v in res.values() if 'prop' in v and 'letters' in v]
P = np.array([v['prop'] for v in both])
L = np.array([v['letters'] for v in both])
C = np.array([v['clip'] for v in both])
print(f'n = {len(both)} clips | share of phoneme durations changed by letter timing: {np.mean([v["moved"] for v in both]):.0%}')
print(f'Chatterbox clip itself          CER mean {C.mean():.3f} median {np.median(C):.3f}')
print(f'decoder, guessed timings        CER mean {P.mean():.3f} median {np.median(P):.3f}')
print(f'decoder, letter-aligned timings CER mean {L.mean():.3f} median {np.median(L):.3f}')
print(f'letter timing better on {(L < P - 1e-9).sum()} clips, worse on {(L > P + 1e-9).sum()}, same on {(np.abs(L - P) < 1e-9).sum()}')
