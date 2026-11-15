# kokoro-malay-bella / continue.plan.v5.md: the one bounded run.
#   rows with letter-level phoneme timings -> decoder (k09 recipe) -> two text sides (A: k09 recipe;
#   B: Kokoro-read prior first, then Chatterbox rows, checkpoint chosen on dur+asr) -> export both -> Whisper-large.
# Gates (fixed in advance): D1 decoder-only median CER >= 30% lower than the k09 decoder's; D2 smear
# (flatness 2-6 kHz) <= 0.66 and held spectral loss < 2.545; S1 unseen-sentence median CER <= 8%.
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

T0 = time.time()
STAGES = {}
OUT = '/kaggle/working'
W = f'{OUT}/e2e'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd, name=None, cwd=None):
    t = time.time()
    print(f'$ {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    tail = '\n'.join(l for l in p.stdout.splitlines() if not re.search(r'Warning|warn|%\||Xet|FutureWarning', l))
    print(tail[-3000:], flush=True)
    if name:
        STAGES[name] = round(time.time() - t, 1)
        print(f'--- stage {name}: {STAGES[name]} s (total {time.time() - T0:.0f} s)', flush=True)
    if p.returncode != 0:
        json.dump(dict(failed=name or cmd, stages=STAGES, tail=p.stdout[-3000:]), open(f'{OUT}/v5_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED: {cmd}')
    return p.stdout


sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru onnx onnxruntime', 'pip_misc')
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')

DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_rows.py', recursive=True)[0])
K09_DEC = glob.glob('/kaggle/input/**/k09_dec_last.pt', recursive=True)[0]
CLIPS = f'{OUT}/cbwav'
os.makedirs(CLIPS, exist_ok=True)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

_pt = torch.load(glob.glob('/kaggle/input/**/cb_clips.pt', recursive=True)[0])
for _i, _a in _pt.items():
    sf.write(f'{CLIPS}/{_i}.wav', _a.numpy().astype('float32') / 32767, 24000)
print('unpacked', len(_pt), 'clips', flush=True)
del _pt
os.makedirs(W, exist_ok=True)
shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
import pandas as pd  # noqa: E402

tsv = pd.read_csv(f'{DS}/cb_1100.tsv', sep='\t', dtype=str, keep_default_na=False).sample(frac=1, random_state=0)
tsv.to_csv(f'{W}/ref_shuffled.tsv', sep='\t', index=False)
HELD = 60
PY = sys.executable

sh(f'{PY} scripts/ms_init.py init', 'init_weights', W)
rows_out = sh(f'{PY} scripts/ms_rows.py --clips {CLIPS} --tsv ref_shuffled.tsv --g2p {DS} --out data/teacher/shard_0000.pt --device cuda --letters', 'rows', W)
os.makedirs(f'{OUT}/rows', exist_ok=True)
shutil.copy(f'{W}/data/teacher/shard_0000.pt', f'{OUT}/rows/shard_0000.pt')
sh(f'{PY} scripts/ms_kokoro_rows.py --tsv {DS}/corpus_train.tsv --exclude {DS}/cb_1100.tsv --n 3000 --out data/kokoro/shard_0000.pt --device cuda', 'kokoro_rows', W)

dec = sh(f'{PY} scripts/ms_train.py dec --init init/dec_row0.pt --held {HELD} --steps 6000 --bs 16 --max-minutes 12 --out ckpt/dec', 'train_decoder', W)
txtA = sh(f'{PY} scripts/ms_train.py text --init init/text_row0.pt --held {HELD} --steps 6000 --bs 16 --eval-every 100000 --max-minutes 12 --out ckpt/textA', 'train_text_A', W)
sh(f'{PY} scripts/ms_train.py text --init init/text_row0.pt --data data/kokoro --held 20 --steps 3000 --bs 16 --eval-every 100000 --max-minutes 6 --out ckpt/textB1', 'train_text_B1', W)
txtB = sh(f'{PY} scripts/ms_train.py text --init ckpt/textB1/last.pt --data data/teacher --held {HELD} --steps 4000 --bs 16 --lr 2e-4 --eval-every 250 --patience 8 --score dur_asr --max-minutes 10 --out ckpt/textB', 'train_text_B2', W)
sh(f'{PY} scripts/export_paradee_v1_1.py ckpt/dec/last.pt --text ckpt/textA/last.pt --out models/A', 'export_A', W)
sh(f'{PY} scripts/export_paradee_v1_1.py ckpt/dec/last.pt --text ckpt/textB/last.pt --out models/B', 'export_B', W)
for t in ('A', 'B'):
    shutil.copy(f'{W}/models/{t}/paradee_int8.onnx', f'{OUT}/ms_{t}_int8.onnx')
    shutil.copy(f'{W}/ckpt/text{t}/last.pt', f'{OUT}/text{t}_last.pt')
shutil.copy(f'{W}/ckpt/dec/last.pt', f'{OUT}/dec_last.pt')

# ---------------- scoring ----------------
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, f'{W}/scripts')
sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import alignment  # noqa: E402
from transformers import pipeline  # noqa: E402
from vhead import VDecoder  # noqa: E402

rows = torch.load(f'{W}/data/teacher/shard_0000.pt')
asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3', torch_dtype=torch.float16, device=0)


def hear(a, sr=24000):
    w = torchaudio.functional.resample(torch.from_numpy(np.asarray(a, dtype=np.float32)), sr, 16000).numpy()
    return asr({'raw': w[:16000 * 29], 'sampling_rate': 16000}, batch_size=1,
               generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})['text'].strip()


def clean(s):
    return re.sub(r'[^a-z]', '', normalize(s).lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return min(1.0, lev(clean(ref), clean(hyp)) / max(1, len(clean(ref))))


def bright(a, sr=24000):
    S = np.abs(librosa.stft(np.asarray(a, dtype=np.float32), n_fft=1024, hop_length=256)) + 1e-9
    P = S ** 2
    tot = P.sum(0)
    keep = tot > np.percentile(tot, 40)
    f = np.fft.rfftfreq(1024, 1 / sr)
    band = S[(f > 2000) & (f < 6000)][:, keep]
    return [float((P[f > 4000][:, keep].sum(0) / tot[keep]).mean()), float((P[(f > 1500) & (f < 4000)][:, keep].sum(0) / tot[keep]).mean()),
            float((np.exp(np.log(band).mean(0)) / band.mean(0)).mean())]


def stats(v):
    v = np.array(v)
    return dict(n=len(v), median=round(float(np.median(v)), 3), capped_mean=round(float(v.mean()), 3), share_le_8=round(float((v <= .08).mean()), 2))


held = rows[:HELD]
res = dict(decoder_only={}, full={})
os.makedirs(f'{OUT}/audio', exist_ok=True)
for name, ck in (('k09_decoder', K09_DEC), ('new_decoder', f'{W}/ckpt/dec/last.pt')):
    vd = VDecoder(head='v2', dim=160, blocks=4).to('cuda').eval()
    vd.load_ckpt(ck, 'cuda')
    c, b = [], []
    for k, r in enumerate(held):
        aln = alignment(r['pred_dur'])
        Fr = aln.shape[1]
        with torch.no_grad():
            torch.manual_seed(0)
            y = vd((r['t_en'].float() @ aln)[None].cuda(), r['F0'][None, :2 * Fr].cuda(), r['N'][None, :2 * Fr].cuda()).squeeze().cpu().numpy()
        c.append(cer(r['text'], hear(y)))
        b.append(bright(y))
        if k < 12:
            sf.write(f'{OUT}/audio/dec_{name}_{r["id"]}.wav', y, 24000)
    res['decoder_only'][name] = dict(cer=stats(c), brightness=dict(over_4k=float(np.mean([x[0] for x in b])), mid=float(np.mean([x[1] for x in b])), flat=float(np.mean([x[2] for x in b]))))
    del vd
tc, tb = [], []
for r in held:
    a = r['audio'].float().numpy() / 32767
    tc.append(cer(r['text'], hear(a)))
    tb.append(bright(a))
res['decoder_only']['chatterbox_clip'] = dict(cer=stats(tc), brightness=dict(over_4k=float(np.mean([x[0] for x in tb])), mid=float(np.mean([x[1] for x in tb])), flat=float(np.mean([x[2] for x in tb]))))

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
t200 = pd.read_csv(f'{DS}/test_200.tsv', sep='\t', dtype=str, keep_default_na=False).head(40)
per = []
for tag in ('A', 'B'):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(f'{W}/models/{tag}/paradee_int8.onnx', so, providers=['CPUExecutionProvider'])
    ch, ct, bb = [], [], []
    for k, r in enumerate(held):
        y = sess.run(None, {'input_ids': r['ids'][None].numpy(), 'speed': np.ones(1, np.float32)})[0][0]
        h = hear(y)
        ch.append(cer(r['text'], h))
        bb.append(bright(y))
        per.append(dict(model=tag, split='held', id=r['id'], text=r['text'], cer=ch[-1], heard=h))
        if k < 12:
            sf.write(f'{OUT}/audio/{tag}_held_{r["id"]}.wav', y, 24000)
    for k, r in enumerate(t200.itertuples()):
        ref = normalize(r.text)
        y = sess.run(None, {'input_ids': np.array([g2p.tokens(g2p(ref))], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]
        h = hear(y)
        ct.append(cer(ref, h))
        per.append(dict(model=tag, split='test', id=r.id, text=ref, cer=ct[-1], heard=h))
        if k < 12:
            sf.write(f'{OUT}/audio/{tag}_test_{r.id}.wav', y, 24000)
    res['full'][tag] = dict(held=stats(ch), test=stats(ct), brightness=dict(over_4k=float(np.mean([x[0] for x in bb])), mid=float(np.mean([x[1] for x in bb])), flat=float(np.mean([x[2] for x in bb]))))
pd.DataFrame(per).to_csv(f'{OUT}/v5_scores.csv', index=False)


def last(s, key):
    return next((json.loads(l.split(':', 1)[1]) for l in reversed(s.splitlines()) if l.startswith(key)), None)


res.update(stages=STAGES, total_s=round(time.time() - T0), decoder_train=last(dec, 'after:'),
           text_A=last(txtA, 'after:'), text_B=last(txtB, 'after:'),
           rows_line=[l for l in rows_out.splitlines() if l.startswith(('rows', 'words'))],
           sizes_mb=round(os.path.getsize(f'{OUT}/ms_A_int8.onnx') / 1e6, 2))
json.dump(res, open(f'{OUT}/v5_report.json', 'w'), indent=1)
print(json.dumps(res, indent=1), flush=True)
shutil.rmtree(f'{W}/data', ignore_errors=True)
