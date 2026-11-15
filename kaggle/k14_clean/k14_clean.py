# kokoro-malay-bella / kmb-14-clean: package C, stages C0 + C1 + C2 (continue.plan.v7.md).
# C0: the shipped v0.2 ONNX scored on ALL 200 test sentences, per category (the baseline every later
#     stage is compared against), plus the Chatterbox teacher clips scored per clip.
# C2: drop bad teacher clips (CER > 15%, speaking-rate outliers), retrain decoder + text side A with the
#     k12 recipe on the cleaned set, export int8, score held 60 + all 200 test rows per category.
# C1: learning curve - decoder trained on 25% / 50% / 100% of the train rows, decoder-only CER on held 60,
#     decides whether C4 (more generated speech) is worth 3.5 GPU-h per 1,000 clips.
# Gates (fixed): C2 is kept only if capped-mean CER improves against the C0 baseline; C4 runs only if the
#     curve improves >= 15% relative from 50% to 100%.
# Inputs: seanlon/kokoro-malay-g2p (tsv + g2p), seanlon/kmb-paradee-ms (scripts), seanlon/kmb-rows-v02
#     (rows/shard_0000.pt from kmb-12-v5 + the shipped ms_A_int8_gpu.onnx). Internet on (Paradee init,
#     Whisper large-v3).
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

T0 = time.time()
OUT = '/kaggle/working'
W = f'{OUT}/e2e'
DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_rows.py', recursive=True)[0])
ROWS_IN = glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0]
V02 = glob.glob('/kaggle/input/**/ms_A_int8_gpu.onnx', recursive=True)
V02 = V02[0] if V02 else glob.glob('/kaggle/input/**/model_quantized.onnx', recursive=True)[0]
HELD = 60
PY = sys.executable


def sh(cmd, tag, cwd=W):
    print(f'== {tag}: {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    tail = '\n'.join(l for l in (p.stdout + p.stderr).splitlines()
                     if not re.search(r'Warning|warn|%\||Xet|FutureWarning|UserWarning', l))
    open(f'{OUT}/log_{tag}.txt', 'w').write(p.stdout + '\n---\n' + p.stderr)
    print(tail[-3000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f'{tag} failed rc={p.returncode}')
    return p.stdout


os.makedirs(W, exist_ok=True)
shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
os.makedirs(f'{W}/data/teacher', exist_ok=True)
shutil.copy(ROWS_IN, f'{W}/data/teacher/shard_0000.pt')
sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru onnx onnxruntime', 'pip_misc')
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')
sh(f'{PY} scripts/ms_init.py init', 'init_weights', W)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, f'{W}/scripts')
sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import alignment  # noqa: E402
from transformers import pipeline  # noqa: E402
from vhead import VDecoder  # noqa: E402

rows = torch.load(f'{W}/data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)
print('rows', len(rows), flush=True)
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
t200 = pd.read_csv(f'{DS}/test_200.tsv', sep='\t', dtype=str, keep_default_na=False)

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)


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


def synth_onnx(sess, ids):
    return sess.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]


def score_model(sess, items, out_rows, tag, save_audio=None):
    """items: list of (id, text, category|None, ids). Appends (tag, id, category, cer, heard) to out_rows."""
    cers = []
    for k, (iid, text, cat, ids) in enumerate(items):
        y = synth_onnx(sess, ids)
        h = hear(y)
        c = cer(text, h)
        cers.append(c)
        out_rows.append(dict(model=tag, id=iid, category=cat, text=text, cer=c, heard=h))
        if save_audio and k < save_audio:
            sf.write(f'{OUT}/audio/{tag}_{iid}.wav', y, 24000)
    return cers


# ---------------- C0: baseline v0.2 on all 200 test sentences ----------------
os.makedirs(f'{OUT}/audio', exist_ok=True)
so = ort.SessionOptions()
so.intra_op_num_threads = 4
base_sess = ort.InferenceSession(V02, so, providers=['CPUExecutionProvider'])
sc, sc_rows = [], []
for cat in t200.category.unique():
    sub = t200[t200.category == cat]
    items = [(r.id, r.text, r.category, g2p.tokens(g2p(normalize(r.text)))) for r in sub.itertuples()]
    v = score_model(base_sess, items, sc_rows, 'v02')
    sc.append(dict(category=cat, **stats(v)))
baseline = dict(overall=stats([r['cer'] for r in sc_rows]), per_category=sc)
print('C0 baseline v0.2:', json.dumps(baseline), flush=True)
json.dump(baseline, open(f'{OUT}/baseline_v02.json', 'w'), indent=1)

# ---------------- teacher clip quality (feeds C2 cleaning and C3 word lists) ----------------
# cached scan from a previous version skips ~30 min of Whisper transcriptions
_tcsv = glob.glob('/kaggle/input/**/teacher_clips.csv', recursive=True)
if _tcsv:
    teach_rows = pd.read_csv(_tcsv[0], dtype={'id': str}).to_dict('records')
    print('teacher scan: LOADED CACHED', _tcsv[0], len(teach_rows), flush=True)
else:
    teach_rows = []
    for r in rows:
        a = r['audio'].float().numpy() / 32767
        h = hear(a)
        c = cer(r['text'], h)
        sec = len(a) / 24000
        teach_rows.append(dict(id=r['id'], text=r['text'], cer=c, heard=h, seconds=round(sec, 2),
                               rate=len(r['ps']) / sec))
pd.DataFrame(teach_rows).to_csv(f'{OUT}/teacher_clips.csv', index=False)
cers_t = np.array([r['cer'] for r in teach_rows])
rates = np.array([r['rate'] for r in teach_rows])
lo, hi = np.percentile(rates, 2), np.percentile(rates, 98)
drop_cer = cers_t > 0.15
drop_rate = (rates < lo) | (rates > hi)
drop = drop_cer | drop_rate
keep_ids = {teach_rows[i]['id'] for i in np.where(~drop)[0]}
print(f'teacher clips: {int(drop_cer.sum())} above 15% CER, {int(drop_rate.sum())} rate outliers, '
      f'{int(drop.sum())} dropped of {len(rows)}', flush=True)

clean_rows = [r for i, r in enumerate(rows) if r['id'] in keep_ids or i < HELD]
n_held = len([r for r in clean_rows if r['id'] in {x['id'] for x in rows[:HELD]}])
print('clean set:', len(clean_rows), 'rows (held kept:', n_held, ')', flush=True)
os.makedirs(f'{W}/data/teacher_clean', exist_ok=True)
torch.save(clean_rows, f'{W}/data/teacher_clean/shard_0000.pt')
os.makedirs(f'{OUT}/rows_clean', exist_ok=True)
shutil.copy(f'{W}/data/teacher_clean/shard_0000.pt', f'{OUT}/rows_clean/shard_0000.pt')

# ---------------- C1: learning curve (decoder on 25 / 50 / 100 % of train rows) ----------------
train_all = rows[HELD:]
curve = {}
for frac in (0.25, 0.5, 1.0):
    tag = f'c{int(frac * 100)}'
    if frac == 1.0:
        data_dir = 'data/teacher'
    else:
        sub = rows[:HELD] + train_all[:int(round(frac * len(train_all)))]
        data_dir = f'data/curve_{tag}'
        os.makedirs(f'{W}/{data_dir}', exist_ok=True)
        torch.save(sub, f'{W}/{data_dir}/shard_0000.pt')
    sh(f'{PY} scripts/ms_train.py dec --init init/dec_row0.pt --data {data_dir} --held {HELD} '
       f'--steps 6000 --bs 16 --max-minutes 12 --out ckpt/curve_{tag}', f'curve_{tag}', W)
    vd = VDecoder(head='v2', dim=160, blocks=4).to('cuda').eval()
    vd.load_ckpt(f'{W}/ckpt/curve_{tag}/last.pt', 'cuda')
    c = []
    for r in rows[:HELD]:
        aln = alignment(r['pred_dur'])
        Fr = aln.shape[1]
        with torch.no_grad():
            torch.manual_seed(0)
            y = vd((r['t_en'].float() @ aln)[None].cuda(), r['F0'][None, :2 * Fr].cuda(),
                   r['N'][None, :2 * Fr].cuda()).squeeze().cpu().numpy()
        c.append(cer(r['text'], hear(y)))
    del vd
    torch.cuda.empty_cache()
    curve[tag] = stats(c)
    print(f'curve {tag}: {curve[tag]}', flush=True)
gain = 1 - curve['c50']['median'] / max(1e-9, curve['c100']['median'])
data_lever = gain >= 0.15
print(f'C1: 50%->100% median gain {gain:.0%}; data is the lever: {data_lever}', flush=True)

# ---------------- C2: retrain on the cleaned set (k12 recipe) ----------------
dec = sh(f'{PY} scripts/ms_train.py dec --init init/dec_row0.pt --data data/teacher_clean --held {HELD} '
         f'--steps 6000 --bs 16 --max-minutes 12 --out ckpt/dec', 'train_decoder_clean', W)
txtA = sh(f'{PY} scripts/ms_train.py text --init init/text_row0.pt --data data/teacher_clean --held {HELD} '
          f'--steps 6000 --bs 16 --eval-every 100000 --max-minutes 12 --out ckpt/textA', 'train_text_clean', W)
sh(f'{PY} scripts/export_paradee_v1_1.py ckpt/dec/last.pt --text ckpt/textA/last.pt --out models/A', 'export_A', W)
shutil.copy(f'{W}/models/A/paradee_int8.onnx', f'{OUT}/ms_clean_int8.onnx')
shutil.copy(f'{W}/ckpt/dec/last.pt', f'{OUT}/dec_clean_last.pt')
shutil.copy(f'{W}/ckpt/textA/last.pt', f'{OUT}/textA_clean_last.pt')

clean_sess = ort.InferenceSession(f'{OUT}/ms_clean_int8.onnx', so, providers=['CPUExecutionProvider'])
test_items = [(r.id, r.text, r.category, g2p.tokens(g2p(normalize(r.text)))) for r in t200.itertuples()]
held_items = [(r['id'], r['text'], None, r['ids'].numpy().tolist()) for r in rows[:HELD]]
cl, cl_rows = [], []
for cat in t200.category.unique():
    v = score_model(clean_sess, [x for x in test_items if x[2] == cat], cl_rows, 'clean')
    cl.append(dict(category=cat, **stats(v)))
held_c = score_model(clean_sess, held_items, cl_rows, 'clean_held')
clean_res = dict(overall=stats([r['cer'] for r in cl_rows if r['model'] == 'clean']),
                 per_category=cl, held=stats(held_c))
print('C2 clean:', json.dumps(clean_res), flush=True)
pd.DataFrame(sc_rows + cl_rows).to_csv(f'{OUT}/test_scores.csv', index=False)

# full student on held 60 also gives brightness for the smear watch
b0 = [bright(synth_onnx(base_sess, x[3])) for x in test_items[:60]]
b1 = [bright(synth_onnx(clean_sess, x[3])) for x in test_items[:60]]
clean_res['brightness'] = dict(v02_over4k=float(np.mean([x[0] for x in b0])), clean_over4k=float(np.mean([x[0] for x in b1])))

# ---------------- gates ----------------
def gate_per_cat():
    base = {d['category']: d for d in sc}
    out = []
    for d in cl:
        b = base[d['category']]
        out.append(dict(category=d['category'], v02=d['capped_mean'], clean=d['capped_mean'],
                        better=d['capped_mean'] <= b['capped_mean']))
    return out


gates = dict(
    keep_clean=bool(clean_res['overall']['capped_mean'] <= baseline['overall']['capped_mean']),
    no_cat_worse=all(g['better'] for g in gate_per_cat()),
    per_category=gate_per_cat(),
    c1_data_lever=bool(data_lever), c1_gain=round(float(gain), 3),
)
json.dump(dict(baseline=baseline, clean=clean_res, curve=curve, gates=gates,
               runtime_s=round(time.time() - T0)),
          open(f'{OUT}/report.json', 'w'), indent=1)
print('GATES', json.dumps(gates, indent=1), flush=True)
print('runtime', round(time.time() - T0), 's', flush=True)
