# kokoro-malay-bella / continue.plan.v7.md C4 part 2 (kmb-16-train-c4).
# Inputs: kmb-15-cbgen2 output (cb_clips2.pt, chosen.csv, gen_plan.csv) as a dataset,
#   kmb-rows-v02 (old 742-row shard + teacher_clips.csv), kokoro-malay-g2p, kmb-paradee-ms scripts.
# 1. Unpack the new clips, build rows with ms_rows.py --letters (same recipe as k12).
# 2. Training set = old rows cleaned by the C2 rule (CER>15% or rate outliers dropped, held 60 kept)
#    + new rows cleaned the same way (CER from chosen.csv + rate percentiles over the merged set).
# 3. Train decoder 12,000 steps and text side 10,000 steps (--score dur_asr), export int8.
# 4. Score three models on the same runs in one session for an apples-to-apples comparison:
#    v0.2 (shipped), k14 clean model, and the C4 model - all 200 test rows + held 60.
# 5. C4 gate (fixed in the plan): weak-category median CER down >= 25% relative vs v0.2.
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
HELD = 60
PY = sys.executable


def sh(cmd, tag, cwd=W):
    print(f'== {tag}: {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    open(f'{OUT}/log_{tag}.txt', 'w').write(p.stdout + '\n---\n' + p.stderr)
    tail = '\n'.join(l for l in (p.stdout + p.stderr).splitlines()
                     if not re.search(r'Warning|warn|%\||Xet|FutureWarning|UserWarning', l))
    print(tail[-2500:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f'{tag} failed rc={p.returncode}')
    return p.stdout


os.makedirs(W, exist_ok=True)
DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_rows.py', recursive=True)[0])
ROWS_OLD = glob.glob('/kaggle/input/**/shard_0000.pt', recursive=True)[0]
TEACH = glob.glob('/kaggle/input/**/teacher_clips.csv', recursive=True)[0]
CLIPS_NEW = glob.glob('/kaggle/input/**/cb_clips2.pt', recursive=True)[0]
CHOSEN = glob.glob('/kaggle/input/**/chosen.csv', recursive=True)[0]
PLAN = glob.glob('/kaggle/input/**/gen_plan.csv', recursive=True)[0]
V02 = glob.glob('/kaggle/input/**/ms_A_int8_gpu.onnx', recursive=True)
V02 = V02[0] if V02 else None
V_CLEAN = glob.glob('/kaggle/input/**/ms_clean_int8.onnx', recursive=True)
V_CLEAN = V_CLEAN[0] if V_CLEAN else None

shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
os.makedirs(f'{W}/data/merged', exist_ok=True)
os.makedirs(f'{OUT}/audio', exist_ok=True)
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
from transformers import pipeline  # noqa: E402

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
t200 = pd.read_csv(f'{DS}/test_200.tsv', sep='\t', dtype=str, keep_default_na=False)
rows_old = torch.load(ROWS_OLD, map_location='cpu', weights_only=False)
teach = pd.read_csv(TEACH, dtype={'id': str})

# ---------------- new clips -> rows ----------------
clips = torch.load(CLIPS_NEW, map_location='cpu', weights_only=False)
os.makedirs('/tmp/newclips', exist_ok=True)
for i, a in clips.items():
    sf.write(f'/tmp/newclips/{i}.wav', a.numpy().astype('float32') / 32767, 24000)
chosen = pd.read_csv(CHOSEN, dtype={'id': str})
plan = pd.read_csv(PLAN, dtype=str, keep_default_na=False)
n_chosen = len(chosen)
chosen = chosen[chosen.cer <= 0.15]
tsv = plan[plan.id.isin(set(chosen.id))][['id', 'category', 'text', 'teacher']]
tsv.to_csv(f'{W}/new.tsv', sep='\t', index=False)
print('new clips:', len(clips), 'of', n_chosen, 'chosen; rows to build:', len(tsv), flush=True)
os.makedirs(f'{W}/data/new', exist_ok=True)
sh(f'{PY} scripts/ms_rows.py --clips /tmp/newclips --tsv new.tsv --g2p {DS} '
   f'--out data/new/shard_0000.pt --device cuda --letters', 'rows_new', W)
rows_new = torch.load(f'{W}/data/new/shard_0000.pt', map_location='cpu', weights_only=False)
print('new rows:', len(rows_new), flush=True)

# ---------------- clean both sets with the C2 rule ----------------
cers_old = dict(zip(teach.id, teach.cer))
rates_old = dict(zip(teach.id, teach.rate))


def rate_of(r):
    return len(r['ps']) / max(len(r['audio']) / 24000, 1e-6)


all_rates = list(rates_old.values()) + [rate_of(r) for r in rows_new]
lo, hi = np.percentile(all_rates, 2), np.percentile(all_rates, 98)
old_keep = [r for r in rows_old[HELD:]
            if cers_old.get(r['id'], 0) <= 0.15 and lo <= rates_old.get(r['id'], 0) <= hi]
new_keep = [r for r in rows_new if lo <= rate_of(r) <= hi]
merged = rows_old[:HELD] + old_keep + new_keep
print(f'merged rows: {len(merged)} (held 60 + old {len(old_keep)} of {len(rows_old) - HELD} + new {len(new_keep)} of {len(rows_new)})', flush=True)
torch.save(merged, f'{W}/data/merged/shard_0000.pt')

# ---------------- train + export ----------------
sh(f'{PY} scripts/ms_train.py dec --init init/dec_row0.pt --data data/merged --held {HELD} '
   f'--steps 12000 --bs 16 --max-minutes 32 --out ckpt/dec', 'train_decoder', W)
sh(f'{PY} scripts/ms_train.py text --init init/text_row0.pt --data data/merged --held {HELD} '
   f'--steps 10000 --bs 16 --eval-every 500 --score dur_asr --max-minutes 32 --out ckpt/textA', 'train_text', W)
# the exporter peeks at data/teacher/shard_*.pt for a held-out sanity check: point it at the merged shard
os.makedirs(f'{W}/data/teacher', exist_ok=True)
shutil.copy(f'{W}/data/merged/shard_0000.pt', f'{W}/data/teacher/shard_0000.pt')
os.makedirs(f'{OUT}/rows_merged', exist_ok=True)
shutil.copy(f'{W}/ckpt/dec/last.pt', f'{OUT}/dec_c4_last.pt')
shutil.copy(f'{W}/ckpt/textA/last.pt', f'{OUT}/textA_c4_last.pt')
shutil.copy(f'{W}/data/merged/shard_0000.pt', f'{OUT}/rows_merged/shard_0000.pt')
sh(f'{PY} scripts/export_paradee_v1_1.py ckpt/dec/last.pt --text ckpt/textA/last.pt --out models/C4', 'export_C4', W)
# WebGPU has no float64 (inlined scripts/webgpu_patch.py: that file lives in the repo, not in
# paradee_ms, so it is absent from this kernel's scripts/ copy).
import onnx  # noqa: E402
from onnx import TensorProto as T, numpy_helper  # noqa: E402

m = onnx.load(f'{W}/models/C4/paradee_int8.onnx')
BIG = 2 ** 31 - 1
for n in m.graph.node:
    for a in n.attribute:
        if a.name == 'to' and a.i == T.DOUBLE:
            a.i = T.FLOAT
        elif a.t is not None and a.t.data_type == T.DOUBLE:
            a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32), a.t.name))
        elif a.t is not None and a.t.data_type == T.INT64:
            arr = numpy_helper.to_array(a.t)
            if (np.abs(arr) >= 2 ** 31).any():
                a.t.CopyFrom(numpy_helper.from_array(np.clip(arr, -BIG, BIG).astype(np.int64), a.t.name))
for i in list(m.graph.initializer):
    if i.data_type == T.DOUBLE:
        m.graph.initializer.remove(i)
        m.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(i).astype(np.float32), i.name))
    elif i.data_type == T.INT64:
        arr = numpy_helper.to_array(i)
        if (np.abs(arr) >= 2 ** 31).any():
            m.graph.initializer.remove(i)
            m.graph.initializer.append(numpy_helper.from_array(np.clip(arr, -BIG, BIG).astype(np.int64), i.name))
cons = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in m.graph.node
        if n.op_type == 'Constant' and n.attribute and n.attribute[0].t is not None}
prod = {o: n for n in m.graph.node for o in n.output}
for n in m.graph.node:
    if n.op_type == 'Slice' and len(n.input) >= 5 and n.input[4] in cons and (cons[n.input[4]] < 0).any():
        src = prod.get(n.input[0])
        if src is not None and src.op_type == 'RandomNormalLike':
            first = n.input[0]
            del n.input[:]
            n.input.append(first)
            del n.attribute[:]
            n.op_type = 'Identity'
del m.graph.value_info[:]
onnx.checker.check_model(m)
onnx.save(m, f'{W}/models/C4/paradee_int8_webgpu.onnx')
print('webgpu patch applied (inline)', flush=True)
shutil.copy(f'{W}/models/C4/paradee_int8_webgpu.onnx', f'{OUT}/ms_c4_int8.onnx')
shutil.copy(f'{W}/ckpt/dec/last.pt', f'{OUT}/dec_c4_last.pt')
shutil.copy(f'{W}/ckpt/textA/last.pt', f'{OUT}/textA_c4_last.pt')
os.makedirs(f'{OUT}/rows_merged', exist_ok=True)
shutil.copy(f'{W}/data/merged/shard_0000.pt', f'{OUT}/rows_merged/shard_0000.pt')

# ---------------- score v0.2, k14-clean and C4 on the same rows ----------------
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


def stats(v):
    v = np.array(v)
    return dict(n=len(v), median=round(float(np.median(v)), 3), capped_mean=round(float(v.mean()), 3))


so = ort.SessionOptions()
so.intra_op_num_threads = 4
sess = {'c4': ort.InferenceSession(f'{OUT}/ms_c4_int8.onnx', so, providers=['CPUExecutionProvider'])}
if V02:
    sess['v02'] = ort.InferenceSession(V02, so, providers=['CPUExecutionProvider'])
if V_CLEAN:
    sess['clean'] = ort.InferenceSession(V_CLEAN, so, providers=['CPUExecutionProvider'])

test_items = [(r.id, r.text, r.category, g2p.tokens(g2p(normalize(r.text)))) for r in t200.itertuples()]
held_items = [(r['id'], r['text'], 'held', r['ids'].numpy().tolist()) for r in rows_old[:HELD]]
out_rows = []
for tag, s in sess.items():
    for iid, text, cat, ids in test_items + held_items:
        y = s.run(None, {'input_ids': np.array([ids], dtype=np.int64), 'speed': np.ones(1, np.float32)})[0][0]
        h = hear(y)
        out_rows.append(dict(model=tag, id=iid, category=cat, text=text, cer=cer(text, h), heard=h))
    print('scored', tag, flush=True)
    if tag == 'c4':
        sf.write(f'{OUT}/audio/c4_c0000.wav', y, 24000)
pd.DataFrame(out_rows).to_csv(f'{OUT}/test_scores_c4.csv', index=False)

res = {}
for tag in sess:
    d = pd.DataFrame([r for r in out_rows if r['model'] == tag])
    res[tag] = dict(overall=stats(d[d.category != 'held'].cer),
                    per_category={c: stats(g.cer) for c, g in d[d.category != 'held'].groupby('category')},
                    held=stats(d[d.category == 'held'].cer))

# ---------------- gates ----------------
weak = ['general', 'codeswitch']
g = {}
for c in weak:
    b = res['v02']['per_category'].get(c, {}).get('median')
    n4 = res['c4']['per_category'].get(c, {}).get('median')
    g[c] = dict(v02=b, c4=n4, rel=round(1 - n4 / b, 3) if b else None, pass_=bool(b and n4 is not None and 1 - n4 / b >= 0.25))
g['overall_better'] = bool(res['c4']['overall']['capped_mean'] <= res['v02']['overall']['capped_mean'])
g['pass'] = bool(g['overall_better'] and all(g[c]['pass_'] for c in weak))
json.dump(dict(models=res, gates=g, runtime_min=round((time.time() - T0) / 60, 1)),
          open(f'{OUT}/c4_report.json', 'w'), indent=1)
print('C4 GATES', json.dumps(g, indent=1), flush=True)
print('runtime', round((time.time() - T0) / 60, 1), 'min', flush=True)
