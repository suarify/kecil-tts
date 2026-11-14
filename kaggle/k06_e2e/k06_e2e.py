# kokoro-malay-bella / v4 end-to-end smoke test on Kaggle (target: 30 min).
# 32 Chatterbox voice-converted clips (already on Kaggle as dataset kmb-ref) ->
#   rows (MMS alignment, pYIN pitch, mel loudness, Kokoro text features)
#   -> decoder fine-tune -> text-side fine-tune -> ONNX export (fp32 + int8)
#   -> free-running text->audio with the int8 file -> Whisper-large-v3 CER.
# The point is plumbing and a first quality signal, not quality: 28 training clips.
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
    print(tail[-3500:], flush=True)
    if name:
        STAGES[name] = round(time.time() - t, 1)
        print(f'--- stage {name}: {STAGES[name]} s (total {time.time() - T0:.0f} s)', flush=True)
    if p.returncode != 0:
        json.dump(dict(failed=name or cmd, stages=STAGES, tail=p.stdout[-3000:]), open(f'{OUT}/e2e_report.json', 'w'), indent=1)
        raise SystemExit(f'FAILED: {cmd}')
    return p.stdout


sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"', 'pip_kokoro')
sh('pip install -q loguru onnx onnxruntime', 'pip_misc')

# misaki stub as real files so subprocesses see it (package __init__ must import its submodules)
stubs = f'{OUT}/stubs/misaki'
os.makedirs(stubs, exist_ok=True)
for name in ('en', 'espeak'):
    open(f'{stubs}/{name}.py', 'w').write('def __getattr__(name):\n    return type(name, (), {})\n')
open(f'{stubs}/__init__.py', 'w').write('from . import en, espeak\n')
os.environ['PYTHONPATH'] = f'{OUT}/stubs:' + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, f'{OUT}/stubs')

DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
REF = os.path.dirname(glob.glob('/kaggle/input/**/ref_32.tsv', recursive=True)[0])
PMS = os.path.dirname(glob.glob('/kaggle/input/**/ms_rows.py', recursive=True)[0])
CLIPS = os.path.dirname(glob.glob(os.path.join(REF, '**', 'T2v', '*.wav'), recursive=True)[0])
print('DS', DS, '\nREF', REF, '\nPMS', PMS, '\nCLIPS', CLIPS, flush=True)

os.makedirs(W, exist_ok=True)
shutil.copytree(PMS, f'{W}/scripts', dirs_exist_ok=True)
import pandas as pd  # noqa: E402

tsv = pd.read_csv(f'{REF}/ref_32.tsv', sep='\t', dtype=str, keep_default_na=False).sample(frac=1, random_state=0)
tsv.to_csv(f'{W}/ref_shuffled.tsv', sep='\t', index=False)

HELD = 4
PY = sys.executable
sh(f'{PY} scripts/ms_init.py init', 'init_weights', W)
sh(f'{PY} scripts/ms_rows.py --clips {CLIPS} --tsv ref_shuffled.tsv --g2p {DS} --out data/teacher/shard_0000.pt --device cuda', 'rows', W)
dec = sh(f'{PY} scripts/ms_train.py dec --init init/dec_row0.pt --held {HELD} --steps 1500 --bs 16 --max-minutes 9 --out ckpt/dec', 'train_decoder', W)
txt = sh(f'{PY} scripts/ms_train.py text --init init/text_row0.pt --held {HELD} --steps 800 --bs 16 --max-minutes 4 --out ckpt/text', 'train_text', W)
exp = sh(f'{PY} scripts/export_paradee_v1_1.py ckpt/dec/last.pt --text ckpt/text/last.pt --out models/ms', 'export', W)

# ---------------- score the exported int8 model, free-running text -> audio ----------------
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

sys.path.insert(0, DS)
from ms_norm import normalize  # noqa: E402

rows = torch.load(f'{W}/data/teacher/shard_0000.pt')
so = ort.SessionOptions()
so.intra_op_num_threads = 1
sess = ort.InferenceSession(f'{W}/models/ms/paradee_int8.onnx', so, providers=['CPUExecutionProvider'])
os.makedirs(f'{OUT}/audio', exist_ok=True)
picked = list(range(HELD)) + list(range(HELD, min(len(rows), HELD + 4)))        # 4 held-out + 4 trained-on
gen, rt = {}, []
for i in picked:
    r = rows[i]
    t = time.perf_counter()
    y = sess.run(None, {'input_ids': r['ids'][None].numpy(), 'speed': np.ones(1, np.float32)})[0][0]
    rt.append((len(y) / 24000) / (time.perf_counter() - t))
    gen[i] = y
    sf.write(f"{OUT}/audio/{'held' if i < HELD else 'train'}_{r['id']}_onnx_int8.wav", y, 24000)
    sf.write(f"{OUT}/audio/{'held' if i < HELD else 'train'}_{r['id']}_target.wav", r['audio'].float().numpy() / 32767, 24000)
for f in glob.glob(f'{W}/ckpt/dec/held_*.wav'):
    shutil.copy(f, f'{OUT}/audio/' + os.path.basename(f))
shutil.copy(f'{W}/models/ms/paradee_int8.onnx', f'{OUT}/ms_int8.onnx')
shutil.copy(f'{W}/models/ms/paradee.onnx', f'{OUT}/ms_fp32.onnx')

from transformers import pipeline  # noqa: E402

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


res = []
for i in picked:
    r = rows[i]
    ref = normalize(r['text'])
    tgt = hear(r['audio'].float().numpy() / 32767)
    out = hear(gen[i])
    res.append(dict(id=r['id'], split='held' if i < HELD else 'train', text=ref, cer_target=lev(clean(ref), clean(tgt)) / len(clean(ref)),
                    cer_onnx=lev(clean(ref), clean(out)) / len(clean(ref)), heard=out, sec=len(gen[i]) / 24000))
df = pd.DataFrame(res)
df.to_csv(f'{OUT}/e2e_scores.csv', index=False)
by = df.groupby('split')[['cer_target', 'cer_onnx']].mean().round(3)
print(by.to_string(), flush=True)
print(df[['id', 'split', 'cer_onnx', 'text', 'heard']].to_string(max_colwidth=60), flush=True)

parse = lambda s, key: next((json.loads(l.split(':', 1)[1]) for l in s.splitlines() if l.startswith(key)), None)  # noqa: E731
rep = dict(stages=STAGES, total_s=round(time.time() - T0),
           rows=json.load(open(f'{W}/data/teacher/shard_0000_report.json')),
           decoder=dict(before=parse(dec, 'before:'), after=parse(dec, 'after:')),
           text=dict(before=parse(txt, 'before:'), after=parse(txt, 'after:')),
           onnx_realtime_x=round(float(np.mean(rt)), 1), cer_by_split=json.loads(by.to_json(orient='index')),
           sizes_mb=dict(int8=round(os.path.getsize(f'{OUT}/ms_int8.onnx') / 1e6, 2), fp32=round(os.path.getsize(f'{OUT}/ms_fp32.onnx') / 1e6, 2)),
           export_lines=[l for l in exp.splitlines() if 'check' in l or 'MB' in l or 'realtime' in l])
json.dump(rep, open(f'{OUT}/e2e_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1), flush=True)
shutil.rmtree(f'{W}/data', ignore_errors=True)
