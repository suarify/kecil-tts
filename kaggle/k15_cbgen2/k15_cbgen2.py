# kokoro-malay-bella / continue.plan.v7.md C4: more training speech (kmb-15-cbgen2, v2).
# 1. Sentences: the 398 rows of cb_1100.tsv that kmb-08 never generated, then 400 new rows from
#    corpus_train.tsv weighted to the weak categories: general 200, code-switch 120 (ranked by how many
#    lexicon_en English words the sentence carries, template-deduped), numbers 80.
# 2. Generation: Chatterbox Multilingual (same settings as kmb-08) -> raw wavs written straight into
#    /kaggle/working/raw2 so a later crash never loses audio. One seed for leftovers/numbers, TWO seeds
#    (1234, 5678) for new general + code-switch sentences.
# 3. Whisper large-v3 scores every variant with a DIRECT model call: the ASR pipeline breaks here because
#    chatterbox-tts upgrades transformers and the pipeline dict input then wants torchcodec/FFmpeg
#    (v1 died there after generating all 1,118 clips). Better seed wins; two-seed sentences are dropped
#    when BOTH variants exceed 15% CER.
# 4. Voice conversion toward Bella -> cb_clips2.pt (chunks every 100 clips), cb_raw2.pt (chosen raws,
#    for the package D second voice), chosen.csv, dropped.csv, gen_report.json.
# 5. Splits sentences across two workers when two GPUs are visible (v1 measured 1,118 clips in 102 min
#    on 2 T4s); workers skip wavs already on disk (resume).
import glob
import json
import os
import re
import subprocess
import sys
import time

T0 = time.time()
OUT = '/kaggle/working'
RAW = f'{OUT}/raw2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
GEN_CAP_MIN = float(os.environ.get('GEN_CAP_MIN', 390))
VC_CAP_MIN = float(os.environ.get('VC_CAP_MIN', 90))


def sh(cmd, tag):
    print(f'== {tag}: {cmd}', flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    open(f'{OUT}/log_{tag}.txt', 'w').write(p.stdout + '\n---\n' + p.stderr)
    print((p.stdout + p.stderr)[-1200:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f'{tag} failed rc={p.returncode}')


os.makedirs(RAW, exist_ok=True)
sh('pip install -q chatterbox-tts', 'pip_cb')
sh('pip uninstall -y -q torchvision || true', 'pip_tv')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

DS = os.path.dirname(glob.glob('/kaggle/input/**/cb_1100.tsv', recursive=True)[0])
ROWS_CSV = glob.glob('/kaggle/input/**/teacher_clips.csv', recursive=True)[0]
REF_MS = glob.glob('/kaggle/input/**/refs/ref_ms.wav', recursive=True)[0]
REF_EN = glob.glob('/kaggle/input/**/refs/ref_en.wav', recursive=True)[0]
LEX = json.load(open(glob.glob('/kaggle/input/**/lexicon_en.json', recursive=True)[0], encoding='utf-8'))
json.dump(dict(ref_ms=REF_MS, ref_en=REF_EN, sr=24000), open(f'{OUT}/refs.json', 'w'))

gpus = [l for l in subprocess.run('nvidia-smi --query-gpu=index --format=csv,noheader',
                                  shell=True, capture_output=True, text=True).stdout.splitlines() if l.strip()]
print('GPUs visible:', len(gpus), flush=True)

# ---------------- sentence plan ----------------
cb = pd.read_csv(f'{DS}/cb_1100.tsv', sep='\t', dtype=str, keep_default_na=False)
done = set(pd.read_csv(ROWS_CSV, dtype=str)['id'])
left = cb[~cb['id'].isin(done)].copy()
left['source'], left['seeds'] = 'leftover', '1234'

corpus = pd.read_csv(f'{DS}/corpus_train.tsv', sep='\t', dtype=str, keep_default_na=False)
used = set(cb['text'].str.strip())
cand = corpus[~corpus['text'].str.strip().isin(used)].copy()
cand['nw'] = cand['text'].str.split().str.len()
cand = cand[(cand.nw >= 4) & (cand.nw <= 30) & (cand['text'].str.len() <= 140)]
cand = cand[~cand['text'].str.contains(r'http|www\.|References|Message-ID|\[\d+\]', regex=True, na=False)]


def stride(df, n):
    if len(df) <= n:
        return df
    step = len(df) / n
    return df.iloc[[int(i * step) for i in range(n)]]


gen_rows = []
gs = cand[cand.category == 'general']
gen_rows.append(stride(gs.sample(frac=1.0, random_state=7), 200).assign(source='new', seeds='1234;5678'))
cs = cand[cand.category == 'codeswitch'].copy()
cs['nlex'] = cs['text'].apply(lambda t: sum(w.lower() in LEX for w in t.split()))
cs = cs.sort_values('nlex', ascending=False).head(400)
cs['tmpl'] = cs['text'].str.lower().str.split().str[:4].str.join(' ')
cs = cs.groupby('tmpl', group_keys=False).head(2).sample(frac=1.0, random_state=11)
gen_rows.append(stride(cs, 120).assign(source='new', seeds='1234;5678'))
nb = cand[cand.category == 'numbers']
gen_rows.append(stride(nb, 80).assign(source='new', seeds='1234'))

plan = pd.concat([left[['id', 'category', 'text', 'teacher', 'source', 'seeds']]] + gen_rows, ignore_index=True)[
    ['id', 'category', 'text', 'teacher', 'source', 'seeds']]
plan.to_csv(f'{OUT}/gen_plan.csv', index=False)
print('plan:', len(plan), 'sentences |', plan.category.value_counts().to_dict(),
      '| generations:', sum(len(str(s).split(';')) for s in plan.seeds), flush=True)

# ---------------- workers (one GPU each) ----------------
open(f'{OUT}/gen_worker.py', 'w').write('''
import json, os, sys, time
import pandas as pd
import torch
OUT = '/kaggle/working'
RAW = f'{OUT}/raw2'
part, nparts = int(sys.argv[1]), int(sys.argv[2])
plan = pd.read_csv(f'{OUT}/gen_plan.csv', dtype=str, keep_default_na=False).to_dict('records')
mine = plan[part::nparts]
refs = json.load(open(f'{OUT}/refs.json'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
tts = ChatterboxMultilingualTTS.from_pretrained(device='cuda')
os.makedirs(RAW, exist_ok=True)
t0 = time.time()
n = skip = 0
for k, r in enumerate(mine):
    if (time.time() - t0) / 60 > float(os.environ.get('GEN_CAP_MIN', 390)):
        print('GEN CAP at', k, flush=True)
        break
    for seed in str(r['seeds']).split(';'):
        key = r['id'] + '#' + seed
        path = f'{RAW}/{key}.wav'
        if os.path.exists(path):
            skip += 1
            continue
        torch.manual_seed(int(seed))
        try:
            wav = tts.generate(r['teacher'], language_id='ms', audio_prompt_path=refs['ref_ms'],
                               exaggeration=0.5, cfg_weight=0.3, temperature=0.8)
        except Exception as e:
            print('FAIL', key, repr(e)[:140], flush=True)
            continue
        a = wav.squeeze(0).float().cpu().numpy()
        sf.write(path, a, tts.sr)
        n += 1
        if n % 50 == 0:
            print(f'gen part {part}: {n} new clips | {(time.time() - t0) / 60:.1f} min', flush=True)
print('part', part, 'done: generated', n, 'skipped existing', skip, flush=True)
''')

nproc = max(1, min(len(gpus), 2))
procs = [subprocess.Popen([sys.executable, f'{OUT}/gen_worker.py', str(i), str(nproc)],
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))) for i in range(nproc)]
for p in procs:
    p.wait()
if any(p.returncode != 0 for p in procs):
    raise RuntimeError('a generator worker failed')

wavs = sorted(glob.glob(f'{RAW}/*.wav'))
print('raw wavs on disk:', len(wavs), flush=True)
json.dump({os.path.basename(w)[:-4]: sf.info(w).duration for w in wavs}, open(f'{OUT}/raw_seconds.json', 'w'))

# ---------------- Whisper large-v3, direct call ----------------
import torchaudio  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

sys.path.insert(0, DS)
from ms_norm import normalize  # noqa: E402

wp = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3',
                                                     torch_dtype=torch.float16).to('cuda').eval()


def transcribe(a):
    f = wp(a, sampling_rate=16000, return_tensors='pt').input_features.to('cuda', dtype=torch.float16)
    with torch.no_grad():
        try:
            ids = wm.generate(f, language='ms', task='transcribe', max_new_tokens=200)
        except Exception:
            ids = wm.generate(f, language='malay', task='transcribe', max_new_tokens=200)
    return wp.batch_decode(ids, skip_special_tokens=True)[0].strip()


def clean(s):
    return re.sub(r'[^a-z]', '', normalize(s).lower())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for k, ca in enumerate(a, 1):
        cur = [k]
        for j, cb2 in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb2)))
        prev = cur
    return prev[-1]


cer = lambda ref, hyp: min(1.0, lev(clean(ref), clean(hyp)) / max(1, len(clean(ref))))

plan_by_id = {r['id']: r for r in plan.to_dict('records')}
scores = []
for k, w in enumerate(wavs):
    key = os.path.basename(w)[:-4]
    iid, seed = key.split('#')
    a, sr = sf.read(w, dtype='float32')
    a16 = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()[:16000 * 29]
    heard = transcribe(a16)
    scores.append(dict(key=key, id=iid, seed=int(seed), category=plan_by_id[iid]['category'],
                       seconds=len(a) / sr, cer=cer(plan_by_id[iid]['text'], heard), heard=heard))
    if k % 100 == 0:
        print('score', k, '/', len(wavs), flush=True)
del wm
torch.cuda.empty_cache()

by_id = {}
for s in scores:
    by_id.setdefault(s['id'], []).append(s)
chosen, dropped = [], []
for i, ss in by_id.items():
    best = min(ss, key=lambda x: x['cer'])
    if ';' in str(plan_by_id[i]['seeds']) and best['cer'] > 0.15:
        dropped.append(dict(id=i, best_cer=round(best['cer'], 3)))
        continue
    chosen.append(best)
print('chosen', len(chosen), 'dropped (both seeds >15% CER)', len(dropped), flush=True)

# chosen raw audio (package D second voice) + all raws as a safety net
raw_all = {}
for w in wavs:
    a, sr = sf.read(w, dtype='float32')
    raw_all[os.path.basename(w)[:-4]] = torch.from_numpy((np.clip(a, -1, 1) * 32767).astype(np.int16))
torch.save(raw_all, f'{OUT}/cb_raw_all.pt')
torch.save({c['id']: raw_all[c['key']] for c in chosen}, f'{OUT}/cb_raw2.pt')
del raw_all

# ---------------- voice conversion toward Bella ----------------
from chatterbox.vc import ChatterboxVC  # noqa: E402

vc = ChatterboxVC.from_pretrained('cuda')
clips = {}
t_vc = time.time()
for k, c in enumerate(chosen):
    if (time.time() - t_vc) / 60 > VC_CAP_MIN:
        print('VC cap at', k, flush=True)
        break
    try:
        w = vc.generate(audio=f'{RAW}/{c["key"]}.wav', target_voice_path=REF_EN)
    except Exception as e:
        print('VC FAIL', c['id'], repr(e)[:120], flush=True)
        continue
    a = w.squeeze(0).float().cpu().numpy()
    clips[c['id']] = torch.from_numpy((np.clip(a, -1, 1) * 32767).astype(np.int16))
    if k % 100 == 99:
        torch.save(clips, f'{OUT}/cb_clips2.pt')
        print(f'vc {k + 1}/{len(chosen)} | {(time.time() - t_vc) / 60:.1f} min', flush=True)
torch.save(clips, f'{OUT}/cb_clips2.pt')

pd.DataFrame([dict(id=c['id'], category=c['category'], seeds=plan_by_id[c['id']]['seeds'],
                   cer=round(c['cer'], 3), winner_seed=c['seed'], heard=c['heard'])
              for c in chosen]).to_csv(f'{OUT}/chosen.csv', index=False)
pd.DataFrame(dropped).to_csv(f'{OUT}/dropped.csv', index=False)
pd.DataFrame(scores).to_csv(f'{OUT}/scores_all.csv', index=False)

cat = pd.DataFrame(chosen)
report = dict(sentences=len(plan), wavs=len(wavs), chosen=len(chosen), converted=len(clips),
              dropped_15=len(dropped), gpus=len(gpus), workers=nproc,
              per_category=cat.groupby('category').agg(n=('id', 'size'), median_cer=('cer', 'median'),
                                                       mean_cer=('cer', 'mean')).round(3).to_dict('index'),
              audio_min=round(sum(s['seconds'] for s in scores) / 60, 1),
              runtime_min=round((time.time() - T0) / 60, 1))
json.dump(report, open(f'{OUT}/gen_report.json', 'w'), indent=1)
print(json.dumps(report, indent=1), flush=True)
