# kokoro-malay-bella / continue.plan.md Phase A1: search for a zero-training Malay teacher.
# Blend af_bella with non-English Kokoro voices (acoustic half cols 0:128, prosody half 128:256
# blended separately), try G2P conventions, score CER + accent distance + similarity.
# Accent distance = DTW cosine distance between Whisper-large-v3 ENCODER features of the clip and
# of the Chatterbox reference clip for the same sentence (lower = closer to the Malay clip).
import gc
import glob
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import types
import zipfile

T0 = time.time()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def sh(cmd):
    print('$', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh('pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"')
sh('pip install -q loguru speechbrain')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402


def _stub(name):
    m = types.ModuleType(name)

    def _ga(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return type(attr, (), {})
    m.__getattr__ = _ga
    return m


sys.modules['misaki'] = _stub('misaki')
for sub in ('en', 'espeak'):
    sys.modules['misaki.' + sub] = _stub('misaki.' + sub)
    setattr(sys.modules['misaki'], sub, sys.modules['misaki.' + sub])

from huggingface_hub import hf_hub_download  # noqa: E402
from kokoro import KModel  # noqa: E402

DS = os.path.dirname(glob.glob('/kaggle/input/**/ms_g2p.py', recursive=True)[0])
REF = os.path.dirname(glob.glob('/kaggle/input/**/ref_32.tsv', recursive=True)[0])
sys.path.insert(0, DS)
from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
print('DS', DS, 'REF', REF, flush=True)
print('ref contents', sorted(os.listdir(REF)), flush=True)

OUT = '/kaggle/working'
os.makedirs(f'{OUT}/voices', exist_ok=True)
SR = 24000
DEV = 'cuda'
model = KModel(repo_id='hexgrad/Kokoro-82M').to(DEV).eval()


def voice(name):
    return torch.load(hf_hub_download('hexgrad/Kokoro-82M', f'voices/{name}.pt'), weights_only=True)


def synth(ph, pack):
    ph = ph[:510]
    with torch.no_grad():
        a = model(ph, pack[len(ph) - 1].to(DEV), 1.0)
    return a.float().cpu().numpy()


def blend(B, D, wa, wp):
    return torch.cat([wa * B[..., :128] + (1 - wa) * D[..., :128],
                      wp * B[..., 128:] + (1 - wp) * D[..., 128:]], -1)


BELLA = voice('af_bella')
DONOR_NAMES = ['ef_dora', 'if_sara', 'pf_dora', 'hf_alpha', 'hf_beta', 'jf_alpha', 'ff_siwis', 'zf_xiaoxiao']
DONORS = {}
for n in DONOR_NAMES:
    try:
        DONORS[n] = voice(n)
    except Exception as e:
        print('missing donor', n, e, flush=True)
if all(k in DONORS for k in ('ef_dora', 'if_sara', 'hf_alpha')):
    DONORS['mix3'] = (DONORS['ef_dora'] + DONORS['if_sara'] + DONORS['hf_alpha']) / 3
print('donors', list(DONORS), flush=True)


def rd(name):
    return pd.read_csv(os.path.join(DS, name), sep='\t', dtype=str, keep_default_na=False)


ref32 = pd.read_csv(os.path.join(REF, 'ref_32.tsv'), sep='\t', dtype=str, keep_default_na=False)
test40 = rd('test_40.tsv')
etest = rd('e_test.tsv')

gen = ref32[ref32.category == 'general']
cs = ref32[ref32.category == 'codeswitch']
pick = lambda df, k: df.iloc[[int(i * (len(df) - 1) / max(k - 1, 1)) for i in range(k)]]  # noqa: E731
search = pd.concat([pick(gen, 12), pick(cs, 4)])          # the 16 search sentences
nums = test40[test40.category == 'numbers']
listen_nums = pick(nums, 3)

# ---------------- scoring models ----------------
from transformers import pipeline  # noqa: E402

asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3',
               torch_dtype=torch.float16, device=0)
FE = asr.feature_extractor
ENC = asr.model.get_encoder()

from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402

ecapa = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir='/tmp/ecapa',
                                       run_opts={'device': 'cuda:0'})


def to16(a, sr):
    t = torch.from_numpy(np.asarray(a, dtype=np.float32))
    if t.ndim > 1:
        t = t.mean(-1)
    if sr != 16000:
        t = torchaudio.functional.resample(t, sr, 16000)
    return t.numpy()


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


def transcribe(wavs):
    batch = [{'raw': to16(w, SR)[:16000 * 29], 'sampling_rate': 16000} for w in wavs]
    out = asr(batch, batch_size=4,
              generate_kwargs={'language': 'malay', 'task': 'transcribe', 'max_new_tokens': 200})
    return [o['text'].strip() for o in out]


@torch.no_grad()
def enc_feats(wavs16):
    """Whisper encoder features, frames at 40 ms (every 2nd of 20 ms), trimmed to the clip length."""
    res = []
    for i in range(0, len(wavs16), 4):
        chunk = wavs16[i:i + 4]
        f = FE([w[:16000 * 29] for w in chunk], sampling_rate=16000, return_tensors='pt').input_features
        h = ENC(f.to(DEV, torch.float16)).last_hidden_state.float()
        for w, x in zip(chunk, h):
            n = min(1500, math.ceil(min(len(w), 16000 * 29) / 320))
            x = torch.nn.functional.normalize(x[:n:2], dim=-1)
            res.append(x.cpu().numpy())
    return res


def dtw_cos(a, b):
    c = 1.0 - a @ b.T
    n, m = c.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], c[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return float(D[n, m] / (n + m))


@torch.no_grad()
def spk(wavs16):
    out = []
    for w in wavs16:
        e = ecapa.encode_batch(torch.from_numpy(w).unsqueeze(0).to('cuda:0')).squeeze()
        out.append(torch.nn.functional.normalize(e, dim=0).cpu())
    return out


bella_clips = sorted(glob.glob(os.path.join(REF, '**', 'bella_en', '*.wav'), recursive=True))
def load16(p):
    a, sr = sf.read(p, dtype='float32')
    return to16(a, sr)


CENTROID = torch.nn.functional.normalize(torch.stack(spk([load16(p) for p in bella_clips])).mean(0), dim=0)

# Chatterbox reference clips (T2 = untouched Malay reference clip, T2v = voice-converted toward Bella)
ref_wav = {}
for sub in ('T2', 'T2v'):
    for i in ref32.id:
        ps = glob.glob(os.path.join(REF, '**', sub, f'{i}.wav'), recursive=True)
        if ps:
            a, sr = sf.read(ps[0], dtype='float32')
            ref_wav[(sub, i)] = to16(a, sr)
REF_FEATS = {}
keys = list(ref_wav)
for k, f in zip(keys, enc_feats([ref_wav[k] for k in keys])):
    REF_FEATS[k] = f
print('reference clips', len(ref_wav), flush=True)

# ---------------- evaluation of one config ----------------
G2P_CACHE = {}


def g2p_for(stress, r, final_a):
    key = (stress, r, final_a)
    if key not in G2P_CACHE:
        G2P_CACHE[key] = G2P(stress=stress, r=r, diph='glide', final_a=final_a)
    return G2P_CACHE[key]


def run_config(name, pack, conv, sents, ref_key='T2', keep_wavs=None):
    """sents: DataFrame with id,text,category. Returns list of row dicts."""
    g = g2p_for(*conv)
    wavs, rows = [], []
    for r in sents.itertuples():
        norm = normalize(r.text)
        ph = g(norm)
        a = synth(ph, pack)
        wavs.append(a)
        rows.append(dict(config=name, id=r.id, category=r.category, norm=norm, n_ph=len(ph),
                         seconds=len(a) / SR))
    hyps = transcribe(wavs)
    w16 = [to16(a, SR) for a in wavs]
    feats = enc_feats(w16)
    sims = spk(w16)
    for row, h, f, s in zip(rows, hyps, feats, sims):
        row['hyp'] = h
        row['cer'] = lev(clean(row['norm']), clean(h)) / max(1, len(clean(row['norm'])))
        row['sim'] = float(s @ CENTROID)
        row['rate'] = row['n_ph'] / row['seconds']
        k = (ref_key, row['id'])
        row['acc'] = dtw_cos(f, REF_FEATS[k]) if k in REF_FEATS else float('nan')
    if keep_wavs is not None:
        keep_wavs.update({(name, r['id']): w for r, w in zip(rows, wavs)})
    return rows


def summarise(df):
    return df.groupby('config').agg(cer=('cer', 'mean'), acc=('acc', 'mean'), sim=('sim', 'mean'),
                                    rate=('rate', 'mean')).reset_index()


BASE_CONV = (False, 'r', 'a')       # stress, r, final_a
all_rows = []

# ---- accent-score sanity check: stock Bella vs the voice-converted Chatterbox clip, both against T2 ----
b0_rows = run_config('B0', BELLA, BASE_CONV, ref32)
all_rows += b0_rows
b0 = pd.DataFrame(b0_rows).set_index('id')
chk = []
for i in ref32.id:
    kv = ('T2v', i)
    if kv in REF_FEATS and ('T2', i) in REF_FEATS:
        chk.append(dtw_cos(REF_FEATS[kv], REF_FEATS[('T2', i)]) < b0.loc[i, 'acc'])
ACC_OK = float(np.mean(chk)) if chk else 0.0
print(f'accent-score check: T2v closer to T2 than stock Bella on {ACC_OK:.0%} of {len(chk)}', flush=True)
B0_ACC, B0_CER = float(b0.loc[search.id, 'acc'].mean()), float(b0.loc[search.id, 'cer'].mean())
print('B0 (16 search sentences): cer', round(B0_CER, 4), 'acc', round(B0_ACC, 4), flush=True)

# ---- stage 1: voice grid ----
configs = [('B0s', BELLA)]
for dn, D in DONORS.items():
    configs.append((f'{dn}_pure', D))
    for wa, wp in itertools.product((0.7, 0.5, 0.3), (0.5, 0.0)):
        configs.append((f'{dn}_a{wa}_p{wp}', blend(BELLA, D, wa, wp)))
VOICES = dict(configs)
print('stage 1 configs', len(configs), flush=True)
s1 = []
for k, (name, pack) in enumerate(configs):
    s1 += run_config(name, pack, BASE_CONV, search)
    if k % 8 == 0:
        print(f'  stage1 {k}/{len(configs)} {time.time() - T0:.0f}s', flush=True)
        pd.DataFrame(s1).to_csv(f'{OUT}/search_stage1_partial.csv', index=False)
s1df = pd.DataFrame(s1)
s1df.to_csv(f'{OUT}/search_stage1.csv', index=False)
sum1 = summarise(s1df)
sum1['donor'] = sum1.config.str.replace(r'_(pure|a[\d.]+_p[\d.]+)$', '', regex=True)
ok = sum1[(sum1.cer <= B0_CER + 0.02) & (sum1.config != 'B0s')].sort_values('acc')
top6, per = [], {}
for r in ok.itertuples():
    if per.get(r.donor, 0) >= 2:
        continue
    top6.append(r.config)
    per[r.donor] = per.get(r.donor, 0) + 1
    if len(top6) == 6:
        break
print('stage 1 top6:', top6, flush=True)
print(sum1.sort_values('acc').head(15).round(4).to_string(), flush=True)

# ---- stage 2: phoneme conventions for each of the top 6 ----
s2 = []
for name in top6:
    for stress, r, fa in itertools.product((False, True), ('r', 'ɾ'), ('a', 'ə')):
        s2 += run_config(f'{name}|s{int(stress)}_{r}_{fa}', VOICES[name], (stress, r, fa), search)
s2df = pd.DataFrame(s2)
s2df.to_csv(f'{OUT}/search_stage2.csv', index=False)
sum2 = summarise(s2df)
sum2['blend'] = sum2.config.str.split('|').str[0]
sum2['conv'] = sum2.config.str.split('|').str[1]
best = []
for b in top6:
    cand = sum2[(sum2.blend == b) & (sum2.cer <= B0_CER + 0.03)]
    if cand.empty:
        cand = sum2[sum2.blend == b]
    best.append(cand.sort_values(['acc', 'cer']).iloc[0])
best = sorted(best, key=lambda r: r.acc)
NF = 4 if ACC_OK >= 0.8 else 6
finalists = best[:NF]
print('finalists:', [(f.blend, f.conv, round(f.acc, 4), round(f.cer, 4)) for f in finalists], flush=True)


def parse_conv(c):
    m = re.match(r's(\d)_(.)_(.)', c)
    return (bool(int(m.group(1))), m.group(2), m.group(3))


# ---- finalists on the full listening / scoring sets ----
listen = pd.concat([ref32, listen_nums])
fin_rows, clips = [], {}
fin_info = []
for i, f in enumerate(finalists):
    name = f'F{i + 1}'
    conv = parse_conv(f.conv)
    torch.save(VOICES[f.blend].clone(), f'{OUT}/voices/{name}.pt')
    fin_info.append(dict(name=name, blend=f.blend, conv=f.conv, stress=conv[0], r=conv[1], final_a=conv[2],
                         acc_search=float(f.acc), cer_search=float(f.cer), sim_search=float(f.sim)))
    fin_rows += run_config(name, VOICES[f.blend], conv, listen, keep_wavs=clips)
    fin_rows += run_config(name + '_t40', VOICES[f.blend], conv, test40, ref_key='none')
    fin_rows += run_config(name + '_et', VOICES[f.blend], conv, etest.rename(columns={'word': 'id'}).assign(category='etest'), ref_key='none')
# alt final_a of the top finalist, to hear that one difference alone
top = finalists[0]
tc = parse_conv(top.conv)
alt = (tc[0], tc[1], 'ə' if tc[2] == 'a' else 'a')
fin_rows += run_config('F1_alt', VOICES[top.blend], alt, listen, keep_wavs=clips)
fin_info.append(dict(name='F1_alt', blend=top.blend, conv=f's{int(alt[0])}_{alt[1]}_{alt[2]}'))
# stock Bella clips for the page
fin_rows += run_config('B0', BELLA, BASE_CONV, listen, keep_wavs=clips)

fdf = pd.DataFrame(fin_rows)
fdf.to_csv(f'{OUT}/search_finalists.csv', index=False)
with zipfile.ZipFile(f'{OUT}/clips_finalists.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for (name, i), w in clips.items():
        p = f'/tmp/{name}_{i}.wav'
        sf.write(p, w, SR)
        z.write(p, f'{name}/{i}.wav')
json.dump(fin_info, open(f'{OUT}/finalists.json', 'w'), indent=1, ensure_ascii=False)

main = fdf[~fdf.config.str.contains('_t40|_et')]
t40 = fdf[fdf.config.str.endswith('_t40')]
report = {
    'accent_check_fraction': ACC_OK, 'b0_search_cer': B0_CER, 'b0_search_acc': B0_ACC,
    'stage1_top6': top6, 'finalists': fin_info,
    'finalist_summary': summarise(main).round(4).to_dict(orient='records'),
    'test40_cer': {c: round(float(g.cer.mean()), 4) for c, g in t40.groupby('config')},
    'runtime_s': round(time.time() - T0),
}
json.dump(report, open(f'{OUT}/search_report.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
