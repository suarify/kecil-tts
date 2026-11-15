"""Assemble the Hugging Face upload folders under hf/ from the repo:  python scripts/build_hf.py --onnx PATH --version 0.1 [--samples-from runs/k09]

hf/model          the model repo   <user>/suaradee-v<version>
hf/space_web      static Space     (transformers.js demo, WebGPU with WASM fallback)
hf/space_gradio   Gradio Space     (onnxruntime on CPU)
"""
import argparse
import json
import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--onnx', required=True)
ap.add_argument('--version', default='0.1')
ap.add_argument('--user', default='lseanlon')
ap.add_argument('--samples-from', default='runs/k09')
ap.add_argument('--card', default=os.path.join(ROOT, 'hf', 'model_card_v0.1.md'))
A = ap.parse_args()
REPO = f'{A.user}/suaradee-v{A.version}'
G2P_FILES = ['lexicon_e.json', 'lexicon_en.json', 'lexicon_exceptions.json', 'vocab.json']


def fresh(p):
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(os.path.join(ROOT, src) if not os.path.isabs(src) else src, dst)


def py_package(dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(os.path.join(ROOT, 'python', 'suaradee')):
        if f.endswith(('.py', '.json')):
            shutil.copy(os.path.join(ROOT, 'python', 'suaradee', f), os.path.join(dst, f))


def js_front(dst):
    os.makedirs(dst, exist_ok=True)
    for f in ['ms_norm.js', 'ms_g2p.js'] + G2P_FILES:
        shutil.copy(os.path.join(ROOT, 'g2p', f), os.path.join(dst, f))
    shutil.copy(os.path.join(ROOT, 'g2p', 'package.json'), os.path.join(dst, 'package.json'))


# ---------------- model repo ----------------
M = os.path.join(ROOT, 'hf', 'model')
fresh(M)
copy(A.onnx, os.path.join(M, 'onnx', 'model_quantized.onnx'))
copy('export/ms_bella_8m/tokenizer.json', os.path.join(M, 'tokenizer.json'))
json.dump({'model_type': 'style_text_to_speech_2', 'architectures': ['StyleTextToSpeech2Model'], 'sample_rate': 24000, 'n_token': 178,
           'voices': ['default'], 'note': 'Single voice. Inputs: input_ids (int64 [1,T], phoneme ids with 0 at both ends, T <= 402) and speed (float32 [1]). '
                                          'No style input. Phonemes come from the Malay front end in suaradee/ (Python) or js/ (JavaScript).',
           'g2p': {'stress': False, 'r': 'r', 'diph': 'glide', 'final_a': 'a'}}, open(os.path.join(M, 'config.json'), 'w'), indent=1)
py_package(os.path.join(M, 'suaradee'))
js_front(os.path.join(M, 'js', 'g2p'))
copy('web/suaradee.js', os.path.join(M, 'js', 'suaradee.js'))
copy('python/pyproject.toml', os.path.join(M, 'pyproject.toml'))
copy('hf/LICENSE', os.path.join(M, 'LICENSE'))
copy('hf/NOTICE', os.path.join(M, 'NOTICE'))
for _w in os.listdir(os.path.join(ROOT, 'hf', 'wheel')):
    copy(os.path.join('hf', 'wheel', _w), os.path.join(M, 'dist', _w))
copy('hf/example.py', os.path.join(M, 'examples', 'python_example.py'))
copy('hf/example_node.mjs', os.path.join(M, 'examples', 'node_example.mjs'))
shutil.copy(A.card, os.path.join(M, 'README.md'))
copy('hf/space_gradio/app.py', os.path.join(M, 'examples', 'gradio_app.py'))

# samples from the exported model (never edited): 3 call-centre sentences it got right, 3 typical held-out ones
os.makedirs(os.path.join(M, 'samples'), exist_ok=True)
lines = []
v5 = os.path.join(ROOT, A.samples_from, 'v5_scores.csv')
if os.path.exists(v5):                                  # v0.2 layout (kernel k12): model A is the shipped one
    d5 = pd.read_csv(v5)
    d5 = d5[d5.model == 'A']
    audio_dir = os.path.join(ROOT, A.samples_from, 'audio')
    have = lambda r: os.path.exists(os.path.join(audio_dir, f'A_{r.split}_{r.id}.wav'))                 # noqa: E731
    cand = d5[[have(r) for r in d5.itertuples()]]
    good = cand[(cand.split == 'test') & (cand.cer <= 0.05)].sort_values('id').iloc[[0, 2, 4]]
    typ = cand[cand.split == 'held'].sort_values('cer')
    typ = typ.iloc[[len(typ) // 3, len(typ) // 2, 2 * len(typ) // 3]]
    for k, r in enumerate(pd.concat([good, typ]).itertuples(), 1):
        shutil.copy(os.path.join(audio_dir, f'A_{r.split}_{r.id}.wav'), os.path.join(M, 'samples', f'{k}_suaradee.wav'))
        lines.append(f'{k}\t{r.split}\tCER {r.cer:.0%}\t{r.text}')
    open(os.path.join(M, 'samples', 'sentences.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
else:
    sc = pd.read_csv(os.path.join(ROOT, A.samples_from, 'e2e_scores.csv'))
    good = sc[(sc.split == 'test') & (sc.cer_onnx <= 0.08)].sort_values('id').iloc[[0, 2, 4]]
    typ = sc[sc.split == 'held'].sort_values('cer_onnx')
    typ = typ.iloc[[len(typ) // 3, len(typ) // 2, 2 * len(typ) // 3]]
    for k, r in enumerate(pd.concat([good, typ]).itertuples(), 1):
        shutil.copy(os.path.join(ROOT, A.samples_from, 'audio', f'{r.split}_{r.id}_onnx_int8.wav'), os.path.join(M, 'samples', f'{k}_suaradee.wav'))
        lines.append(f'{k}\t{r.split}\tCER {r.cer_onnx:.0%}\t{r.text}')
    open(os.path.join(M, 'samples', 'sentences.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')


# ---------------- static Space ----------------
S = os.path.join(ROOT, 'hf', 'space_web')
os.makedirs(S, exist_ok=True)
for f in os.listdir(S):
    if f not in ('index.html',):
        p = os.path.join(S, f)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
copy('web/suaradee.js', os.path.join(S, 'suaradee.js'))
js_front(os.path.join(S, 'g2p'))
open(os.path.join(S, 'README.md'), 'w', encoding='utf-8').write(f"""---
title: Suaradee (web)
emoji: 🔊
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: apache-2.0
short_description: 8.5 MB Malay text-to-speech running in your browser
models:
  - lseanlon/suaradee-v0.3
  - lseanlon/suaradee-v0.2
  - lseanlon/suaradee-v0.1
---
Malay text to speech in the browser with transformers.js and ONNX (WebGPU when available, otherwise WASM).
Models: [v0.3](https://huggingface.co/lseanlon/suaradee-v0.3), [v0.2](https://huggingface.co/lseanlon/suaradee-v0.2) and [v0.1](https://huggingface.co/lseanlon/suaradee-v0.1).
""")

# ---------------- Gradio Space ----------------
Gd = os.path.join(ROOT, 'hf', 'space_gradio')
os.makedirs(Gd, exist_ok=True)
app = open(os.path.join(Gd, 'app.py'), encoding='utf-8').read().replace('lseanlon/suaradee-v0.1', REPO).replace('v0.1', f'v{A.version}')
open(os.path.join(Gd, 'app.py'), 'w', encoding='utf-8').write(app)
shutil.rmtree(os.path.join(Gd, 'suaradee'), ignore_errors=True)
py_package(os.path.join(Gd, 'suaradee'))
open(os.path.join(Gd, 'requirements.txt'), 'w').write('onnxruntime>=1.16\nnumpy\nhuggingface_hub\n')
open(os.path.join(Gd, 'README.md'), 'w', encoding='utf-8').write(f"""---
title: Suaradee {A.version} (Gradio)
emoji: 🔊
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: 8.5 MB Malay text-to-speech on a CPU
models:
  - {REPO}
---
Malay text to speech with onnxruntime on CPU. Model: [{REPO}](https://huggingface.co/{REPO}).
""")
print('built', REPO, '| model repo MB:', round(sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(M) for f in fs) / 1e6, 2))
