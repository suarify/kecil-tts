---
license: apache-2.0
language:
- ms
library_name: transformers.js
pipeline_tag: text-to-speech
base_model: sahilmahendrakar/Paradee-8M-v1.3
tags:
- text-to-speech
- tts
- onnx
- transformers.js
- webgpu
- malay
- bahasa-melayu
- tiny
---

# Suaradee v0.4

A **Malay text-to-speech model that is 8.5 MB**. One ONNX file, one voice, runs in a browser tab (transformers.js, WebGPU or WASM) or on a single CPU thread in Python.

**v0.4 is trained on roughly twice the speech of [v0.3](https://huggingface.co/lseanlon/suaradee-v0.3)** (1,281 aligned sentences, about 118 minutes, instead of 634 / 49 minutes). General text improves by another 8% relative, the overall error is the lowest of any version, and the held-out score is the best so far. **The muffled sound is unchanged** (see Limitations).

- **Demo in the browser (nothing leaves your device):** [Suaradee web Space](https://huggingface.co/spaces/lseanlon/suaradee-web) (pick v0.4 in the version box)
- **Gradio demo:** `examples/gradio_app.py` (`pip install gradio`, then `python examples/gradio_app.py`)

| | |
|---|---|
| Model file | `onnx/model_quantized.onnx`, **8.46 MB** (int8 weights) |
| Parameters | 7.9 M |
| Output | 24 kHz mono |
| Speed | about 25x real time on one CPU thread (onnxruntime); about 15x in a browser tab on WebGPU (6x on the first call) |
| Language | Bahasa Melayu (Malaysia), with common English words used in Malaysian speech |
| Inputs | `input_ids` (int64 `[1,T]`, phoneme ids with a 0 at each end, `T` up to 402) and `speed` (float32 `[1]`) |

## Quick start

**Python** (`onnxruntime` + `numpy`; no PyTorch):

```bash
pip install onnxruntime numpy huggingface_hub soundfile
pip install https://huggingface.co/lseanlon/suaradee-v0.4/resolve/main/dist/suaradee-0.4.0-py3-none-any.whl
```
```python
import soundfile as sf
from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained("lseanlon/suaradee-v0.4")
sf.write("hello.wav", tts("Selamat pagi, boleh saya bantu encik hari ini?"), SAMPLE_RATE)
```

**Browser or Node** (transformers.js 3.x):

```js
import { Suaradee } from './js/suaradee.js';        // see the web Space for a complete page
const tts = await Suaradee.load({ model: 'lseanlon/suaradee-v0.4', g2pBase: './js/g2p/' });   // WebGPU if available, else WASM
const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
```

`examples/` has runnable Python and Node scripts. The ONNX file is WebGPU-safe: no float64 operators (the pitch-phase accumulator runs in float32), clamped 64-bit slice bounds, and the pointless noise reversal removed. Voice conversion to a second speaker only needs the pitch/loudness path, which this export keeps intact.

## How good is it?

Measured by playing the model's speech to Whisper large-v3 and comparing what it heard with the text (character error rate, CER; lower is better). All three versions were scored **in the same session on the same 200 test sentences**, so the comparison has no cross-run noise. The recogniser itself misreads about 9% of the characters of the training clips, so no model scores 0; treat differences under about 1.5 points as noise.

**All 200 test sentences** (median / capped mean CER):

| Category | n | v0.2 | v0.3 | **v0.4** |
|---|---|---|---|---|
| General text (Wikipedia-style, names, places) | 70 | 9.9% / 13.1% | 6.4% / 11.2% | **5.9% / 10.2%** |
| Call-centre style (appointments, bills, delivery) | 60 | 5.2% / 6.5% | 4.9% / 7.3% | **4.3% / 6.2%** |
| Numbers and money | 30 | 5.0% / 6.5% | **0% / 5.8%** | 2.2% / 6.0% |
| Malay with English words | 20 | 18.4% / 23.9% | 19.2% / 26.2% | **17.9% / 24.9%** |
| Questions | 10 | 1.7% / 3.3% | **0% / 2.4%** | 0% / 4.3% |
| Short phrases | 10 | 13.5% / 15.9% | **0% / 15.4%** | 4.0% / 10.2% |
| **Overall** | **200** | **7.5% / 10.8%** | **5.9% / 10.5%** | **5.5% / 9.5%** |

On the separate 60-sentence held-out set: **v0.4 4.8% / 8.5%**, v0.2 4.9% / 8.9%, v0.3 5.6% / 10.3%. v0.4 is the best of the three on the held-out set, which is the number that matters most for unseen text.

Listen to `samples/` (three sentences it got right, three typical held-out ones; `samples/sentences.txt` lists the text and score of each).

## Limitations (read before using)

- **Muffled sound, unchanged since v0.1.** High-frequency energy is the same as v0.3 (measured on 24 sentences: 0.0125 vs 0.0129 of total energy above 4 kHz), with the same damp/phasey quality. It comes from the decoder's training, not from timing or data volume; a no-training repair was tested and rejected (the shipped sound was preferred 8 times out of 8).
- **English words are still the weak spot**: sentences that mix Malay and English score about 18%. Doubling the training speech did **not** move this number, so the limit is the English text/pronunciation side, not the amount of audio. English words are only handled when they are in a list of about 430 common ones.
- Individual sentences are noisy: comparing v0.4 with v0.3 row by row, 103 got better, 62 stayed the same and 95 got worse; the net effect is the aggregate improvement above.
- **One synthetic female voice**, no emotion or style control.
- **Written for short sentences.** Long text is split at sentence ends and sent in pieces of at most 400 phonemes.
- Numbers, amounts, dates, times and phone numbers are read out in Malay by a rule-based normaliser.
- It is synthetic speech. Please say so when you use it.

## What changed since v0.3

v0.3 dropped the 120 of 742 training clips the recogniser could not read back. v0.4 keeps that cleaned set and adds **767 newly generated clips** (about 69 minutes) aimed at the weak categories: 295 general-text sentences, 136 Malay-with-English sentences, 174 conversational, 142 with numbers, 20 questions. Every new clip was generated twice with different seeds and the variant the recogniser read better was kept (31 sentences were dropped because neither variant was readable). The decoder was trained for 12,000 steps and the text side for up to 10,000 with checkpoint selection on duration + recognition, on 1,281 aligned sentences in total.

## What changed in v0.3

The training clips were scored with the recogniser and the 120 worst were dropped before retraining: −15% capped-mean error against v0.2, mostly general text and numbers.

## What changed in v0.2

Half of the phoneme boundaries inside words in v0.1's training data were more than one frame (25 ms) off, because they were guessed from proportions. v0.2 times the phonemes of every Malay word from a letter-level forced alignment of the training audio.

## License and credits

Code and weights: Apache-2.0. `suaradee/lexicon_e.json` and `js/g2p/lexicon_e.json` are derived from Wiktionary data and are CC BY-SA 4.0. See `NOTICE` for all credits: Paradee-8M (Sahil Mahendrakar), Kokoro-82M (hexgrad), Chatterbox Multilingual (Resemble AI), misaki, WikiPron, g2p-id, Tatoeba and Malay Wikipedia.
