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

# Kecil-TTS Yati v0.1

A **Malay text-to-speech model that is 8.5 MB**. One ONNX file, one voice, runs in a browser tab (transformers.js, WebGPU or WASM) or on a single CPU thread in Python, at about 25 times real time.

**A newer, more accurate version exists: [Kecil-TTS Yati v0.2](https://huggingface.co/Suarify/kecil-tts-yati-v0.2)** (typical error roughly halved; the muffled sound is the same). This page describes v0.1.

**This is a first release.** It is understandable, but it sounds slightly muffled, and roughly one sentence in ten has a wrong word. Details and numbers are below, so you can judge whether it is good enough for your use.

- **Demo in the browser (nothing leaves your device):** [Kecil-TTS web Space](https://huggingface.co/spaces/lseanlon/suaradee-web)
- **Gradio demo:** `examples/gradio_app.py` (`pip install gradio`, then `python examples/gradio_app.py`). A hosted Gradio Space needs a paid Hugging Face plan; the code is ready in `hf/space_gradio` of the project if you want to host one.

| | |
|---|---|
| Model file | `onnx/model_quantized.onnx`, **8.46 MB** (int8 weights) |
| Parameters | 7.9 M |
| Output | 24 kHz mono |
| Speed | about 25x real time on one CPU thread (onnxruntime); about 15x in a browser tab on WebGPU (6x on the first call) |
| Language | Bahasa Melayu (Malaysia), with common English words used in Malaysian speech |
| Inputs | `input_ids` (int64 `[1,T]`, phoneme ids with a 0 at each end, `T` up to 402) and `speed` (float32 `[1]`) |

## Runs on

Tested for this release: Python 3.12 with onnxruntime on CPU (Windows, clean install from this repo's wheel), Node 24 with transformers.js 3.8 (output matches Python sample for sample), and a Chromium browser with **both WebGPU and WASM** through the web Space. The ONNX file is built to be WebGPU-safe: a double-precision phase accumulator was moved to float32 and 64-bit slice bounds were clamped, because ONNX Runtime Web's WebGPU backend cannot run them. The audio is unchanged by this (measured against the unpatched export, within the sampling noise of the model).

## Quick start

**Python** (`onnxruntime` + `numpy`; no PyTorch):

```bash
pip install onnxruntime numpy huggingface_hub soundfile
pip install https://huggingface.co/Suarify/kecil-tts-yati-v0.1/resolve/main/dist/suaradee-0.1.0-py3-none-any.whl
```
```python
import soundfile as sf
from suaradee import SAMPLE_RATE, Kecil-TTS

tts = Kecil-TTS.from_pretrained("Suarify/kecil-tts-yati-v0.1")
sf.write("hello.wav", tts("Selamat pagi, boleh saya bantu encik hari ini?"), SAMPLE_RATE)
```

**Browser or Node** (transformers.js 3.x). The Malay front end (`js/`) turns text into phonemes; the model is loaded by transformers.js as a `StyleTextToSpeech2Model`:

```js
import { Kecil-TTS } from './js/suaradee.js';        // see the web Space for a complete page
const tts = await Kecil-TTS.load({ model: 'Suarify/kecil-tts-yati-v0.1', g2pBase: './js/g2p/' });   // WebGPU if available, else WASM
const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
```

`examples/` has runnable Python and Node scripts.

## How good is it?

Quality was measured by playing the model's speech to Whisper large-v3 and comparing what it heard with the text (character error rate, CER; lower is better). The recogniser itself misreads 7-9% of the characters of the human-like clips this model was trained to copy, so no model scores 0.

| Sentences the model had not seen | n | Median CER | Share at or under 8% |
|---|---|---|---|
| Call-centre style (appointments, bills, delivery) | 40 | **9.7%** | 42% |
| Mixed held-out set, overall | 60 | 17% | 23% |
| &nbsp;&nbsp;of which general text | 17 | 17% | |
| &nbsp;&nbsp;of which numbers and money | 12 | 21% | |
| &nbsp;&nbsp;of which Malay with English words | 6 | 17% | |

Listen to `samples/` (three sentences it got right, three typical ones; the file `samples/sentences.txt` lists the text and score of each). Free text outside call-centre style, names, place names and numbers are the weak spots.

## Limitations (read before using)

- **Muffled sound.** The audio has roughly 40% less energy above 4 kHz than the speech it was trained on, and its upper harmonics are smeared (a "damp" or slightly phasey quality). This is a known issue of the first training run.
- **About one sentence in ten has a wrong word**, more for names, numbers and rare words.
- **One synthetic female voice**, no emotion or style control.
- **Written for short sentences.** Long text is split at sentence ends and sent in pieces of at most 400 phonemes.
- **English words** are only handled when they are in a list of about 430 common ones (`confirm`, `booking`, `email`, ...).
- Numbers, amounts, dates, times and phone numbers are read out in Malay by a rule-based normaliser.
- It is synthetic speech. Please say so when you use it.

## How it was made

1. A rule-based Malay front end (`suaradee/`, ported line for line to `js/`) writes Malay as phonemes that use the 178-symbol vocabulary of Kokoro-82M. It tells the pepet vowel from the taling vowel with a lexicon of 24,000 words, and handles numbers, abbreviations and English loanwords.
2. About 750 sentences were turned into speech with Chatterbox Multilingual (a larger model), converted to one consistent voice. Roughly 49 minutes of that speech was kept after trimming, 60 sentences of it held back for testing.
3. The 8M-parameter architecture and the training recipe are those of [Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3), starting from its weights: a text side that predicts phoneme durations, pitch, loudness and features, and a decoder that turns them into audio. Training took under an hour on one T4 GPU.
4. Weights were exported to ONNX (opset 17) with int8 weights.

## What is next

v0.2 is being trained with phoneme timings taken from letter-level alignment (half of the phoneme boundaries in v0.1's training data were more than one frame off), which should reduce wrong words and the smeared sound. It will be released only if it measures better than v0.1.

## License and credits

Code and weights: Apache-2.0. `suaradee/lexicon_e.json` and `js/g2p/lexicon_e.json` are derived from Wiktionary data and are CC BY-SA 4.0. See `NOTICE` for all credits: Paradee-8M (Sahil Mahendrakar), Kokoro-82M (hexgrad), Chatterbox Multilingual (Resemble AI), misaki, WikiPron, g2p-id, Tatoeba and Malay Wikipedia.
