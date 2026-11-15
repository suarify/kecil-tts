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

# Suaradee v0.2

A **Malay text-to-speech model that is 8.5 MB**. One ONNX file, one voice, runs in a browser tab (transformers.js, WebGPU or WASM) or on a single CPU thread in Python.

**v0.2 is clearly more accurate than [v0.1](https://huggingface.co/lseanlon/suaradee-v0.1)** (typical sentence error roughly halved or better). **It does not fix the muffled sound**: it is just as dull as v0.1. Details are below.

- **Demo in the browser (nothing leaves your device):** [Suaradee web Space](https://huggingface.co/spaces/lseanlon/suaradee-web) (pick v0.2 in the version box)
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
pip install https://huggingface.co/lseanlon/suaradee-v0.2/resolve/main/dist/suaradee-0.2.0-py3-none-any.whl
```
```python
import soundfile as sf
from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained("lseanlon/suaradee-v0.2")
sf.write("hello.wav", tts("Selamat pagi, boleh saya bantu encik hari ini?"), SAMPLE_RATE)
```

**Browser or Node** (transformers.js 3.x). The Malay front end (`js/`) turns text into phonemes; the model is loaded by transformers.js as a `StyleTextToSpeech2Model`:

```js
import { Suaradee } from './js/suaradee.js';        // see the web Space for a complete page
const tts = await Suaradee.load({ model: 'lseanlon/suaradee-v0.2', g2pBase: './js/g2p/' });   // WebGPU if available, else WASM
const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
```

`examples/` has runnable Python and Node scripts. The ONNX file is WebGPU-safe (see v0.1's notes): the float64 phase accumulator is float32, 64-bit slice bounds are clamped and a pointless noise reversal is removed; the audio is unchanged by this.

## How good is it?

Quality was measured by playing the model's speech to Whisper large-v3 and comparing what it heard with the text (character error rate, CER; lower is better). Both versions were scored on the same sentences with the same recogniser. The recogniser itself misreads about 9% of the characters of the clips this model was trained to copy, so no model scores 0.

| Sentences the model had not seen | n | v0.1 median CER | **v0.2 median CER** | v0.2 at or under 8% |
|---|---|---|---|---|
| Call-centre style (appointments, bills, delivery) | 40 | 9.7% | **4.8%** | 70% |
| Mixed held-out set, overall | 60 | 17% | **6.3%** | 52% |
| &nbsp;&nbsp;of which call-centre style | 23 | 12.5% | **2.9%** | |
| &nbsp;&nbsp;of which numbers and money | 12 | 21% | **2.9%** | |
| &nbsp;&nbsp;of which general text | 17 | 17% | **12%** | |
| &nbsp;&nbsp;of which Malay with English words | 6 | 17% | 18% | |

Listen to `samples/` (three sentences it got right, three typical held-out ones; `samples/sentences.txt` lists the text and score of each). The held-out sentences come from the same pool of prompts as the training sentences (different sentences, similar style), so free text from elsewhere will be harder than these numbers suggest.

## Limitations (read before using)

- **Muffled sound, unchanged from v0.1.** The audio has roughly 30% less energy above 4 kHz than the speech it was trained on, and the upper harmonics are smeared (a "damp" or slightly phasey quality). The cause is the decoder's training, not the timing fix that improved the words.
- **English words are the weak spot**: sentences that mix Malay and English score about 18%. English words are only handled when they are in a list of about 430 common ones.
- General text (Wikipedia-style sentences, names, place names) has about 12% median error.
- **One synthetic female voice**, no emotion or style control.
- **Written for short sentences.** Long text is split at sentence ends and sent in pieces of at most 400 phonemes.
- Numbers, amounts, dates, times and phone numbers are read out in Malay by a rule-based normaliser.
- It is synthetic speech. Please say so when you use it.

## What changed since v0.1

Half of the phoneme boundaries inside words in v0.1's training data were more than one frame (25 ms) off, because they were guessed from proportions. v0.2 times the phonemes of every Malay word from a letter-level forced alignment of the training audio (6,050 of 6,318 words), then trains the same architecture the same way. The decoder alone made about 35% fewer wrong characters on its own, and the text side improved further.

## How it was made

1. A rule-based Malay front end (`suaradee/`, ported line for line to `js/`) writes Malay as phonemes that use the 178-symbol vocabulary of Kokoro-82M, with a lexicon of 24,000 words for the pepet and taling vowels.
2. About 750 sentences were turned into speech with Chatterbox Multilingual (a larger model), converted to one consistent voice. Roughly 49 minutes of that speech was kept after trimming; 60 sentences were held back for testing.
3. The 8M-parameter architecture and the training recipe are those of [Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3), starting from its weights. Training took under an hour on one T4 GPU.
4. Weights were exported to ONNX (opset 17) with int8 weights.

## License and credits

Code and weights: Apache-2.0. `suaradee/lexicon_e.json` and `js/g2p/lexicon_e.json` are derived from Wiktionary data and are CC BY-SA 4.0. See `NOTICE` for all credits: Paradee-8M (Sahil Mahendrakar), Kokoro-82M (hexgrad), Chatterbox Multilingual (Resemble AI), misaki, WikiPron, g2p-id, Tatoeba and Malay Wikipedia.
