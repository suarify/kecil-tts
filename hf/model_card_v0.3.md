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

# Kecil-TTS Yati v0.3

A **Malay text-to-speech model that is 8.5 MB**. One ONNX file, one voice, runs in a browser tab (transformers.js, WebGPU or WASM) or on a single CPU thread in Python.

**v0.3 is the accuracy release.** The same architecture as [v0.2](https://huggingface.co/Suarify/kecil-tts-yati-v0.2), retrained after removing the training clips the recogniser misread: overall word-level accuracy improves by about 15%, general text improves the most. **The muffled sound is unchanged** (see Limitations).

- **Demo in the browser (nothing leaves your device):** [Kecil-TTS web Space](https://huggingface.co/spaces/lseanlon/suaradee-web) (pick v0.3 in the version box)
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
pip install https://huggingface.co/Suarify/kecil-tts-yati-v0.3/resolve/main/dist/suaradee-0.3.0-py3-none-any.whl
```
```python
import soundfile as sf
from suaradee import SAMPLE_RATE, Kecil-TTS

tts = Kecil-TTS.from_pretrained("Suarify/kecil-tts-yati-v0.3")
sf.write("hello.wav", tts("Selamat pagi, boleh saya bantu encik hari ini?"), SAMPLE_RATE)
```

**Browser or Node** (transformers.js 3.x). The Malay front end (`js/`) turns text into phonemes; the model is loaded by transformers.js as a `StyleTextToSpeech2Model`:

```js
import { Kecil-TTS } from './js/suaradee.js';        // see the web Space for a complete page
const tts = await Kecil-TTS.load({ model: 'Suarify/kecil-tts-yati-v0.3', g2pBase: './js/g2p/' });   // WebGPU if available, else WASM
const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
```

`examples/` has runnable Python and Node scripts. The ONNX file is WebGPU-safe (see v0.1's notes): the float64 phase accumulator is float32, 64-bit slice bounds are clamped and a pointless noise reversal is removed; the audio is unchanged by this.

## How good is it?

Quality was measured by playing the model's speech to Whisper large-v3 and comparing what it heard with the text (character error rate, CER; lower is better). The recogniser itself misreads about 9% of the characters of the clips this model was trained to copy, so no model scores 0. Recogniser output varies slightly between runs; treat differences under about 2 points as noise.

**All 200 test sentences the model had never seen** (median / capped mean CER):

| Category | n | v0.2 | **v0.3** |
|---|---|---|---|
| General text (Wikipedia-style, names, places) | 70 | 11.8% / 14.7% | **7.0% / 10.0%** |
| Call-centre style (appointments, bills, delivery) | 60 | 5.6% / 6.3% | **5.3% / 7.1%** |
| Numbers and money | 30 | 5.0% / 8.3% | **0.8% / 6.3%** |
| Malay with English words | 20 | 21.2% / 23.0% | 22.5% / 24.6% |
| Questions | 10 | 0% / 3.0% | **0% / 2.7%** |
| Short phrases | 10 | 7.3% / 14.3% | **0% / 10.4%** |
| **Overall** | **200** | **7.5% / 11.4%** | **6.2% / 9.7%** |

On the separate 60-sentence held-out set (different sentences, similar style) the median is 5.3%.

Listen to `samples/` (three sentences it got right, three typical held-out ones; `samples/sentences.txt` lists the text and score of each). Held-out sentences come from the same pool of prompts as the training sentences, so free text from elsewhere will be harder than these numbers suggest.

## Limitations (read before using)

- **Muffled sound, unchanged from v0.1/v0.2.** The audio has roughly 30% less energy above 4 kHz than the speech it was trained on, and the upper harmonics are smeared (a "damp" or slightly phasey quality). The cause is the decoder's training, not the timing fix that improved the words. A no-training repair was tested and rejected: the shipped sound was preferred 8 times out of 8.
- **English words are still the weak spot**: sentences that mix Malay and English score about 22%. English words are only handled when they are in a list of about 430 common ones.
- General text (Wikipedia-style sentences, names, place names) has about 7% median error.
- **One synthetic female voice**, no emotion or style control.
- **Written for short sentences.** Long text is split at sentence ends and sent in pieces of at most 400 phonemes.
- Numbers, amounts, dates, times and phone numbers are read out in Malay by a rule-based normaliser.
- It is synthetic speech. Please say so when you use it.

## What changed since v0.2

The training clips were scored with the same recogniser used for evaluation, and the 120 worst of 742 were dropped (109 the recogniser could not read back, 30 with unusual speaking rates, overlapping) before retraining the decoder and the text side with the identical recipe. 92 of the 200 test sentences got better, 40 stayed the same, 68 got worse; the net effect is −15% on the capped-mean error and −17% on the median. Per category, general text and numbers improved clearly, questions and short phrases improved, while call-centre style and English-mixed sentences moved by amounts inside the recogniser's run-to-run variation.

## What changed in v0.2 (for comparison)

Half of the phoneme boundaries inside words in v0.1's training data were more than one frame (25 ms) off, because they were guessed from proportions. v0.2 times the phonemes of every Malay word from a letter-level forced alignment of the training audio (6,050 of 6,318 words), then trains the same architecture the same way.

## How it was made

1. A rule-based Malay front end (`suaradee/`, ported line for line to `js/`) writes Malay as phonemes that use the 178-symbol vocabulary of Kokoro-82M, with a lexicon of 24,000 words for the pepet and taling vowels.
2. About 750 sentences were turned into speech with Chatterbox Multilingual (a larger model), converted to one consistent voice. 60 sentences were held back for testing.
3. The 8M-parameter architecture and the training recipe are those of [Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3), starting from its weights. Training took under an hour on one T4 GPU.
4. v0.3 repeats step 3 on the cleaned clip set (634 of 742 clips), then exports ONNX (opset 17) with int8 weights.

## License and credits

Code and weights: Apache-2.0. `suaradee/lexicon_e.json` and `js/g2p/lexicon_e.json` are derived from Wiktionary data and are CC BY-SA 4.0. See `NOTICE` for all credits: Paradee-8M (Sahil Mahendrakar), Kokoro-82M (hexgrad), Chatterbox Multilingual (Resemble AI), misaki, WikiPron, g2p-id, Tatoeba and Malay Wikipedia.
