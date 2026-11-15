# progress.log.md — Suaradee, 2026-11-15

Plain-language summary of where everything stands. The detailed running log is `STATUS.md`; the plan being
executed is `continue.plan.v7.md`.

## Shipped (public, under the Suarify account)

| What | Where | State |
|---|---|---|
| Suaradee v0.1 (8.46 MB) | `Suarify/suaradee-v0.1` | live, WebGPU + WASM + Python verified |
| Suaradee v0.2 (8.46 MB, letter-level timings) | `Suarify/suaradee-v0.2` | live; accepted by the owner by ear; test median CER 4.8% (conversational), held-out 6.3% |
| Web demo (v0.1/v0.2 selector) | Space `Suarify/suaradee-web` | live |
| Gradio demo | code only (`hf/space_gradio`, `examples/gradio_app.py`) | free HF accounts cannot host Gradio Spaces (402) |

The older `lseanlon/` copies still exist and were left alone.

## Done today (all local / CPU, no GPU after kernel kmb-12)

- **Why Paradee v1.1 sounds clear:** deterministic Kokoro teacher audio, 23.9 h of data, 40k spectral + 15k
  adversarial steps. Ours: approximate inputs, 49 min of data.
- **Plan v6 (pitch refinement) dropped:** the stricter held-out-band test did not confirm it. Never trained.
- **Noise-to-harmonic knob** (`scripts/tonal_knob.py`): flatness went 0.69 -> 0.625, but the owner heard no
  difference between the three settings. Flatness is not what he hears as damp.
- **Plan v7 written** (`continue.plan.v7.md`): A effect test, B card upgrade, C lower CER, D Michael voice.
- **B1 streaming done:** `stream()` in `python/suaradee/tts.py` and `web/suaradee.js`; comma-only splitting
  (first piece <= 60 phonemes, later <= 160); Python tests 11 pass; Node parity pass; examples
  `hf/stream_python.py`, `hf/stream_node.mjs`; Stream switch in the web Space. Time to first audio for a
  3-sentence paragraph: Python 0.12 s, Node 0.08 s (browser numbers still to measure on the live Space).
  Package version 0.2.1 (not yet published).
- **A1 done, with the owner's ears:** of four versions per sentence, the Chatterbox clip and Kokoro were never
  damp; the decoder fed the clip's own measured inputs was damp once; v0.2 from text was damp 4 times. The
  damp comes from what the text side predicts (pitch and loudness contour), not from the decoder or the
  training audio. Plan step A2 (EQ and exciter effects) was therefore skipped.
- **Contour repair test rejected by the owner:** four no-training repairs of the predicted pitch (gap fill,
  smoothing, range cap) were all worse than the shipped contour (shipped picked 8 of 8). Int8 export is not
  the cause (fp32 and int8 match at 0.994-0.997 envelope correlation).

## What this means

No audio effect or contour patch fixes the damp sound without training. The remaining route is better
pitch/loudness prediction by the text side, which overfits (held-out f0/N loss 0.56/2.34 vs training
0.12/0.37), i.e. a training job.

## Open, in order, and what each needs

| Step | Needs | Cost | Why |
|---|---|---|---|
| B2 Kokoro comparison kernel (200 test sentences: v0.1, v0.2, Kokoro) | owner's yes for GPU | about 0.2-0.5 GPU-h | numbers and audio for the v0.2 card; also the baseline for CER work |
| B3/B4 card with samples, comparison, streaming; publish 0.2.1 wheel to Suarify | B2 output; owner's yes to publish | CPU | owner asked for this |
| Text-side retrain aimed at pitch/loudness generalisation (C2/C5 style) | owner's yes for GPU | about 0.5 GPU-h | the only lead left for the damp sound |
| C CER work (learning curve, clean clips, English words) | after B2 | 1.1 GPU-h first part | target <= 5% median on all 200 test sentences |
| D Michael voice (`am_michael`) | 30-clip listening check, then GPU | about 1 GPU-h | second voice |

## Budget

GPU used since the pivot: 7.77 h of the 30 h weekly allowance (resets 2026-11-21). Nothing is running on
Kaggle now. Git is clean; last commit `2810c7b`.

## Notes for whoever continues

- Local torch scripts: `PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe ...`.
- v0.2 weights: `runs/k12/ckpt/`; training rows (742): `runs/k12_rows/`.
- Listening pages live in `runs/listen/`; the owner opens them directly or via `python -m http.server 8765 --directory runs`.
- Do not propose Kokoro voice blends as the Malay voice, and do not propose more pitch/contour transforms.
- Ask the owner before any GPU step and before publishing.
