# Plan v7: clarity effect test, v0.2 card upgrade, lower CER, second voice (Michael)

2026-11-15. Written so another agent can execute it without the earlier conversation. It supersedes
`continue.plan.v6.md`. `STATUS.md` is the running log; append to it after every step.

The owner decided three things on 2026-11-15:

1. The three noise settings in `runs/listen/v03.html` sounded the same to him. He wants to know whether a
   no-training audio effect ("plugin / patch") can fix the damp sound. He will take part in the listening.
2. v0.2 stays the release. Its Hugging Face card gets more samples, a comparison with Kokoro, and streaming
   examples.
3. One consolidated plan to lower the error rate (CER), and a second voice trained toward Kokoro's
   `am_michael`.

Work packages below are **A** (effect), **B** (card), **C** (CER), **D** (Michael). Recommended order:
B, then A, then C, then D. B and A need almost no GPU. D reuses what C builds.

---

## 0. What an executing agent must know first

### 0.1 Where things are

| Thing | Place |
|---|---|
| Project (git repo) | `C:\Users\Acer\Downloads\kokoro-malay-bella` |
| Shipped models | `Suarify/suaradee-v0.1`, `Suarify/suaradee-v0.2`, Space `Suarify/suaradee-web` |
| v0.2 ONNX (WebGPU-safe) | `runs/k12/ms_A_int8_gpu.onnx` |
| v0.2 PyTorch weights | `runs/k12/ckpt/dec_last.pt`, `runs/k12/ckpt/textA_last.pt` |
| Training code (Paradee-style) | `paradee_ms/` (mirror for Kaggle: `kaggle/ds_pms/`, dataset `seanlon/kmb-paradee-ms`) |
| Malay front end | `g2p/ms_g2p.py`, `g2p/ms_norm.py` and JS ports; Kaggle dataset `seanlon/kokoro-malay-g2p` (`kaggle/ds_g2p/`) |
| Sentence sets | `kaggle/ds_g2p/`: `cb_1100.tsv` (training sentences), `corpus_train.tsv` (7,500 unused), `test_200.tsv` (never trained on) |
| Clip generation kernel | `kaggle/k08_cbgen/` (Chatterbox Multilingual, then ChatterboxVC toward a reference voice) |
| Last training kernel | `kaggle/k12_v5/k12_v5.py` (copy its structure for new kernels) |
| Release builders | `scripts/build_hf.py`, `scripts/publish_hf.py`, `scripts/webgpu_patch.py` |
| Listening pages | `runs/listen/*.html`; the owner opens them by double-click or his own `python -m http.server 8765 --directory runs` |

### 0.2 Rules that cost time when broken

- **Local Python for torch work:** `PYTHONPATH=runs/k09/stubs PYTHONUTF8=1 .venv/Scripts/python.exe ...` (the stub
  replaces `misaki`, which Kokoro imports). Plain `python` has librosa and onnxruntime but not Kokoro.
- **Kaggle:** CLI user `seanlon`. Image is Python 3.13: install Kokoro with
  `pip install --no-deps --ignore-requires-python "kokoro==0.9.4"` and write the misaki stub (see the top of
  `k12_v5.py`). Find inputs with `glob('/kaggle/input/**/name', recursive=True)`, never fixed paths.
  Whisper large-v3 needs `batch_size=1` to fit beside other models.
- **After editing anything in `paradee_ms/`** copy it to `kaggle/ds_pms/` and run
  `kaggle datasets version -p kaggle/ds_pms -m "..."` before pushing a kernel that uses it.
- **Scoring is fixed:** Whisper large-v3, `language='malay'`, CER on letters only after `normalize()`, capped at
  1.0 per sentence. Always report median and capped mean (hallucinations inflate a plain mean).
- **Disk:** about 3 GB free on C:. Do not download whole kernel outputs; use
  `kaggle kernels output <kernel> -p DIR --file-pattern "<regex>"`.
- **Do not write multi-line Python through Bash heredocs into project files** (backslashes get mangled). Use
  the file tools.
- **Publishing:** `python scripts/publish_hf.py --user Suarify --profile "Suarify Malaysia"`; run `--dry-run`
  first. Card sources name `lseanlon/`; the script retargets them. Free HF accounts cannot host Gradio Spaces.
- **Do not propose** Kokoro voice blends (Dora etc.) as the Malay voice. The owner rejected them twice; only
  the Chatterbox-derived sound "sounds Malay" to him.
- **GPU so far:** about 6.7 GPU-h since the pivot. Ask the owner before any step marked **GPU** below.

### 0.3 Baseline numbers (v0.2, kernel kmb-12-v5)

| Measure | Value |
|---|---|
| Test sentences `te00000`-`te00039` (all "conversational"), median CER | 4.8% (capped mean 6.5%, 70% at or under 8%) |
| 60 held-out clips, from text, median CER | 6.3% (capped mean 10.5%) |
| Same clips, decoder fed the measured timings/pitch, median CER | 5.9% (capped mean 14.5%) |
| Same clips, the Chatterbox audio itself, median CER | **1.9%** (capped mean 8.7%) |
| Held-out by category, median CER | conversational 2.9% (23), numbers 2.9% (12), general 12.3% (17), code-switch 18.3% (6) |
| Size / speed | 8.46 MB int8; about 21-25x real time on one CPU thread |
| Training data | 742 clips, 48.8 min; 6,000 decoder steps (8 min) and 6,000 text steps (10 min) on a T4 |

Two facts drive package C:

- The reported 4.8% covers conversational sentences only (the first 40 rows of `test_200.tsv`).
- The decoder alone already loses most of the accuracy (1.9% for the clip, 5.9% once the decoder re-speaks
  it with perfect inputs). The text side adds little. So CER work is mostly decoder and data work.

### 0.4 What was tried for the damp sound and failed (do not repeat)

| Attempt | Result |
|---|---|
| Fixed EQ toward the Chatterbox clips (`scripts/eq_test.py`) | Average spectrum already within 1-2.5 dB; no fix |
| Short adversarial stage (k11, 843 steps) | No gain; a full stage would cost about 6.5 GPU-h |
| Letter-level timings (k12) | Fixed wrong words, not the sound |
| Refined pitch (`paradee_ms/f0_refine.py`) | Failed a held-out-band test on CPU; never trained |
| Less noise, more harmonics in the decoder head (`scripts/tonal_knob.py`) | Measured flatness 0.69 -> 0.625, **owner heard no difference** |

Consequence: "flatness 2-6 kHz" is not what the owner hears as damp. Do not use it as a gate again.

---

## A. Can a no-training effect fix the damp sound?

Goal: find out in at most two listening rounds, on CPU, whether a signal-processing effect makes v0.2 clearer
to the owner's ears. If one does, bake it into the ONNX file so Python, Node and the browser all get it with
no code change. If none does, close the topic and leave it to package C.

### A1. First find where the damp sound comes from (owner: about 10 minutes)

Build `runs/listen/damp_where.html` with `scripts/listen_damp_where.py` (new; copy the page style of
`scripts/listen_v03.py`). Eight held-out sentences, four versions each, labelled openly:

1. the Chatterbox clip the model learned from (`runs/k12/cbwav/<id>.wav`),
2. the v0.2 decoder fed that clip's measured inputs (render locally; needs the training rows:
   `kaggle kernels output seanlon/kmb-12-v5 -p runs/k12 --file-pattern "^rows/shard_0000\.pt$"`, about 250 MB),
3. v0.2 from text (the shipped ONNX),
4. Kokoro `af_bella` reading the same Malay phonemes (code in `kaggle/k01_baseline/`).

Per sentence the owner ticks every version that sounds damp. He should use headphones. Read the result so:

| Owner's answer | Meaning | Next |
|---|---|---|
| 1 is damp too | The training audio is the limit | Effects that add brightness and presence (A2: E1, E2, E3); later C5 |
| 1 is fine, 2 and 3 are damp | The decoder loses it | All of A2; later C5 |
| Only 3 is damp | The text side (flat pitch or loudness) | Skip A2. Test a pitch-range and loudness-range stretch instead (`F0`, `N` scaling in `scripts/tonal_knob.py`'s `text_side`) |
| Even 4 is damp | His playback chain | Stop; ask him to retry on other headphones |

Also write `scripts/damp_profile.py` (CPU, 30 clips per system) and put its table in `STATUS.md`. For
systems 1, 3 and 4 measure: level per third-octave band (long-term average), the frequency below which 95%
of the energy sits, energy above 4 kHz and above 8 kHz, onset sharpness (median rise of the 2-8 kHz
envelope at word starts, dB per 10 ms), decay time after vowel ends, and pitch range (10th-90th percentile
in semitones). The measure where v0.2 is furthest from both other systems is the first suspect.

### A2. Effect candidates (pure functions `wav -> wav` at 24 kHz, numpy/scipy, in `scripts/fx.py`)

| Id | Effect | Recipe to start from |
|---|---|---|
| E1 | Match-EQ to Kokoro | Difference of third-octave long-term spectra (Kokoro minus v0.2), smoothed, limited to +/-9 dB, as a 255-tap linear-phase FIR |
| E2 | Presence EQ | High-pass 70 Hz; -2 dB wide cut at 300 Hz; +2 dB peak at 3 kHz; +5 dB high shelf from 5 kHz |
| E3 | Harmonic exciter | Band-pass 2-6 kHz -> `tanh(drive * x)` -> high-pass 5 kHz -> add back at 10-25%. Creates new energy above 5 kHz that follows the voice; EQ cannot do that when the band is empty |
| E4 | Consonant emphasis | In the 3-10 kHz band, boost frames whose level rises fast (upward expander on the band envelope, at most +6 dB, 5 ms attack, 40 ms release) |
| E5 | Tail cut (only if A1 shows long decay) | Per-band envelope follower; attenuate energy that falls slower than 60 dB per 250 ms |
| E6 | Best pair | The two the owner liked most in round 1, chained |
| E7 | Pretrained enhancer, reference only | Resemble Enhance or VoiceFixer on the v0.2 output. Not shippable in an 8 MB model; it only shows how far any post-processing can go. Optional **GPU** (about 0.2 h) or slow CPU |

Rules for every candidate: normalise all versions of a sentence to the same loudness (RMS of the loudest 60%
of frames) so "louder" cannot win; no sample above 0.99; keep the sample count.

### A3. Listening rounds (owner: 15 minutes, then 10)

- **Round 1**, `runs/listen/fx1.html`: 10 unseen test sentences, each with the raw v0.2 and E1-E5 (E7 too
  if made). Blind: shuffle the order per sentence, store the key in the page, show it only in the result
  box. Include the raw clip twice in two sentences (a hidden repeat checks that picks are consistent). Per
  sentence: "best" and "any that is worse than the others".
- **Round 2**, `runs/listen/fx2.html`: the winner at three strengths plus raw, 10 other sentences.
- Take sentences evenly from the categories in `test_200.tsv`.

### A4. Gate and stop rule (fixed now)

- **Pass:** in round 2 one strength is picked over raw on at least 7 of 10 sentences and marked worse on at
  most 1.
- **No harm:** median CER on `test_200` moves by less than 0.5 points against raw (score on Kaggle together
  with B2, or on CPU overnight).
- **Stop:** if round 1 has no candidate picked on at least 6 of 10, or round 2 fails, stop package A. Record
  it in `STATUS.md` and do not try further effects.
- If only E7 wins: do not ship an effect. Use that enhancer on the **training clips** in step C5 instead.

### A5. Shipping a winner (CPU only)

1. Write `scripts/add_fx.py IN.onnx OUT.onnx --fx <id> --strength <x>`: append nodes after `waveform`
   (FIR = `Conv`, exciter = `Conv` -> `Tanh` -> `Conv` -> `Add`). E1-E3 are static and need only those ops.
   E4 and E5 change over time; if one of them wins, build it from the conv-based STFT in
   `paradee_ms/export_paradee_v1_1.py` (`ConvSTFT`) or ship it in the wrappers instead (step 4).
2. Run `scripts/webgpu_patch.py` on the result. Check: output equals the numpy effect on the unpatched
   model's audio (same noise is not possible, so compare the effect on a fixed input by exporting a
   test graph with the effect alone); file under 10 MB; speed within 5% of v0.2.
3. Verify like the v0.2 release: Python wheel in a clean venv, Node `transformers.js`, browser WebGPU and
   WASM through the Space (use `?model=` to point at the new repo).
4. Fallback if the graph route fails: `clarity=` option in `python/suaradee/tts.py` and `web/suaradee.js`,
   same maths in both, parity test on 5 sentences.
5. Release as `suaradee-v0.3` ("v0.2 weights plus clarity filter") only on the owner's yes; add it to
   `VERSIONS` in `scripts/publish_hf.py` and to the Space's selector.

---

## B. v0.2 card upgrade: samples, Kokoro comparison, streaming

No model change. Deliverables: a richer `hf/model_card_v0.2.md`, new sample files, streaming in the Python
package and the JS front end, a "Stream" mode in the web Space, all published to `Suarify/`.

### B1. Streaming (there is none today)

The model makes one sentence per call, so streaming is per sentence or clause: audio for the first piece
plays while the next is being made.

1. `python/suaradee/tts.py`: add `stream(text, speed=1.0, pause=0.12, first_max=60)`, a generator that
   yields one float32 array per piece. Split long sentences at commas when a piece exceeds `first_max`
   phonemes for the first piece and 160 for later ones, so the first sound comes early. `__call__` becomes
   `np.concatenate(list(self.stream(...)))` with the same output as today (test: identical piece
   boundaries for texts without commas).
2. `web/suaradee.js`: `async *stream(text, opts)` with the same splitting; `speak()` keeps its behaviour.
   Port line for line and add both to the Python/JS parity test.
3. Examples (in `hf/` so `build_hf.py` copies them into `examples/`):
   - `stream_python.py`: plays chunks as they arrive with `sounddevice`, prints time to first audio.
   - `stream_node.mjs`: writes chunks to stdout as 16-bit PCM (pipe into `ffplay -f s16le -ar 24000 -`).
   - Web Space: a "Stream" switch; chunks scheduled gap-free on one `AudioContext`
     (`source.start(nextTime)`), with time to first audio shown.
4. Measure time to first audio for a 3-sentence paragraph: Python on 1 CPU thread, browser WASM, browser
   WebGPU (after warm-up). Put the three numbers in the card.
5. Bump the package to 0.2.1 (`python/pyproject.toml`); the model file does not change.

### B2. Kokoro comparison, measured on the same sentences (**GPU**, about 0.2 h)

Kernel `kaggle/k13_compare/` (copy the scoring half of `k12_v5.py`). On all 200 rows of `test_200.tsv`:

| System | How |
|---|---|
| Suaradee v0.2 | `runs/k12/ms_A_int8_gpu.onnx` uploaded in a small dataset, onnxruntime CPU |
| Suaradee v0.1 | `runs/k09/ms_int8_gpu.onnx`, same |
| Kokoro-82M `af_bella` | `KModel` fed our Malay phonemes (as in `kaggle/k01_baseline/`) |

Outputs: `compare_scores.csv` (system, id, category, cer, heard), a report with median / capped mean /
share at or under 8% overall and per category, and the audio of 24 fixed sentences (4 per category) for
every system. Speed is measured locally, one CPU thread, same 20 sentences, for Suaradee (onnxruntime) and
Kokoro (PyTorch, `.venv`). This run also gives package C its full-test baseline (C0).

### B3. Card content (`hf/model_card_v0.2.md`)

1. **Samples table**, 18 rows (3 per category of `test_200.tsv`: conversational, numbers, general, question,
   short, code-switch). Columns: text, Suaradee v0.2, Kokoro `af_bella`, CER of the Suaradee clip. Choose rows
   by a fixed rule (first three ids of each category), not by how good they sound. Use
   `<audio controls src="https://huggingface.co/lseanlon/suaradee-v0.2/resolve/main/samples/..."></audio>`.
2. **Reference row set**: 6 held-out sentences with the Chatterbox clip the model learned from next to the
   model's version.
3. **Comparison table**: Suaradee v0.2 against Kokoro-82M on parameters (7.9M / 82M), file size (8.46 MB /
   325 MB), speed, median CER overall and per category, training (48.8 minutes of synthetic Malay speech and
   under 1 GPU-hour on a T4 for Suaradee; Kokoro's own figures quoted from its card with a link).
4. **A plain paragraph on what the numbers mean:** Kokoro is clearer and may score lower CER because its
   English-trained voice articulates crisply, but it reads Malay with a foreign accent (in the owner's
   blind test of 60 sentences stock Bella was never preferred). State whichever way the measured numbers
   fall; do not soften a loss.
5. **Streaming section** with the three snippets and the measured time to first audio.
6. Fix the accuracy table: replace the 40-sentence figure with the 200-sentence figures per category from B2.
7. Keep the Limitations section; the muffled sound stays listed.

`scripts/build_hf.py` needs a `--samples-table compare_dir` path for the new sample layout; keep the old
behaviour for v0.1.

### B4. Publish and verify

`publish_hf.py --dry-run`, then publish to Suarify (the owner asked for this card update on 2026-11-15).
Verify on the live pages: every audio player loads, the Space streams on WebGPU and WASM, and
`pip install` of the 0.2.1 wheel in a clean venv runs `stream_python.py`. Leave the old `lseanlon/` copies
alone unless the owner says to delete them.

---

## C. Lower CER (target release: v0.4, or v0.3 if package A ships nothing)

Targets on the full `test_200.tsv`: median CER at or under 5% overall, general at or under 8%, code-switch at
or under 10%, with no category worse than v0.2. Fill the v0.2 column from B2 before starting.

### C0. Baseline (done by B2)

Per-category CER of v0.2 and of Kokoro on all 200 test sentences. Add the same per-category split for the
60 held-out clips of the **Chatterbox audio itself** (from `rows/shard_0000.pt`): if the teacher clips are
already poor in a category, more of the same data cannot fix that category; C3 must.

### C1. Does more data help? Learning curve (**GPU**, about 0.6 h)

Kernel `k14_curve`: rows from the kmb-12 output (no rebuild). Train the decoder on 25%, 50% and 100% of the
682 training clips (same seed, 6,000 steps each, 8 min each), score decoder-only CER on the 60 held clips.

- If 50% -> 100% improves median or capped-mean CER by 15% or more (relative): data is the lever. Do C4.
- If the curve is flat: more of the same data will not help. Skip C4, put the effort in C2, C3 and C5.

### C2. Clean the training clips (**GPU**, about 0.5 h including a retrain)

1. Score all 742 training clips with Whisper large-v3 (`k15_clean`). The held-out sample has median 1.9% but
   capped mean 8.7%, so roughly one clip in six is misread or mis-spoken.
2. Drop clips with CER above 15%, clips where `ms_rows.py` timed fewer than 90% of the words from letters,
   and clips whose speaking rate is outside the 2nd-98th percentile.
3. Retrain decoder and text side with the k12 recipe on the cleaned set; compare on held-out and test.
   Keep the cleaning only if capped-mean CER improves.

### C3. English words and unknown words (CPU, the largest category gap)

1. From `compare_scores.csv`, list every word Whisper got wrong in the code-switch and general rows, for
   v0.2 **and** for the Chatterbox clips. Sort into: English word not in `lexicon_en.json` (about 430
   entries today), Malay word with the wrong `e` (pepet / taling), names and places, numbers.
2. English: grow `lexicon_en.json` to about 3,000 words. Take the most frequent English words in Malaysian
   text (count Latin-script non-Malay tokens in `corpus_train.tsv` and Malay Wikipedia against the Malay
   lexicon) and take their pronunciations from misaki's `us_gold.json` / `us_silver.json`, which are
   already in Kokoro's symbols; `scripts/build_lexicon_en.py` does this today for a hand-picked list, so
   feed it the longer list. Add a fallback for unknown English-looking words (not in the Malay
   lexicon, contains `th`, `sh`, final `-tion`, `x`, `q`, doubled consonants...): simple English
   letter-to-sound rules instead of Malay ones.
3. Teacher side: Chatterbox must also say these words correctly, or the student learns the error. Extend
   `teacher_respell.json` for the same words and listen to 30 generated clips before generating in bulk.
4. Rebuild the JS lexicon, keep the Python/JS parity test at 2,000/2,000.

### C4. More training speech (**GPU**, about 3.5 h per 1,000 clips; only if C1 says so)

Measured cost in kmb-08: 746 clips (61.7 min) took 146 min to generate plus about 21 min to convert.

1. Sentences: first the 394 rows of `cb_1100.tsv` that were never generated (the run hit its time cap), then
   new rows from `corpus_train.tsv`, weighted toward the weak categories: general 50%, code-switch 30%
   (using the C3 word list), numbers and names 20%.
2. Kernel `k16_cbgen2` from `k08_cbgen.py`, with three changes:
   - **Save the raw Chatterbox audio too** (`cb_raw.pt`, before voice conversion). A second voice then costs
     only the conversion pass (package D).
   - **Best of two:** generate each sentence with two seeds, keep the one Whisper reads better, drop the
     sentence if both exceed 15% CER. This halves throughput; use it for general and code-switch rows only.
   - Check `nvidia-smi`: if the session has two T4s, run two generator processes, one per GPU.
3. Rebuild rows with `ms_rows.py --letters`, train with the k12 recipe but 12,000 decoder steps and 10,000
   text steps (still under 40 minutes), text checkpoint chosen with `--score dur_asr`.
4. Gate after the first 1,000 new clips: weak-category median CER down by 25% (relative) or more. Only then
   generate a second 1,000.

### C5. Decoder quality (only after C1-C4, and only if still needed)

In order of cost:

1. If package A found that a pretrained enhancer (E7) is what the owner prefers: enhance the training clips
   with it, rebuild rows, retrain the decoder (**GPU** about 0.5 h). The model then learns the clearer sound
   and no run-time effect is needed.
2. Adversarial stage sized to fit: discriminators at half width, segment 48 frames, mixed precision, about
   6,000 steps, spectral weight 3 (**GPU** about 3 h). Gate on the owner's ears, not on flatness.

### C6. Release gate for the new version

All of: targets at the top of C met; test-set median not worse than v0.2 in any category; file under 10 MB;
WebGPU and WASM verified; the owner prefers it or hears no loss on 12 blind pairs against v0.2.

---

## D. Second voice: Michael (`am_michael`)

Result: `Suarify/suaradee-michael-v0.1`, a separate 8.5 MB model with the same interface. One model with two
voices is possible later (`VDecoder(n_voices=2)`) but changes the ONNX inputs and the exporter; not now.

Do D after C4 if C4 runs, because C4 saves the raw clips that make D cheap and both voices then share the
better data. If the owner wants Michael sooner, use D2 route (b).

### D1. Reference audio (CPU)

Render 12 English sentences (`kaggle/ds_g2p/en_ref.tsv`) with Kokoro `am_michael` using Kokoro's normal
English pipeline, as was done for Bella (`kaggle/ds_ref/bella_en/`). Join 10-15 seconds into
`ref_en_michael.wav`. Listen once: it must be clean, with no clipped ends.

### D2. Convert the clips (**GPU**)

| Route | Source | Cost | Risk |
|---|---|---|---|
| (a) preferred | Raw Chatterbox clips saved by C4 (`cb_raw.pt`) | About 0.03 min per clip: 0.4 h for 750, 1 h for 2,000 | None new |
| (b) no raw clips yet | The Bella-converted clips (`cb_clips.pt` in the kmb-08 output), converted again | 0.4 h for 746 | Converting twice may add artefacts |
| (c) | Regenerate raw speech | 2.5 h per 746 clips | Cost only |

Before a full run, convert 30 clips and give the owner a page (`runs/listen/michael_ref.html`): Kokoro
Michael in English, then the 30 Malay clips. Gate: he accepts the voice as Michael-like and Malay-sounding
on at least 24 of 30. For route (b) add the Bella versions beside them so he can hear whether the second
conversion damaged anything.

### D3. Code changes for a male voice

| File | Change |
|---|---|
| `paradee_ms/ms_rows.py` | `--voice am_michael` replaces the hard-coded `af_bella` voice pack (it sets Kokoro's duration guess and the loudness offset `N_OFF` / `N_TARGET`); `--f0-min 55 --f0-max 300` passed to `measure_f0` (Bella uses 70-450) |
| `paradee_ms/ms_train.py` | `harm_loss` counts a frame as voiced above 100 Hz; make it a flag and use 70 Hz, or most of Michael's frames are ignored |
| `scripts/f0_check.py`, any metric script | Same thresholds (140 Hz "steady voiced" filter becomes 80 Hz) |
| `scripts/build_hf.py`, `scripts/publish_hf.py` | `--name suaradee-michael` so repo, wheel default and card are named per voice |
| `python/suaradee/tts.py`, `web/suaradee.js` | No change; the voice is the model file |

Check after building rows: share of voiced frames near 0.7 (Bella rows: 0.72) and median pitch between 90
and 140 Hz. A median near 200 Hz means octave errors: lower `--f0-max`.

### D4. Train, score, release (**GPU**, about 0.5 h)

Same kernel as the Bella model at that time (k12 recipe, or the C4 recipe if C ran), started from the same
Paradee v1.3 weights (`ms_init.py`). Gates: test median CER within 1.5 points of the Bella model trained on
the same clips; the owner accepts 12 sentences by ear; WebGPU and WASM verified. Then card (reuse the v0.2
card structure, with Kokoro `am_michael` as the comparison voice), samples, publish, and add the voice to
the web Space's selector.

---

## Budget and owner touchpoints

| Package | GPU (estimate) | Owner's time |
|---|---|---|
| B card, comparison, streaming | 0.2 h | Read the card once |
| A effect test | 0 h (0.2 h if E7 is included) | 10 + 15 + 10 minutes of listening |
| C0-C3 | 1.1 h | None |
| C4 per 1,000 clips, with training | about 4 h | None |
| C5 if needed | 0.5 h or 3 h | 12 blind pairs |
| D Michael, route (a) or (b) | about 1 h | 30-clip check, then 12 sentences |

Each **GPU** step needs the owner's yes before it starts. The estimates come from logged times of kernels
k08 and k12; treat them as plus or minus 30%.
