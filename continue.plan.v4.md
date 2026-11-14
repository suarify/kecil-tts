# kokoro-malay-bella — plan v4: the Chatterbox clip is the target (cap 5 GPU-hours)

Written 2026-11-14 after Gate R. **This replaces sections 2–5 and 8 of `continue.plan.v3.md`.**
Still valid: `continue.plan.md` sections 1–5 (goal, repo map, facts, rules, traps — with the stub
fix in v3 section 7), 10, 11 and 13 (export, runtimes, acceptance). `STATUS.md` is the log.

---

## 0. One page

**Gate R result (owner, 14 sentences rated).** Borrowed-rhythm versions took 12 of 14 picks
(`TPd` 7, `TPf` 5, plain `F1` 2), so rhythm does help. But the owner's verdict was:
*the Chatterbox reference sounds Malay; most of the Kokoro options still do not; focus on the
Chatterbox reference and drop Dora.* Three sentences had every Kokoro version marked not acceptable.

**What that means.** Rhythm was part of the gap but not all of it. The rest is how the sounds
themselves are made, and that lives in the audio. A Kokoro teacher cannot supply it. So the
student has to learn from the **Chatterbox audio itself** (the voice-converted clips, `T2v`,
which is what the reference player was).

**How, without the expensive route.** The 8M student's two halves need, for each training clip:
phoneme timings, a pitch curve, a loudness curve, Kokoro's text features, and the audio. v3
measured the first two from Chatterbox clips. v4 measures loudness too and uses the Chatterbox
audio as the target. Kokoro-82M is still not fine-tuned; it only supplies text features, which
need no audio.

**Checked on the owner's CPU before planning** (`scripts/direct_local.py`, 8 clips, 5 minutes):

| Check | Result |
|---|---|
| Pitch measured on audio vs Kokoro's own pitch values | ratio 1.001 — same convention |
| Loudness measured on audio vs Kokoro's own values | correlation 0.97, same spread, constant offset −1.05 (subtract it) |
| Forced alignment of Chatterbox clips | every clip tried: 32 raw clips and 8 voice-converted ones |
| Pretrained 8M decoder driven by inputs measured from Chatterbox clips, **no training** | intelligible: Whisper-small CER 14% (the Chatterbox clip itself 5%, decoder with Kokoro's own inputs 3%) |

So the measured inputs are in the right format and scale; training starts from a working point.
**Not yet shown:** that about 1.4 hours of Chatterbox audio is enough for the small model to copy
its sound. Gate C2 below measures that before the rest is built on it.

**Budget: 5 GPU-hours new, enforced by time limits** (0.85 h spent earlier on the search).

| Step | What | Cap |
|---|---|---|
| C1 | 1,100 Chatterbox clips + voice conversion, then timings, pitch, loudness | 2 h 50 min |
| C2 | Train the decoder on them | 65 min |
| C3 | Train the text side (Kokoro rows first, then Chatterbox rows) | 55 min |
| C4 | Export, score | 10 min |
| E, F | Runtimes (CPU), owner's final listen | 0 |

**Blocker to clear first:** the last GPU kernel (`kmb-05-transplant`) has been queued on Kaggle
for hours without starting. Nothing in C1–C4 can run until Kaggle gives a GPU. The owner should
check the quota and active sessions on kaggle.com and cancel that queued run (its output is no
longer needed).

---

## 1. Step C0 — build and test locally (CPU, free)

1. **Phoneme-level timings.** Add an optional `spans` list to `G2P._letters` (Python only): at the
   top of each loop iteration append `(i, len(out))`, so every phoneme knows which letter produced
   it. With the aligner's per-letter times this gives a start time for every phoneme of a
   Malay-path word. For words read from the English lexicon or the exceptions file there is no
   letter mapping: split the word's time span in proportion to Kokoro's predicted durations (as
   the v3 proof did for every word). Two phonemes from one letter share its span equally.
2. **`paradee_ms/`** as in v3 section 4.1, plus `make_rows_cb.py` (below).
3. **Local chain test** with the 150 pilot clips already on disk (`runs/k03/pilot/T2v`):
   build rows → 3 steps of each trainer on CPU → export → run the ONNX file. Must pass before any
   kernel is pushed.

## 2. Step C1 — Chatterbox rows (kernel `kmb-06-cbrows`, cap 2 h 50 min)

**Text:** `data/text/cb_1100.tsv`, 1,140 rows from `corpus_train.tsv`, fixed seed, each with
20–75 phonemes (keeps clips near 4.5 s): 520 conversational, 280 general, 170 numbers, 120
code-switch, 50 questions. The first 40 rows are held out. Columns
`id, category, text, norm, teacher, phonemes`.

**Generation:** exactly the approved setting (`kaggle/k03_pilot/k03_pilot.py`): Chatterbox
Multilingual, `language_id='ms'`, reference `ref_ms.wav`, `cfg_weight=0.3`, `exaggeration=0.5`,
`temperature=0.8`, seed 1234, input = `teacher` column; then ChatterboxVC toward `ref_en.wav`.
No Whisper, no retry. Two worker processes if two GPUs are present. **Stop generating at 110
minutes of wall-clock**, voice-convert what exists (cap 35 min), then measure (cap 25 min).

**Per clip (`make_rows_cb.py`):**
- Align the **converted** clip with MMS forced alignment (letters of the `teacher` words). Cut the
  audio 150 ms after the last word; this removes the babble Chatterbox adds at the end.
- Phoneme durations in 25 ms frames from the letter times (C0.1). Word-gap rule from v3: a word
  runs to the next word's start; a gap of 150 ms or more is a pause given to the space and
  punctuation tokens. Leading silence goes to the first pad token, trailing to the last.
  Durations must add up to `round(seconds × 40)`; audio is cropped or zero-padded to
  `frames × 600` samples.
- `F0`: torchcrepe (`full`, Viterbi, periodicity 0.21) resampled to 80 values per second, 0 where
  unvoiced. `N`: `log(‖1e-5 + mel‖₂)` with torchaudio `MelSpectrogram(n_mels=80, n_fft=2048,
  win_length=1200, hop_length=300)` left at its default sample rate, **minus the offset measured
  on 50 Kokoro renders** in the same kernel (expected about −1.05).
- `t_en`: Kokoro's `text_encoder` output for the phoneme ids. `ids`, `pred_dur` (ints), `dur`
  (the same as floats), `audio` int16, `text`, `ps`. Same keys as Paradee's teacher rows.
- Reject: word-count mismatch, alignment failure, any Malay-path phoneme longer than 400 ms,
  clip length outside 0.6–1.8× Kokoro's prediction; then drop the worst 10% by alignment score.

**Gate C1:** ≥ 850 usable rows and ≥ 1.0 h of audio, including ≥ 30 of the 40 held-out.
**If it fails:** stop and report; do not top up without asking.

## 3. Step C2 — decoder (kernel `kmb-07-student`, cap 65 min)

`train_vhead.py --head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 --init-from init/dec_row0.pt --steps 6000 --eval-every 1000 --max-minutes 60 --tag _cb`
on the Chatterbox rows only (no Kokoro-rendered audio: its timbre is the Dora blend the owner
rejected). `--held 40`.

**Gate C2** (held-out rows, decoder fed the measured inputs):
`dtw_l1` at least 30% below its value at step 0, still falling or flat at the end, and Whisper
CER of the decoder's output within 3 points of the Chatterbox clips' own CER.
**If it fails:** this is the test of whether 1.4 hours is enough. Stop, keep the checkpoints,
report the curve and 8 sample clips, and ask the owner whether to generate a second batch
(same cost again) or stop. Do not run C3.

## 4. Step C3 — text side (same kernel, cap 55 min)

1. Kokoro rows for the prior (cap 15 min): 3,000 corpus sentences through the frozen teacher
   (`F1` voice, `gen_teacher_ms.py`), audio not stored. They teach phoneme features and a
   reasonable starting prosody; their timbre never reaches the decoder.
2. `train_text.py s+mlp --init-from init/text_row0.pt --data data/teacher --steps 3000 --max-minutes 18 --tag _pre`
3. `train_text.py s+mlp --init-from checkpoints/text_s+mlp_pre/last.pt --data data/teacher_cb --held 40 --lr 2e-4 --steps 3000 --max-minutes 20 --tag _cb`, checkpoint every 500 steps; keep the one with the lowest held-out `l_dur + l_f0 + l_n`.

**Gate C3** (40 held-out rows, free-running): word-interval error against the Chatterbox clips
≤ 0.18 (the plain Kokoro teacher scored 0.221), and `l_asr ≤ 0.10`.
**If it fails:** use the last checkpoint that passes `l_asr`, report the timing number honestly.

## 5. Step C4 — join, export, score (cap 10 min)

`export_paradee_v1_1.py` → fp32 + int8. On `test_200` with onnxruntime: Whisper CER, size, speed.
**Gate C4:** int8 ≤ 10 MB; CER ≤ 8% (Chatterbox itself scored 5–7% on these measures, so 6% was
too strict a bar for a copy of it); ≥ 15× real time on one thread.
Then runtimes and the final listen as in `continue.plan.md` sections 10, 11, 13. In the final page
the student is compared with the Chatterbox reference on sentences it never trained on.

## 6. Risks

| Risk | Likelihood | What limits it |
|---|---|---|
| 1.4 h of audio is too little for the decoder to copy the sound | medium | Gate C2 runs before C3; stop-and-ask, no silent overspend |
| Timings at phoneme level are too rough, speech comes out smeared | medium | Letter-level alignment for Malay words (C0.1); Gate C2's CER check |
| Text side overfits 1,060 sentences | medium | Pretrained weights, Kokoro-row prior, held-out checkpoint choice, Gate C3 |
| The student copies Chatterbox's muffled tone | certain, by design | It is the sound the owner chose; the voice-converted clips are the target |
| Kaggle does not give a GPU | unknown | Owner checks quota; everything before C1 runs locally |
| T4 slower than assumed | medium | Wall-clock caps on every stage |

## 7. Dropped

Branch K and branch T of v3 (Kokoro/Dora timbre) — rejected by the owner at Gate R.
The big-model milestone (Kokoro-82M + voice file) — it would have the rejected sound.
