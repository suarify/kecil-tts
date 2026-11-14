# kokoro-malay-bella — plan v3 (validated, under 6 GPU-hours)

Written 2026-11-14 after Gate A failed. This file **replaces sections 6–9, 12 and 14 of
`continue.plan.md`** (the search, corpus, training, fallback and budget). Sections 1–5 (goal,
repo map, background facts, rules, traps) and 10, 11, 13 (export, runtimes, acceptance) of
`continue.plan.md` still apply, with the corrections in section 7 below. `STATUS.md` has the log.

---

## 0. One page

**Where we are.** The Malay front end works (normaliser + G2P in Python and JavaScript, parity
2000/2000). A Kokoro "teacher" voice exists that reads Malay accurately with no training
(`F1` = Bella's timbre blended 50/50 with `ef_dora`, prosody from `ef_dora`; CER 2.9% on
`test_40`). The owner found it acceptable on 13 of 15 sentences but **never preferred it to the
Chatterbox clip** (Chatterbox 12 of 12). What Chatterbox has and Kokoro lacks is Malay rhythm and
melody. The owner's own October notebook reached the same conclusion ("rushed").

**The v3 idea.** Keep Kokoro's clean sound and correct pronunciation. Borrow **only the word
timing and the pitch curve** from Chatterbox clips ("prosody transplant"), and teach those to the
8M student's text side. No Chatterbox audio is trained on, Kokoro is not fine-tuned.

**It was tested before being planned.** A 32-sentence proof ran on the owner's CPU (no Kaggle):
`runs/listen/transplant.html`. All 32 Chatterbox clips aligned (32/32), every rendered clip is
clean audio, and it took 6 minutes. Two measurements came out of it:
- **Overall tempo already matches.** Kokoro `F1` is only 2% faster than Chatterbox (2.99 s
  against 3.03 s per sentence on average). The problem is not speed.
- **The timing inside sentences differs.** Word-to-word intervals are off by about 22% on
  average (standard deviation of the log ratio 0.221, up to 0.41 on single sentences). This is
  the gap the transplant closes, and the number Gate T2b measures.

Whether that difference is what the owner hears is the one thing still unknown. Gate R answers it.

**The plan from here**

| Step | What | New GPU time (hard cap) | Owner |
|---|---|---|---|
| R0 | Listen to the proof page, pick a branch | 0 | 5–10 min |
| T1 | *(branch T)* 1,000 Chatterbox clips → word timings + pitch only | 2.4 h | — |
| T2 | Teacher corpus (8,000 plain + the transplanted rows), train the 8M student, export, score | 2.85 h | — |
| E | Python package + transformers.js version + tests | CPU | — |
| F | Final listen | 0 | 5 min |

**Branch T total: 5.25 GPU-hours, enforced by time limits inside the kernels.**
**Branch K** (skip T1, ship the plain `F1` rhythm): **2.85 GPU-hours.**
Already spent since the pivot: 0.85 h (teacher search).

---

## 1. Validation of the previous plan (`continue.plan.md`)

| # | Assumption in v2 | Evidence | Verdict | Change in v3 |
|---|---|---|---|---|
| 1 | Kokoro can be distilled into an 8M Paradee-style student | Released v1.3 weights **load into the public code after slicing the style table to one row** (text side 4.23M, decoder 3.65M, forward pass makes audio). Checked locally with `scripts/preflight_paradee.py`, torch 2.9.1 CPU. | **Holds, now verified** | Init from v1.3 row 0 is safe to rely on |
| 2 | A zero-training voice blend makes Kokoro read Malay accurately | Search kernel: CER on the search set 6.0% → 0.2%; `test_40` 2.9% (stock Bella 4.1%) | **Holds** | `F1` is the teacher's timbre and pronunciation |
| 3 | …and makes it sound Malay enough for the owner | Gate A: Chatterbox 12 of 12 picks when present; Kokoro blends never chosen over it | **Failed** | Add rhythm and melody by transplant (branch T) |
| 4 | The accent score can rank candidates | It separated stock Bella from Chatterbox perfectly, but all finalists sat at 0.161–0.177 while the owner heard a large gap | **Too coarse** | Not a gate any more. Gates use CER, a word-timing error, and ears |
| 5 | Gate A's rule ("most picks among Kokoro voices") | The Chatterbox anchor absorbed every vote, leaving 3 data points | **Badly designed** | The reference clip is now labelled and cannot be picked |
| 6 | Fallback 12.3 costs about 6 GPU-h on top of the main route | 2,000 clips with voice conversion, retries and Whisper = about 11–12 h in total | **Over budget** | 1,000 clips, no voice conversion, no Whisper, a forced aligner; training shortened to the pt-BR fork's proven step counts |
| 7 | The misaki stub recipe in section 5 | A package-level `__getattr__` breaks `from misaki import en` (`en.MToken` missing). Caught by the local preflight | **Wrong as written** | `misaki/__init__.py` must contain `from . import en, espeak`; only `en.py` and `espeak.py` get the `__getattr__` |
| 8 | Number reading is correct | `Baki akaun anda ialah RM87.20` was read "RM lapan tujuh": the digit-context rule ate the amount, in **every** voice the owner heard | **Bug, fixed** (`60b83c5`, test added, parity kept) | — |
| 9 | Kaggle is always available | The proof kernel sat in the queue for over an hour | **Schedule risk** | Small jobs run on the local CPU (`.venv`); GPU work is in two kernels only |
| 10 | Two T4s per session, seconds per training step, transformers.js loading the graph | Still not measured | **Open** | v3 budgets assume **one** GPU and cap every stage by wall-clock, so the total holds either way |

**Verdict.** The route (frozen Kokoro teacher → 8M student → ONNX → Python and transformers.js)
is sound. One premise was wrong: voice blending alone does not give Malay rhythm. v3 changes only
how the teacher's timing and pitch are produced, and cuts the training to fit the budget.

---

## 2. Gate R — the owner's listen (now)

```bash
python -m http.server 8765 --directory C:/Users/Acer/Downloads/kokoro-malay-bella/runs
```

Open `http://localhost:8765/listen/transplant.html`. Each sentence shows the Chatterbox clip as a
labelled reference, then three unlabelled Kokoro versions:
`F1` (plain), `TPd` (Chatterbox word timing), `TPf` (timing and pitch).

**Branch rule**
- **Branch T** if `TPd` + `TPf` together take ≥ 60% of the picks and the winner of the two has
  "not acceptable" on ≤ 20% of its rated clips. The one with more picks becomes the transplant
  recipe (`TPd` = timing only, `TPf` = timing + pitch).
- **Branch K** if `F1` takes the most picks, or the transplants sound broken.
- If nothing is acceptable to the owner, stop and ask one question: accept `F1`, or fund the
  original fine-tuning route (`plan.md`, 35+ GPU-hours).

---

## 3. Branch T, step T1 — donor timings (kernel `kmb-06-donor`, cap 2.4 h)

**Text.** `data/text/donor_1000.tsv`: 1,060 rows drawn from `corpus_train.tsv` with a fixed seed:
480 conversational, 260 general, 160 numbers, 110 code-switch, 50 questions. Columns
`id, category, text, norm, teacher, phonemes` (`teacher` = English words respelled with
`g2p/teacher_respell.json`; phonemes from the frozen G2P options). **The first 60 rows are the
held-out transplant set** (stratified the same way). Upload with the other text as dataset
`seanlon/kmb-teacher-text`.

**Generation.** Exactly the setting the owner approved (`kaggle/k03_pilot/k03_pilot.py`):
Chatterbox Multilingual, `language_id='ms'`, reference `refs/ref_ms.wav` from kernel
`kmb-01-baseline`, `cfg_weight=0.3`, `exaggeration=0.5`, `temperature=0.8`, seed 1234, input =
the `teacher` column. **No voice conversion, no retry pass, no Whisper.** Keep each clip in memory
as 16 kHz mono; write them into a few `.pt` files, not WAVs.
If `torch.cuda.device_count() == 2`, run two worker processes (one per GPU, half the rows each).
Stop generating at **125 minutes** of wall-clock whatever the count.

**Alignment** (GPU, seconds per clip): `torchaudio.pipelines.MMS_FA` with the CTC Viterbi from
`scripts/transplant_local.py` (same code as the owner's `kokoro-malay-voicetune` notebook).
Words = letters of the `teacher` column. Output per clip: word start and end times.

**Pitch:** `torchcrepe` (`full`, Viterbi decoding, 10 ms hop, periodicity threshold 0.21) on GPU.
Store the raw track (NaN = unvoiced).

**Reject a clip when** the word count differs from the phoneme string's, alignment fails, the
aligned span is outside 0.6–1.8× Kokoro's predicted length, or any word is outside 0.33–3.5× its
predicted length. Then drop the worst 10% by mean alignment log-probability (catches misreadings).

**Output:** `donor.pt` (id, word times, pitch track, alignment score), `donor_report.json`.

**Gate T1:** ≥ 700 usable clips including ≥ 45 of the 60 held-out.
**If it fails:** switch to branch K. Do not generate more.

---

## 4. Step T2 — corpus, student, export (kernel `kmb-07-distill`, cap 2.85 h)

Inputs: `seanlon/kokoro-malay-g2p`, `seanlon/kmb-teacher-text`, `seanlon/kmb-paradee-ms`,
and `kmb-06-donor`'s output (branch T). One GPU assumed; if two are present, run 4.3 and 4.4 at
the same time (`cuda:0` and `cuda:1`).

### 4.1 Code (`paradee_ms/`, built locally first)

As `continue.plan.md` section 8 B1, **plus**: `train_text.py` gets `--data DIR` (default
`data/teacher`) and `--held N` (default 200); `gen_teacher_ms.py` gets `--speed` and
`--donor donor.pt`. Before uploading, run the whole chain on the **local CPU** with 12 sentences
and 3 steps per trainer (`.venv`, about 15 minutes): generate shards → `train_text` → `train_vhead`
→ `export_paradee_v1_1.py` → run the ONNX file in onnxruntime. This catches plumbing errors for
free; do not push the kernel until it passes.

### 4.2 Teacher rows (cap 40 min)

- Teacher voice: `voices/ms_bella.pt` = `F1` (acoustic half 0.5 `af_bella` + 0.5 `ef_dora`,
  prosody half `ef_dora`). G2P: no stress marks, `r`, glide diphthongs, `final_a='a'`.
- **Plain rows:** 8,000 sentences: `test_200.tsv` first (held-out), then the 7,500 training
  rows and 300 extra general rows. Durations are Kokoro's own divided by `speed`, where
  `speed = exp(mean pace error)` from `donor_report.json` (branch K: `speed = 1`). The proof
  measured a pace error of −0.021, so expect `speed ≈ 0.98`; if the donor set gives a value
  outside 0.9–1.1, stop and look at the alignment before continuing.
- **Transplant rows (branch T):** for each usable donor clip, as in `scripts/transplant_local.py`:
  1. A word lasts from its start to the next word's start. If the gap to the next word is
     ≥ 150 ms it is a pause: the word gets a 50 ms tail and the rest goes to the space and
     punctuation tokens. (The aligner's word spans are tight; using them raw makes choppy speech.)
  2. Inside a word, Kokoro's own phoneme durations are scaled to fit (scale limited to 0.4–3.0).
  3. Pitch (`TPf` only): bridge unvoiced gaps in the donor's log-pitch, smooth over 50 ms,
     resample to Kokoro's 12.5 ms grid, shift to the teacher's median pitch, and **write it only
     on frames Kokoro itself voices** (its own F0 > 50 Hz under the new timing).
  4. Energy `N` comes from Kokoro's predictor under the new timing. Audio is rendered by Kokoro's
     decoder. Row keys are the same as plain rows; `dur` = the new durations as floats.
  Shards go to `data/teacher_tp/`; the first 60 rows of its shard 0 are the held-out rows.

### 4.3 Text side (cap 50 min)

1. `train_text.py s+mlp --init-from init/text_row0.pt --data data/teacher --steps 6000 --max-minutes 38 --tag _ms`
2. Branch T: `train_text.py s+mlp --init-from checkpoints/text_s+mlp_ms/last.pt --data data/teacher_tp --held 60 --lr 1e-4 --steps 1500 --max-minutes 12 --tag _tp`, saving a checkpoint every 250 steps.
   Keep the checkpoint with the lowest held-out `l_dur + l_f0`.

**Gate T2a (plain):** `l_dur ≤ 0.06`, `l_f0 ≤ 0.07`, `l_asr ≤ 0.09` on the 200 held-out rows.
**Gate T2b (branch T):** on the 60 held-out transplant rows, the student's word-timing error
(standard deviation of log(predicted ÷ donor word-onset interval)) is **≥ 25% lower** than the
plain student's, and `l_asr` has not risen by more than 15%.
**If T2b fails:** ship the plain student from step 1 (that is branch K's model) and say so.

### 4.4 Decoder (cap 70 min)

`train_vhead.py --head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 --init-from init/dec_row0.pt --steps 5000 --eval-every 1000 --utmos --max-minutes 70 --tag _ms`
on plain and transplant shards together (symlink both sets into one folder). 5,000 spectral steps
with no adversarial stage is the pt-BR fork's recipe. Use time left under the cap for more steps.

**Gate T2c:** `dtw_l1 ≤ 0.62` and not rising over the last two evaluations.

### 4.5 Export and score (cap 10 min)

`export_paradee_v1_1.py` → fp32 and int8 ONNX. Score both with onnxruntime on `test_200`:
Whisper CER, and speed on one CPU thread.

**Gate T2d:** int8 ≤ 10 MB; int8 CER ≤ teacher CER + 1.5 points and ≤ 6%; ≥ 15× real time.

---

## 5. Branch K

Section 4 only, without `--donor`, without step 4.3.2, `speed = 1`. Cap 2.85 h, usually less.
The result is `F1`'s rhythm in an 8 MB model.

---

## 6. After training

- Export folder, Python package, transformers.js version, Node parity test, demo page:
  `continue.plan.md` sections 10 and 11, unchanged.
- Final listen and release: `continue.plan.md` section 13. In the final page the Chatterbox clip
  is a labelled reference, not a choice.
- The big model (Kokoro-82M + `ms_bella.pt` + Malay G2P, section 7 of `continue.plan.md`) stays
  optional and costs no GPU time.

---

## 7. Corrections to `continue.plan.md`

1. **Section 5, misaki stub:** `stubs/misaki/__init__.py` = `from . import en, espeak`.
   `en.py` and `espeak.py` each = `def __getattr__(name): return type(name, (), {})`.
2. **Section 6 (Gate A)** is closed: failed; `F1` kept as timbre and pronunciation.
3. **Section 9:** the adversarial stage (C3) is dropped. Step counts and caps are those in
   section 4 above.
4. **Section 12** is replaced by this file.
5. **Rule 3** stands: no Chatterbox audio is ever a training target. Only its word timings and
   pitch curve are used.
6. **Budget (section 14):** replaced by section 8 below.

---

## 8. Budget

| Stage | Cap (one T4) | How it is enforced |
|---|---|---|
| T1 generate 1,000 donor clips | 125 min | wall-clock stop in the generation loop |
| T1 align + pitch + report | 20 min | — |
| T2 teacher rows | 40 min | row count fixed; ≥ 15× real time checked on the first shard |
| T2 text side | 50 min | `--max-minutes` |
| T2 decoder | 70 min | `--max-minutes` |
| T2 export + scoring | 11 min | 200 sentences |
| **Branch T total** | **5 h 16 min** | |
| **Branch K total** | **2 h 51 min** | |

If a stage hits its cap, it stops with what it has and the gate decides. No stage is re-run more
than once; a second failure of the same gate means stop and report.

---

## 9. Risks

| Risk | What limits it |
|---|---|
| The owner does not prefer the transplant | Gate R happens before any GPU time; branch K remains |
| 1,000 sentences are too few to learn the rhythm | Gate T2b measures it on held-out rows; fallback is the plain student, already trained in the same run |
| Borrowed pitch makes artefacts | Pitch is written only where Kokoro voices; `TPd` (timing only) is a separate, safer recipe the owner can choose |
| T4 slower than assumed | Every stage is capped by wall-clock; gates decide on what was produced |
| Kaggle queue delays | Only two GPU kernels; everything else runs locally |
| Chatterbox babble corrupts timings | Per-word and per-clip ratio limits, plus dropping the worst 10% by alignment score |
