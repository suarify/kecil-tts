# kokoro-malay-bella — plan v5: validated diagnosis, one bounded run

Written 2026-11-15. The owner asked for the plan to be validated before anything else runs, with two
targets only: **wrong words** and **the damp sound**. Everything below was measured on the owner's CPU
with files already on disk. No GPU time was used. `STATUS.md` has the history; v4 describes the pipeline.

## 1. What was measured

**Where the wrong words come from** (`scripts/diagnose_k09.py`, 30 held-out sentences, same ASR on all,
Whisper-small, so numbers are higher than the Whisper-large figures in earlier reports):

| Version of the same sentence | Median CER | Sentences at or under 8% |
|---|---|---|
| Chatterbox clip (the ceiling) | 7% | 53% |
| 8M decoder fed the clip's measured timings, pitch, loudness (text side bypassed) | 18% | 20% |
| Full 8M model from text (what ships) | 31% | 7% |

So the decoder adds about 10 points and the text side about 13 more. Both halves need work.

**Where the damp sound comes from** (same files):

| | Energy above 4 kHz | Energy 1.5–4 kHz | Noise-likeness 2–6 kHz |
|---|---|---|---|
| Chatterbox clip | 0.040 | 0.024 | 0.62 |
| Decoder only | 0.034 | 0.016 | 0.69 |
| Full model | 0.032 | 0.014 | 0.69 |

The decoder alone already has the whole problem; the text side adds nothing to it.
It is **not a tone problem**: the student's average spectrum is within 1–2.5 dB of its target
(`scripts/eq_test.py`), so a fixed tone correction cannot remove it. The difference is in the
"noise-likeness": the harmonics are smeared. A comparison page exists (`runs/listen/eq.html`) but
expect it to sound only slightly brighter.

**Why the decoder smears** (`scripts/align_check.py`): inside each word, phoneme timings in the
training rows were guessed from Kokoro's proportions. Against letter-level alignment, 53% of phoneme
boundaries are off by more than one decoder frame (25 ms) and 25% by more than 50 ms.

**Does fixing the timing help? Tested without training** (`scripts/timing_test.py`, 24 clips, the
same pretrained decoder that has never seen these clips, fed both versions of the rows):

| Phoneme timings | Median CER | Mean CER |
|---|---|---|
| Guessed (what k09 trained on) | 13.2% | 15.0% |
| Letter-aligned | **7.7%** | **11.5%** |
| The clip itself | 0% | 3.7% |

Letter timing was better on 18 clips, worse on 5, same on 1. 167 of 169 words could be timed from letters.

## 2. What is ruled out

| Tried | Result | Conclusion |
|---|---|---|
| More decoder steps (k09: 12,000) | No gain after about 1,000 | The inputs limit it, not training time |
| Early stopping + smoothed targets + jittered inputs (k10) | Median CER 22% vs 9.7% | Wrong lever; dropped |
| Short adversarial stage (k11: 843 steps in 22 min) | CER worse, smear unchanged | Too slow on a T4 to be worth it now |
| Fixed tone correction | Spectrum already within 1–2.5 dB | Cannot fix smear |

## 3. Two plan steps that were specified but never built

1. **v4 C0.1 — letter-level phoneme timings.** Now built (`paradee_ms/ms_rows.py --letters`) and
   tested above. Targets the decoder's share of wrong words, and the most likely cause of the smear.
2. **v4 C3.1–2 — give the text side a Malay prior** from Kokoro-read sentences before it learns from
   the 686 Chatterbox sentences. Its timing and pitch errors are 3–5 times higher on unseen sentences
   than on trained ones (0.195 vs 0.065, 0.56 vs 0.12): it has too few sentences. Kokoro rows cost
   minutes, and their sound never reaches the decoder. Targets the text side's share of wrong words.

## 4. The one run (kernel `kmb-12`), about 50 minutes of GPU, about 0.85 GPU-hours

1. Rebuild the 742 rows with `--letters` (10 min).
2. Decoder: k09 recipe, 6,000 steps, no jitter, no adversarial stage (9 min).
3. Text side, two variants, each exported and scored:
   A. k09 recipe (6,000 steps on Chatterbox rows);
   B. 3,000 Kokoro-read corpus sentences first (rows 5 min, 3,000 steps), then the Chatterbox rows
      (4,000 steps, lr 2e-4), checkpoint chosen on held-out `l_dur + l_asr`.
4. Score with Whisper-large: decoder-only on the 60 held-out rows; full model (A and B) on the 60
   held-out and 40 never-seen sentences; brightness numbers; median and capped mean reported.

**Gates, fixed now**

| Gate | Pass when | k09 value |
|---|---|---|
| D1 decoder word errors | decoder-only median CER at least 30% lower than k09's | to be measured in the same kernel with the k09 decoder |
| D2 decoder smear | noise-likeness ≤ 0.66 (target 0.62) and held spectral loss below 2.545 | 0.69, 2.545 |
| S1 unseen sentences | median CER ≤ 8% and capped mean ≤ 11% | 9.7%, 13.9% |
| S2 owner's ears | acceptable on at least 10 of 12 unseen sentences | not rated |

**Stop rule.** If D1 and D2 both fail, the timing explanation is wrong and nothing else is run: the
conclusion is that about 50 minutes of Chatterbox audio cannot train this decoder cleanly, and the
choice goes back to the owner (ship k09 as is, or fund more Chatterbox audio, which is the expensive
part at about 2.4 GPU-hours per hour of audio). If D1 or D2 passes but S1 fails, the remaining error
is the text side's, and the fix would be more sentences, also the owner's call.

## 5. What this run cannot do

- Sound clearer than the Chatterbox clips themselves. They are duller than Kokoro (0.040–0.055
  against 0.088 above 4 kHz) and Whisper misreads 7–9% of their characters.
- Guarantee the smear goes. The timing test shows intelligibility improves; the smear was not
  measured there because no training was involved.

## 6. Budget

Spent since the pivot: about 5.7 GPU-hours. This run: about 0.85. Total about 6.6, over the 6-hour
limit the owner set, so it needs the owner's yes. Kaggle quota left this week: about 22 hours.
