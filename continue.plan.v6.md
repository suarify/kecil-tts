# Plan v6: fix the damp sound with accurate pitch, ship as Suaradee v0.3

> **Superseded the same day.** A stricter CPU test (`scripts/f0_refine_test.py`) did not confirm the pitch
> fix: refining on one band does not clean up another, so most of the gap in section 1 was selection bias.
> kmb-13 was not run. The live route is the zero-training noise-to-harmonic move (`scripts/tonal_knob.py`,
> listening page `runs/listen/v03.html`); see STATUS.md. If the owner picks a setting, bake it into the
> ONNX export and ship that as v0.3. Sections below are kept for the record.

2026-11-15. Follows `continue.plan.v5.md`. Goal: the quickest fix for the damp / smeared sound, proven on a
small run before anything is shipped.

## 1. What is wrong (measured on CPU, `scripts/f0_check.py`, 24 clips each)

The decoder builds voiced sound from sine waves placed at the pitch and its multiples. In training we tell
it the pitch of each Chatterbox clip, measured with pYIN. That pitch is slightly off, and the error grows
with the harmonic number: 1% off at the pitch is 15% of the gap between harmonics at 3 kHz.

"Contrast" below is how much more energy sits on the multiples of the pitch than halfway between them, in
the 2-4 kHz band. Higher is cleaner.

| Audio | Contrast with the stored pitch | Contrast with the pitch nudged to fit | Frames needing a nudge over 1% |
|---|---|---|---|
| Chatterbox clips (training target) | 3.0 dB | 7.4 dB | 38% |
| Kokoro audio, pitch also from pYIN (control) | 4.8 dB | 10.1 dB | 38% |
| Suaradee v0.2 output | 2.8 dB | 5.2 dB | 17% |

- The training target has clear harmonics (7.4 dB), but through the stored pitch the decoder sees only 3.0 dB.
  Its sine waves land between the real harmonics on many frames, so the loss teaches it to use noise there.
- v0.2 reproduces what it was shown: 5.2 dB at best, against 7.4 dB in its target.
- The control shows the same error on clean synthetic speech, so the fault is the pitch measurement, not the
  Chatterbox audio.
- In the v0.2 training log the harmonic loss stayed at 0.53-0.55 from step 2,700 to step 6,000 (never fell).
- Paradee does not have this problem: its pitch is the exact one its teacher used to make the audio.

This is strong evidence, not proof. The small run below is the proof.

## 2. The fix

Refine the stored pitch per frame: nudge the pYIN value by up to 3% to where the harmonics of the clip
actually are (highest contrast in 1-4 kHz), only on frames where that contrast is strong, then smooth the
nudges over 5 frames. Nothing else in the recipe changes. `ms_rows.py` gets a `--f0-refine` flag and stores
both tracks, so one set of rows serves both arms.

## 3. One Kaggle run (kmb-13), with a stop gate in the middle

| Stage | What | Time on a T4 (from the k12 log) |
|---|---|---|
| Rows | 742 clips, letter timings, stored and refined pitch | 11 min |
| Arm P | decoder, v0.2 recipe, stored pitch, 6,000 steps (control) | 8 min |
| Arm R | decoder, same recipe and seed, refined pitch | 8 min |
| Gate | decoder-only scores on the 60 held clips | 5 min |
| Only if the gate passes | text side on refined pitch, export, end-to-end scores, samples | 20 min |

Stops after about 0.55 GPU-h if the gate fails; about 0.9 GPU-h if it passes. Total since the pivot would
be about 7.6 GPU-h.

## 4. Gates (fixed now)

- **G1 clarity (arm R against arm P, decoder only, 60 held clips):** contrast at least 1.5 dB higher, and
  flatness in 2-6 kHz at most 0.66 (v0.2 is 0.69, the clips are 0.62-0.635).
- **G2 no harm:** decoder-only median CER at most 1 point worse than arm P.
- **G3 end to end (only after G1 and G2):** test median CER at most 6% (v0.2 is 4.8%).
- **G4 ears:** the owner compares v0.2 and the candidate on 12 sentences. Ship as v0.3 only on his yes.

If G1 fails, pitch accuracy is not the main cause; stop and go to section 6.

## 5. What to expect

- Best case the output reaches the clarity of the Chatterbox clips (7.4 dB contrast). It will not reach
  Paradee or Kokoro (10 dB), because the Chatterbox clips themselves are less crisp.
- Wrong words should not change much; this run does not target them.

## 6. If it fails, or to go further later

1. Adversarial stage sized to fit (about 3 GPU-h).
2. More Chatterbox data (about 1.5-2 GPU-h for 4x the audio).
3. Crisper target audio (enhance or re-render the clips) to get past the 7.4 dB ceiling.
