# STATUS (continue.plan.md)

## 2026-11-14 Phase A, search done, waiting on Gate A (owner listens)
- Kernel `kmb-04-teacher-search` v1 ran about 51 min on a T4. Accent-score check: 100% (voice-converted Chatterbox clip closer to the Chatterbox reference than stock Bella on every sentence).
- Stock Bella, 16 search sentences: CER 6.0%, accent distance 0.2345.
- Best blends keep Bella's timbre only partly and take prosody entirely from the donor (wp = 0): ef_dora a0.5 (accent 0.161, CER 0.2%), pf_dora a0.5 (0.163), pf_dora a0.7 (0.164, tap r), ef_dora a0.3 (0.164). Prosody blended 0.5 and Chinese/French/Hindi/Japanese donors were worse.
- Finalists F1..F4 on 32 ref + 3 number sentences: CER F1 2.1%, F2 2.7%, F3 2.6%, F4 2.2% vs stock Bella 3.7%. Test_40 CER: F1 2.9%, F2 4.5%, F3 4.5%, F4 3.5%.
- final_a='ə' (saya -> sajə) made almost no automatic difference on F1 (CER 2.04% vs 2.14%, accent 0.1674 vs 0.1657). Included as F1_alt for the owner.
- Page: runs/listen/teacher.html. Gate A pending.

## 2026-11-14 Gate A result: FAILED (owner ears)
- Owner picks, 15 of 16 rated: Chatterbox voice-converted clip 12, stock Bella 2 (both on number sentences that had no Chatterbox clip), F1 1. Not acceptable: F2 3, F1 2, F3 2, F4 2, F1_alt 2, B0 3.
- Owner: Dora-based voices "not so accurate" on the last few sentences (code-switch and numbers).
- Findings: n10 (RM87.20) was a NORMALISER BUG: the digit-context rule ("akaun ... RM87") ate the digits, so every system read "RM lapan tujuh". Fixed in Python and JS, test added, parity 2000/2000, commit 60b83c5. n05 (OTP) read "kodotipi": the donor prosody runs the letters together.
- Per section 12: skipped step 1 (widening the search; the automatic accent score already saturated at 0.161-0.177 while the owner still picked Chatterbox every time it was present). Running step 3 proof only (50-clip scale, here 32 sentences): kernel kmb-05-transplant = F1 + Chatterbox word timings (TPd) + Chatterbox pitch (TPf). Page: runs/listen/transplant.html.

## 2026-11-14 Plan validated and revised: continue.plan.v3.md (cap 5.25 GPU-h)
- Preflight (local CPU, scripts/preflight_paradee.py): Paradee v1.3 weights load into public code at 2de547e after slicing the style table to row 0. Text side 4.23M, decoder 3.65M, forward pass OK. Found and fixed a wrong misaki stub recipe.
- Kaggle kernel kmb-05-transplant stayed QUEUED for over an hour, so the transplant proof ran locally (scripts/transplant_local.py, MMS forced alignment + pYIN, 6 min on CPU): 32/32 clips aligned, clean audio. Pace error -0.021 (tempo already matches), word-interval error 0.221.
- First attempt used raw aligner word spans: tight spans, would be choppy. Rule changed: a word lasts to the next word's start unless the gap is >= 150 ms.
- Page for the owner: runs/listen/transplant.html (Chatterbox is a labelled reference, not pickable). Gate R pending.
- GPU spent since the pivot: 0.85 h (k04). kmb-05-transplant may still run when Kaggle gives it a GPU (about 0.3 h); its output is not needed.

## 2026-11-14 Gate R result and plan v4
- Owner, 14 of 32 rated: TPd 7, TPf 5, F1 2. Not acceptable: TPf 4, TPd 3, F1 3 (three sentences had all three rejected). Verdict in his words: the Chatterbox reference sounds Malay, most Kokoro options do not; focus on the Chatterbox reference, drop Dora.
- Decision: Kokoro/Dora timbre is abandoned (v3 branches K and T dropped). The student will be trained on Chatterbox voice-converted audio directly, with timings, pitch and loudness measured from the clips: continue.plan.v4.md, cap 5 GPU-h.
- Local check (scripts/direct_local.py, 8 clips, CPU): pitch convention ratio 1.001; loudness correlation 0.972, std ratio 0.986, offset -1.053; pretrained 8M decoder driven by measured Chatterbox inputs with no training gives Whisper-small CER 14.3% (Chatterbox clip 4.7%, decoder with Kokoro inputs 3.2%). Clips in runs/k06_direct.
- Blocker: kernel kmb-05-transplant has been QUEUED on Kaggle for hours; the polling job gave up after 60 min. No GPU step can start until Kaggle gives a GPU.

## 2026-11-15 k09, k10 and the "damp/phaser" sound
- k09 (686 train clips, 12k decoder steps, 6k text steps): test_200 CER 13.9% (median 9.7%, 42% of sentences <= 8%), held 22.4% (Chatterbox clips themselves 8.9%), 8.46 MB int8, 26x real time. Decoder plateaued within ~1,000 steps (held spec loss 2.545); text side overfit f0/N (train 0.12/0.37 vs held 0.56/2.34).
- k10 (early stopping on summed held loss, smoothed f0/N targets, jittered decoder inputs): WORSE. Test CER median 0.258 (k09 0.097), 5% of sentences <= 8%. Early stopping picked step 250 because the summed score is dominated by f0/N; durations got worse (l_dur 0.218 vs 0.195, l_asr 0.064 vs 0.052, length ratio 0.875). Lesson: select text checkpoints on l_dur + l_asr, not the sum. Also the jittered decoder trained to held spec loss 2.696 vs 2.545. Whisper hallucinations ("Terima kasih kerana menonton") inflate mean CER to 0.8; always report median and capped mean.
- Owner listened (k09): sounds phasey / wet / damp. Measured: student energy above 4 kHz is 0.034 vs 0.055 for its Chatterbox target and 0.088 for Kokoro; 1.5-4 kHz 0.014 vs 0.024; 2-6 kHz flatness 0.69 vs 0.62 (more noise-like). int8 = fp32, so not quantisation. Cause: spectral-only decoder loss, no adversarial stage.
- k11 (kaggle/k11_gan): GAN stage on the decoder from the k09 checkpoints (dataset kmb-k09-ckpt), rows from k10 output or rebuilt from cb_clips, text side kept from k09. Reports brightness stats.

## 2026-11-15 k10 rerun, k11 (adversarial stage), alignment diagnosis
- k10 second run confirmed the first: test median CER 22.4% vs k09 9.7%. k09 stays the base.
- k11: adversarial decoder stage from k09 checkpoints. Only 843 steps fit in 22 min (1.57 s/step on a T4, discriminator loss still ~3.9 = untrained). Energy above 4 kHz 0.040 vs target 0.045 (k09: 0.034 vs 0.055 on its sample); 1.5-4 kHz 0.014 vs 0.023 and flatness 0.68 vs 0.62 unchanged. Test CER median 15.0% vs k09 9.7%. Verdict: no real gain; a full adversarial stage (Paradee used 15,000 steps) would cost about 6.5 GPU-h at this speed.
- Diagnosis on CPU (scripts/align_check.py, 40 clips, 1,204 inner phoneme boundaries of Malay words): the rows' within-word phoneme timings (Kokoro-proportional guess) differ from letter-level forced alignment by median 27 ms, mean 42 ms, 90th pct 98 ms; 53% are off by more than one decoder frame (25 ms), 25% by more than 50 ms, 10% by more than 100 ms. The decoder is trained with phoneme features that often do not match the audio, which explains both the smeared sound and the plateau at held spec loss 2.545 within ~1,000 steps. Plan v4 step C0.1 (letter-level phoneme timings) was specified but never implemented in paradee_ms/ms_rows.py.
- G2P._letters now takes an optional spans list (letter index per phoneme); default behaviour unchanged, Gate 1 tests pass.
- GPU spent since the pivot: about 5.7 h of the 6 h limit. Best model so far: k09 (runs/k09/ms_int8.onnx, 8.46 MB).

## 2026-11-15 Validation on CPU (no GPU): continue.plan.v5.md
- diagnose_k09.py (30 held-out, Whisper-small): Chatterbox clip median CER 7%, decoder with measured inputs 18%, full student 31%. Smear numbers identical for decoder-only and full student (flatness 0.69 vs target 0.62): the damp sound is the decoder's.
- eq_test.py: student average spectrum within +1 to +2.5 dB of its target at 1-4 kHz; a fixed EQ cannot fix the smear. Page runs/listen/eq.html.
- timing_test.py (24 clips, pretrained v1.3 decoder, no training): guessed phoneme timings median CER 13.2% / mean 15.0%; letter-aligned 7.7% / 11.5%; better on 18, worse on 5. ms_rows.py --letters times 167/169 words from letters.
- v5 proposes one run (kmb-12, about 0.85 GPU-h) with gates fixed in advance and a stop rule. Not started; needs the owner's yes (total would be about 6.6 GPU-h).

## 2026-11-15 Suaradee v0.1 shipped (k09 model) + v5 run started
- Model repo https://huggingface.co/lseanlon/suaradee-v0.1 (public): onnx/model_quantized.onnx 8.46 MB (k09 int8, WebGPU-safe patch via scripts/webgpu_patch.py), suaradee/ Python package + wheel in dist/, js/ transformers.js front end, examples/ (python, node, gradio), samples/, NOTICE (lexicon_e.json is CC BY-SA 4.0). Whole repo 11 MB.
- Static Space https://huggingface.co/spaces/lseanlon/suaradee-web: verified in a real Chromium: WebGPU (auto default) 3.27 s audio in 0.22 s after warm-up (6.4x on first call), WASM also works.
- Gradio Space NOT created: free HF accounts cannot host Gradio Spaces (402 Payment Required). hf/space_gradio is ready; examples/gradio_app.py ships in the model repo.
- Findings: pip install git+https://huggingface.co/... fails (HF git rejects pip's partial clone) -> wheel URL instead. WebGPU failed at /head/Mul_41 because ORT-web WebGPU Slice drops one element of a reversing slice; the one reversing random noise is now an Identity, plus float64 -> float32 and 64-bit slice bounds clamped. 125 head tensors have identical shapes on WASM and WebGPU.
- kmb-12-v5 (plan v5: letter-level timings, Kokoro-read text prior, gates D1/D2/S1) pushed and RUNNING.

## 2026-11-15 v5 result and Suaradee v0.2 shipped
- kmb-12-v5 (3,261 s): rows with letter-level phoneme timings (6,050/6,318 words timed from letters). Gates: D1 PASS (decoder-only median CER 5.9% vs 9.1% for the k09 decoder, 35% lower); D2 PARTIAL (held spectral loss 2.482 < 2.545 pass; flatness 0.695 vs 0.66 target FAIL, brightness unchanged); S1 PASS (variant A: test median 4.8%, capped mean 6.5%, 70% <= 8%; held median 6.3%); S2 (owner's ears) not yet rated.
- Variant A (k09 recipe text side) beat variant B (Kokoro-read prior): B test median 8.0%, held 9.3%.
- By type, held-out median CER: call-centre 2.9% (was 12.5%), numbers 2.9% (21%), general 12% (17%), code-switch 18% (17%).
- Published lseanlon/suaradee-v0.2 (A, WebGPU-patched, 8.46 MB, wheel 0.2.0); web Space now has a version selector, v0.2 verified on WebGPU in a browser (4.97 s audio in 0.34 s); clean-venv Python install verified.
- Still open: the damp/smeared sound (flatness 0.69 vs 0.62, >4 kHz energy 0.031 vs 0.045). The timing fix did not touch it; the decoder needs more than spectral-loss training (adversarial stage cost 1.6 s/step on a T4) or cleaner source audio.

## 2026-11-15 Gate S2 passed
- Owner listened to Suaradee v0.2 and accepted it ("Yes, acceptable"). All v5 gates are closed: D1 pass, S1 pass, S2 pass, D2 (smear) not achieved and documented as a known limitation in the model card.
- Open for a later release: the muffled/smeared sound (needs decoder work beyond spectral losses) and English-word accuracy.
