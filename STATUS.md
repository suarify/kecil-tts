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

## 2026-11-15 Deployed under the Suarify account
- scripts/publish_hf.py retargets cards, examples, web demo and the wheel's default repo, then uploads with the stored token profile "Suarify Malaysia" (user Suarify).
- Live: Suarify/suaradee-v0.1, Suarify/suaradee-v0.2 (models) and Suarify/suaradee-web (static Space, v0.1/v0.2 selector). Verified: v0.2 on WebGPU in a browser (3.25 s audio in 0.25 s); clean-venv Python install from Suarify/suaradee-v0.2 wheel defaults to Suarify/suaradee-v0.2 and runs at 21x real time.
- The earlier lseanlon/ repos and Space were left in place (not deleted).

## 2026-11-15 Why Paradee v1.1 is clear, and plan v6 (pitch accuracy)
- Read the Paradee-8M-v1.1 card: head works at 80 frames/s from sine waves at the pitch plus shaped noise; 23.9 h of Kokoro-read audio; 40k spectral + 15k adversarial steps; its pitch input is exactly the one the teacher used.
- scripts/f0_check.py (CPU, 24 clips each): 2-4 kHz harmonic contrast of the Chatterbox clips is 3.0 dB with the stored pYIN pitch but 7.4 dB with the pitch nudged to fit (38% of strong frames need more than 1%). Same error on Kokoro audio (4.8 vs 10.1 dB), so it is the pitch measurement. v0.2 output: 5.2 dB at best. The harmonic loss in the k12 log never fell (0.53-0.55).
- continue.plan.v6.md: refine the stored pitch per frame, one gated run kmb-13 (0.55 GPU-h to the gate, 0.9 GPU-h in full), candidate for v0.3. Not started; needs the owner's yes.

## 2026-11-15 Pitch fix NOT confirmed; zero-training noise-to-harmonic test instead (owner listening)
- scripts/f0_refine_test.py: refining the pitch on 1-2.5 kHz does not raise the contrast on the unseen 2.5-4 kHz band (2.2 vs 2.3 dB); fitted on 1-4 kHz it gives 3.1 dB against a "best single-frame nudge" of 7.2 dB. Low-band and high-band nudges correlate only 0.5. So most of the gap reported by f0_check.py was selection bias (best of 61 nudges). Plan v6's kmb-13 run was NOT started; no GPU used.
- scripts/tonal_knob.py (CPU, v0.2 PyTorch weights from the kmb-12 output, now in runs/k12/ckpt): in voiced frames above 1.5 kHz the v0.2 decoder puts about half of the energy (46-51%) in its noise part. Multiplying that noise gain by `keep` and raising the harmonic gain to hold the band energy gives flatness 2-6 kHz 0.693 (keep 1 = v0.2) -> 0.649 (0.5) -> 0.625 (0.25); Chatterbox clips are 0.62-0.635. Energy above 4 kHz 0.020 -> 0.023.
- 20 unseen test sentences at keep 1 / 0.5 / 0.25: runs/listen/v03.html (built by scripts/listen_v03.py). Waiting for the owner's ears. If he picks a setting, bake it into export_paradee_v1_1.py (HeadX) and ship as v0.3; text side and wrong-word rate are unchanged by construction.

## 2026-11-15 Owner verdict on the noise settings; plan v7
- Owner listened to runs/listen/v03.html: the three settings (v0.2, half noise, quarter noise) "no different the quality". So the 2-6 kHz flatness measure is not what he hears as damp; do not gate on it again.
- Decisions: v0.2 stays the release; its card gets more samples, a Kokoro comparison and streaming examples; test whether a no-training effect can fix the damp sound (he will listen); consolidated CER plan and a second voice (am_michael).
- All of it is specified in continue.plan.v7.md (packages A effect, B card, C CER, D Michael) for another agent to execute. Nothing in it has been started. No GPU used today after kmb-12.

## 2026-11-15 Package B started (plan v7); B1 streaming done locally
- Suarify HF token restored as profile "Suarify Malaysia" in ~/.cache/huggingface/stored_tokens (publish_hf.py unblocked); lseanlon token still the default.
- python/suaradee/tts.py: pieces() + _split_commas() (comma-only cutting, first piece <= 60 phonemes, later <= 160; a comma-free piece is never cut, so piece boundaries of comma-free texts are unchanged) and stream() generator; __call__ is now stream() concatenated. JS port in web/suaradee.js (splitCommas/pieces/async *stream), speak() unchanged in behaviour.
- Tests: python/tests/test_suaradee.py +5 stream tests (11 pass, model runs/k12/ms_A_int8_gpu.onnx); web/tests/ref.json now carries pieces and was regenerated against export/ms_bella_8m (the model the Node test loads; the old fixture was built from a different file); node_parity.mjs checks pieces and stream==speak; NODE PARITY PASS (envelope corr 1.000 on all 5).
- Examples hf/stream_python.py (sounddevice) and hf/stream_node.mjs (16-bit PCM to stdout, header claims a large size); Space gets a Stream switch: chunks scheduled gap-free on one AudioContext, time to first audio shown (hf/space_web/index.html).
- Measured time to first audio, 3-sentence paragraph, after warm-up: Python 0.12 s (total 0.41 s, blocking call 0.42 s), Node 0.08 s (total 0.27 s). Browser WASM/WebGPU numbers to be measured on the live Space in B4.
- Package bumped to 0.2.1. Disk was 100% full; npm-cache 4.65 GB + old Temp dirs cleaned, now 2.5-3 GB free.
- Next: B2 Kokoro comparison kernel (GPU 0.2 h, owner yes needed), B3 card, B4 publish.

## 2026-11-15 Package A started; A1 built, waiting on the owner's ears
- B2 (Kokoro comparison kernel) paused: the owner chose to run package A first ("kokoro has no malay"). B1 streaming stays done.
- rows/shard_0000.pt (742 rows, held = first 60) pulled from seanlon/kmb-12-v5 to runs/k12_rows (250 MB).
- scripts/listen_damp_where.py built runs/listen/damp_where.html: 8 held-out rows x 4 open-labelled versions (1 Chatterbox clip, 2 v0.2 decoder fed the clip's measured inputs via t_en @ alignment(pred_dur) with measured F0/N, seed 0; 3 v0.2 from text on the int8 ONNX; 4 Kokoro af_bella on the row's phonemes, local CPU). 32 wavs in runs/listen/damp_where. Owner ticks every version that sounds damp; headphones.
- scripts/damp_profile.py (30 held-out rows per system, CPU) wrote runs/listen/damp_profile/profile.json + profile_table.txt. Medians: rolloff 95% energy 961 Hz (Chatterbox) / 732 (decoder) / 879 (v0.2) / 1576 (Kokoro); energy >4 kHz 0.01 / 0.01 / 0.01 / 0.04 share; onset rise 2-8 kHz 17.0 / 10.2 / 8.9 / 19.8 dB per 10 ms; decay 10 dB 0.03 / 0.07 / 0.06 / 0.04 s; pitch span 6.2 / 6.0 / 5.4 / 5.5 st. First suspect: the slow, soft onsets (8.9 vs 19.8 dB/10 ms) plus longer decay - consistent with the owner's "damp" even though the long-term third-octave spectra of 1/3/4 sit within about 1-3 dB above 500 Hz (Kokoro is 4-5 dB hotter at 2.5-6 kHz).

## 2026-11-15 A1 result: the damp comes from the predicted F0; f0_fix listening built
- Owner's damp_where picks (5/8 rated): version 3 (v0.2 from text) damp on 4, version 2 (decoder + measured inputs) damp on 1, Chatterbox and Kokoro never. Per the A1 table: decoder and training audio are fine; the loss is in the text side's predicted F0/N. A2 EQ effects skipped.
- scripts/f0n_diag2.py (30 held rows): predicted F0 inside frames the clip marks voiced has 17.4% dropouts (< 50 Hz -> unvoiced noise where there should be harmonics), 9.7% of voiced-to-voiced jumps > 25 Hz, 10-90 span 227 Hz vs measured 69 Hz. Predicted N is close to measured (span 11.4 vs 12.5).
- scripts/f0_fix.py: full text side + decoder rendered 4 ways, same noise seed, repairs are pure functions of the predicted contour (no clip info): A shipped, B gaps below 50 Hz between voiced stretches interpolated, C B + 7-frame median + clamp 70-350 + 5-frame N smooth, D C + span cap. runs/listen/f0_fix.html (radio: least damp, 8 held rows). Owner listening pending; if a variant wins, bake it into the ONNX text side and ship as v0.3.

## 2026-11-15 f0_fix result: no-training contour repair REJECTED; no-training route for the damp closed
- Owner listened to runs/listen/f0_fix.html (8/8 rated): A (shipped prediction) picked on ALL 8; B/C/D repairs 0. The gap-fill, smoothing and span-cap repairs all sound worse than the shipped contour, so the dropouts/jagged span found by f0n_diag2 are not what he hears as damp (or the repairs cost more prosody than they give back).
- Objective check: the fp32 render (A) matches the shipped int8 ONNX at 0.994-0.997 envelope correlation in 0-1 kHz, 2-8 kHz and loudness - quantisation/export is not the damp.
- Standing evidence: Chatterbox clip fine, Kokoro fine, decoder + measured inputs fine (1/5), shipped text side damp (4/5), contour surgery worse. Conclusion: closing the gap needs better F0/N PREDICTION, i.e. training work - the text side already shows poor held-out f0/N generalisation in k10/k12 logs (held 0.56/2.34 vs train 0.12/0.37). No-training fix for the damp: NOT FOUND, documented; do not propose further contour transforms.
- GPU budget: 7.77 h of 30 h used this week (resets 2026-11-21). Waiting on the owner: B2 comparison kernel (0.2-0.5 h) and/or a text-side retrain targeting f0/N generalisation (package C2/C5 style, about 0.5 h).

## 2026-11-15 Package C started: kernel kmb-14-clean pushed (C0 + C1 + C2 in one run)
- Owner chose package C (v0.3 to reduce CER); B2 comparison kernel stays paused.
- New dataset seanlon/kmb-rows-v02: rows/shard_0000.pt from kmb-12-v5 (250 MB, held = first 60) + the shipped runs/k12/ms_A_int8_gpu.onnx.
- Kernel seanlon/kmb-14-clean (T4, internet on): C0 = v0.2 scored on ALL 200 test rows per category (the missing baseline); teacher scan = Whisper large-v3 on all 742 clips -> teacher_clips.csv (also the raw material for C3 word lists); C1 = decoder learning curve on 25/50/100% of train rows (data-lever gate: 50->100 median gain >= 15%); C2 = drop clips > 15% CER or rate outliers, retrain decoder + text A (k12 recipe 6k+6k steps), export int8, score held 60 + all 200 per category. Gates fixed in the kernel: keep cleaning only if capped mean improves and no category is worse.
- Deviation from the plan noted: the '< 90% words timed' filter is not applied (that count is not stored in the rows; the CER + rate filters stand in). Version 2 push RUNNING after the dataset finished processing; v1 had been pushed before the dataset was ready (source dropped).
- Estimated GPU 1.5-2 h (Whisper transcriptions ~1,440 clips, 3 curve trainings, 1 retrain).

## 2026-11-15 C3.2 lexicon grower (CPU, while k14 runs)
- scripts/grow_lexicon_en.py: candidates from (1) error-driven word list --errors FILE (top priority, waits on k14 CSVs), (2) corpus tokens, (3) data/lex_src/en_10k.txt (google-10000). All need a misaki US pronunciation inside g2p/vocab.json.
- Hard lessons caught by gate tests: first draft counted 'general' category as English context and added loans like unit/rom/vitamin/media/piano with US pronunciation -> 10 golden rows changed, JS parity 1990/2000. Reverted; now ONLY code-switch sentences count as English context, plus bypass-proof MALAY_ALSO blocklist (countries, religion, honorifics tun/tan/sri/bin, hospital/meter/diet/mac/pas, ms_50k count >= 2000 never bypasses) and a proper-noun capitalization filter. Malay always wins: lex_en is checked BEFORE Malay in ms_g2p, so a wrong entry corrupts the word.
- Applied +6 verified words (submission arrange quotation babirusa melatonin positron): lexicon_en 427 -> 433, web copy synced, gate 5/5, parity 2000/2000. The planned growth to ~3000 needs the error-driven list from kmb-14-clean (Whisper mishearings) - frequency lists alone add nothing safe.
- Brand names (whatsapp/shopee/lazada/...) were already in BRANDS; the 'missing' print line is pre-merge, no action.

## 2026-11-15 k14 v8: C0 + teacher scan landed, crash fixed, v9 running
- v8 completed C0 and the teacher scan, then crashed: rows.index(r) does == on tensors (RuntimeError size mismatch). Fixed with enumerate.
- C0 baseline v0.2 on ALL 200 test rows: overall median 7.7% / capped mean 10.9% (test_40 had said 4.8%/6.5% - the full set is harder). Per category (median/capped): conversational 5.9/7.0, numbers 4.5/5.3, codeswitch 23.0/27.8, question 1.7/3.7, short 2.0/10.3, general 11.2/12.9. Code-switch on the full set is worse than the 18.3% seen on the 40-row subset - C target <= 10% looks far; C gates unchanged.
- Teacher scan: 109 clips > 15% CER, 30 rate outliers, 120 of 742 dropped for the C2 clean set. Mishearings are mostly MALAY words in the Chatterbox teacher audio (encik->entik, isnin->isning, khamis->hamis, duga->dugu) - cleaning material, not lexicon; loans show the teacher speaks them Malay-style (booking->buking, confirmed->confirm) while reference text is English.
- Whisper digit-normalization artifact: empat/sepuluh 'missing' because Whisper writes 4/10 (RM->orm). Scoring stays as the plan fixed it; noted as bias that inflates numbers-category CER.
- teacher_clips.csv uploaded as version 2 of dataset kmb-rows-v02 (cached scan); kernel now loads it and skips ~30 min of Whisper transcription. v9 pushed RUNNING.
- C3.1 scripts/error_words.py written: token-level mishearing extraction with MISS/CLOSE kind and tags; teacher-scan pass saved to runs/c3_error_words.tsv. English-side candidates from the teacher scan are few; the real list comes from test_scores.csv when v9 lands.

## 2026-11-15 k14 v9: crash #2 (missing rows_clean dir), v10 running
- v9 confirmed the cached teacher scan works (LOADED CACHED, 742 rows, ~30 min saved) and re-ran C0: median 7.6 / capped 11.7, codeswitch 18.6 median (v8 said 23.0). Whisper decoding is not perfectly deterministic across runs -> gate comparisons must keep a noise margin; we now have two baseline samples.
- Crash: shutil.copy to {OUT}/rows_clean/ without os.makedirs. Fixed; v10 pushed RUNNING (full pipeline: C0 + clean retrain + curve + gates).

## 2026-11-15 k14 v10 COMPLETE (1.04 h GPU): C2 keeps with caveats, C1 says data IS the lever
- Kernel gate code had TWO bugs (results recomputed locally from test_scores.csv): per-category gate compared clean against clean (v02/clean columns identical -> 'no_cat_worse: false' is meaningless); C1 gain formula inverted (1 - c50/c100 instead of 1 - c100/c50), so 'c1_data_lever: false' is also wrong.
- REAL C2 result (capped mean, 200 rows): v02 11.4% -> clean 9.7% (-15.3% relative), median 7.5% -> 6.2%; 92 rows better / 68 worse / 40 same. Per category: general 14.7 -> 10.0 (big win, n=70), numbers 8.3 -> 6.3, short 14.3 -> 10.4, question 3.0 -> 2.7, conversational 6.3 -> 7.1 (nominally worse; baseline noise band 6.3-7.3 across runs), codeswitch 23.0 -> 24.6 (noise band 18.6-27.8).
- Honest note: conversational + codeswitch are within run-to-run Whisper noise, BUT per-row regressions exist (te00094 'Driver dah on the way ke Kota Kinabalu': v02 0.00 -> clean 0.26 'andi wayi'; te00063 numbers digit drift). Likely cause: the 120 dropped clips included useful code-switch/numbers clips.
- C1 learning curve (decoder-only, held 60 median): 25% 12.8% -> 50% 9.1% -> 100% 6.2%. TRUE 50->100 gain = 32% >= 15% -> data IS the lever -> C4 (more generated clips, ~3.5 GPU-h/1000) is justified.
- Brightness watch: over-4k energy 0.0398 (v02) -> 0.0313 (clean) - less harsh high-frequency energy, no smear regression.
- Held 60 (clean model): median 5.3% / capped 10.5%.
- C3.1 error list from the clean model's test rows (runs/c3_error_words_clean.tsv): remaining English errors (way, lunch, on, in, the, status, booking, confirmed) are ALREADY in lexicon_en -> text side is not the bottleneck anymore; audio/decoder side for English phrases is, which C4 targets. Malay-side artifacts are Whisper digit normalization (setengah->4.30) and teacher-accent words.
- Artifacts in runs/k14_final: ms_clean_int8.onnx, dec_clean_last.pt, textA_clean_last.pt, report.json, test_scores.csv, rows_clean/shard_0000.pt. GPU week usage ~9.8 h of 30 (resets 2026-11-21).
- PENDING OWNER DECISION: (a) accept clean model as v0.3 despite nominal conversational/codeswitch noise-band regressions, or (b) treat C4 (more data, retrain bigger) as the real v0.3. Strict plan gate 'no category worse' technically fails; overall gate passes.

## 2026-11-15 C4 started (owner chose data-scale first): kmb-15-cbgen2 pushed
- Owner decision: go C4 first; v0.3 becomes the data-scaled release (learning curve said 50->100% data = 32% better).
- k15_cbgen2.py: 398 cb_1100 rows never generated by kmb-08 (id not in teacher_clips.csv) + 400 new corpus_train rows (general 200 stride-sampled, codeswitch 120 ranked by lexicon_en English-word count with template dedupe + shuffle - the first draft was 100% 'Driver dah on the way ke <city>' templates, rejected), numbers 80. Dual seeds 1234/5678 for new general+codeswitch only; Whisper large-v3 picks the better seed and drops a pair when both exceed 15% CER. Saves cb_raw2.pt (chosen raw audio, for package D second voice), cb_clips2.pt (voice-converted), chosen.csv, dropped.csv, gen_report.json. Splits across two workers if two GPUs are visible. Estimated ~4.5-5 h GPU; caps GEN 390 min / VC 90 min.
- k16_train_c4.py prepared (not pushed): unpack new clips -> ms_rows --letters -> merged set = held 60 + C2-cleaned old rows + C2-cleaned new rows -> decoder 12,000 steps + text 10,000 steps (--score dur_asr) -> export int8 -> score v0.2 + k14-clean + C4 on the same 200 test + 60 held rows in one session -> C4 gate (weak-category median CER >= 25% relative better than v0.2).
- Needs dataset seanlon/kmb-15-clips (k15 outputs + ms_clean_int8.onnx for the comparison) before k16 can be pushed.

## 2026-11-15 v0.3 PUBLISHED (owner: deploy current model as v0.3, C4 becomes v0.4)
- v0.3 = the k14 cleaned retrain (runs/k14_final/ms_clean_int8.onnx). Model repo: https://huggingface.co/Suarify/suaradee-v0.3 (8.46 MB onnx, 6 fresh samples rendered from this model with its own k14 CERs, wheel 0.3.0). Space https://huggingface.co/spaces/Suarify/suaradee-web updated: v0.3 is the default in the model box, blurb sums the accuracy gain.
- Verified before publishing: web stack probe (web/tests/verify_model.mjs -> transformers.js q8 renders 4 sentences, sane rms/peak), Python suite 11/11 against the v0.3 ONNX (SUARADEE_MODEL=export/ms_clean_3onnx), node parity PASS, and a REMOTE load test after upload (Suarify/suaradee-v0.3 loads in 2.7 s, 3.15 s audio).
- hf/model_card_v0.3.md: full 200-row table (overall 7.5/11.4 -> 6.2/9.7 median/capped, general 11.8/14.7 -> 7.0/10.0, numbers 5.0/8.3 -> 0.8/6.3, codeswitch unchanged within noise, muffled sound documented as unchanged).
- publish_hf.py VERSIONS gained the 0.3 entry; package default repo bumped to lseanlon/suaradee-v0.3 (python, js, examples, gradio app); pyproject 0.3.0.
- C4 naming: kmb-15-cbgen2 (generation) + kmb-16-train-c4 (train) will produce v0.4.
- k15 note: the run showed CANCEL_ACKNOWLEDGED once then RUNNING again (Kaggle preempted the session); watcher restarted. Plan CSV = 798 sentences / 1,118 generations.

## 2026-11-15 v0.3 WebGPU fix (owner reported 'Can't perform binary op' in the Space)
- Cause: the k14 export was published WITHOUT scripts/webgpu_patch.py. It contained 1 Cast-to-double op (pitch-phase accumulator); every shipped model since v0.1 was patched (runs/k12/ms_A_int8_gpu.onnx has 0 doubles, the k14 file had 1). ORT-web WebGPU cannot do that op -> '[Mul] /head/Mul_41 failed'.
- Fix: runs/k14_final/ms_clean_webgpu_int8.onnx = webgpu_patch applied (24 double items switched). Verified: audio corr 0.994 / same length / same rms vs unpatched; Python tests 11/11; node web-stack probe OK; Hub copy re-downloaded shows 0 doubles; remote load OK.
- Samples re-rendered from the patched model before re-upload (make_v03_samples.py now takes the model path, default patched). publish_hf.py VERSIONS 0.3 points at the patched file.
- Space verified in a real browser: 'Model ready on webgpu in 18.6 s', Speak -> 3.15 s audio in 2.27 s, zero kernel errors. Space blurb updated to v0.3.
- k16_train_c4.py now runs scripts/webgpu_patch.py right after the export and ships the patched file, so v0.4 cannot repeat this.
- Lesson: any new export must pass scripts/webgpu_patch.py before build_hf/publish (double ops + int64 slice bounds).

## 2026-11-15 k15 v1 failed AFTER generating everything: audio lost, v2 fixes it
- v1 generated all 1,118 clips in 102 min on TWO T4s (two workers, 559 clips each) and then crashed in the Whisper scoring step: chatterbox-tts upgrades transformers, and the ASR pipeline dict input {'raw': ...} then imports torchcodec (needs FFmpeg) -> RuntimeError. Same call worked in k14's image; the pip install changed the behaviour.
- Worse: workers wrote wavs to /tmp/raw2, so the generated audio died with the session. Nothing of the 1.7 h GPU was salvageable.
- v2 (pushed, version 4): (1) raw wavs are written straight into /kaggle/working/raw2 and workers skip existing files (resume), (2) Whisper runs as a direct model call (WhisperProcessor + WhisperGenerationModel.generate, language 'ms' with 'malay' fallback) with no pipeline, (3) cb_raw_all.pt saved before scoring as a second safety net, (4) VC chunks saved every 100 clips, (5) scores_all.csv written for C2/C3 reuse.
- k16 unchanged apart from the webgpu_patch step; it will consume cb_clips2.pt + chosen.csv + gen_plan.csv from dataset seanlon/kmb-15-clips.

## 2026-11-15 Space embed fix (owner: huggingface.co/spaces URL dead, static URL fine)
- Cause: HF serves the Space inside an iframe. There navigator.gpu exists but the adapter request fails, and the page is NOT cross-origin isolated (crossOriginIsolated false, SharedArrayBuffer undefined), so ORT threaded WASM could never start. Result: 'Loading model...' forever; only the top-level static URL worked.
- Fix in web/suaradee.js load(): probe navigator.gpu.requestAdapter() before choosing WebGPU, and when the page is not cross-origin isolated set env.backends.onnx.wasm.numThreads = 1 (single-thread WASM needs no SAB).
- Verified in the embedded HF page (fresh load): 'Model ready on webgpu in 9.1 s', Speak -> first audio 1.95 s, 3.15 s audio on webgpu, no errors. Node parity still PASS.
- Uploaded to the Space and to Suarify/suaradee-v0.3 js/suaradee.js.

## 2026-11-16 k15 v2 killed mid-run; v3 = batched, resume-first
- v2 result: part 0 finished all 559 of its clips; part 1's process DIED (no traceback, no OOM text in the log) after ~100 min, so the kernel aborted at 559/1,118. The downloaded log still shows v1's pipeline traceback with 'libavutil.so.60: cannot open shared object file' - confirms no FFmpeg for torchcodec.
- No traceback + death after ~550 clips points to the process being killed (memory growth), not an exception.
- v3 (written, compiles): existing wavs are seeded from dataset seanlon/kmb-15-raw and never regenerated; workers now run in BATCHES of 200 sentences with a fresh process (and model) per batch, so a leak cannot accumulate and one dead worker no longer aborts the run; per-clip SIGALRM timeout; scores written every 100; VC checkpoints every 50.
- Downloaded raw2 wavs from the v2 run are being packaged as seanlon/kmb-15-raw so v3 only generates the missing clips.

## 2026-11-16 k15 v2 post-mortem + v3 ready (safe filenames)
- v2 facts: only part 0 appears in the log (559 clips, 90.8 min); worker 1 died before its first clip. Most likely cause: both workers downloading the Chatterbox weights at the same time right after pip install. 559/1,118 done.
- Filenames used '<id>#<seed>.wav'; Kaggle output download passes the URL to requests, so '#' truncated it and every file arrived as 0 bytes. Recovered by monkey-patching requests.get to encode '#' as %23 in scripts/upload_k15_raw.py's download step (414+ files, real sizes ~115 MB).
- k15 v3 (compiles): filenames '<id>_<seed>.wav'; seeds from dataset; one-time model cache warm-up before spawning workers; workers get their own log files; chunk loop recomputes missing clips each round (a dead worker's sentences are retried instead of lost); auto-degrade to one GPU after a worker failure; a chunk that adds no wavs is skipped so it cannot loop.

## 2026-11-16 C4 generation COMPLETE (k15 v3), v0.4 training started (k16)
- k15 v3: 1,118 clips generated (560 resumed from dataset + 558 new), 767 chosen, 31 dropped (both seeds >15% CER), 767 voice-converted. All 8 worker exits 0; 102.6 min of audio; runtime 114.6 min on 2 T4s.
- Teacher-clip CER (Whisper on the new Chatterbox audio, median/mean): codeswitch 3.5%/7.8%, conversational 0%/8.0%, general 0%/4.3%, numbers 0%/12.1%, question 0%/4.0% - much cleaner than the old 742-clip set (109 clips had been >15%).
- Dataset seanlon/kmb-15-clips: cb_clips2.pt (767 clips, 68.6 min, 198 MB), chosen.csv, gen_plan.csv, ms_clean_int8.onnx (k14 comparison baseline).
- k16_train_c4 v2 RUNNING: builds rows for the 767 new clips (ms_rows --letters), merges with the C2-cleaned old rows (held 60 first), trains decoder 12,000 steps + text 10,000 steps (--score dur_asr), applies webgpu_patch after export, then scores v0.2 + k14-clean + v0.4 on the same 200 test + 60 held rows, and evaluates the C4 gate (general and codeswitch median CER >= 25% better than v0.2).
- k16 v1 crash: slug-based input globs ('/kaggle/input/kokoro-malay-g2p/**') did not match Kaggle's mount; reverted to unique-filename globs, which worked in every earlier kernel.

## 2026-11-16 k16 v2: training finished, export crashed; v3 running
- Merged set built: 1,281 rows = held 60 + 567 cleaned old + 654 new (from 767 chosen; 86 single-seed clips above 15% CER filtered, 19 alignment failures in ms_rows).
- Decoder: 12,000 steps, 17.2 min, held spec loss 4.46 -> 2.39. Text: early stop, best step 4000 of 8,000 trained, 15.3 min, dur_mae 0.744 frames, length_ratio 0.927.
- Crash: export_paradee_v1_1.py wants data/teacher/shard_*.pt for a held-out sanity check; k16 trained from data/merged, so the glob was empty (IndexError). Fix: copy the merged shard to data/teacher/shard_0000.pt before export, and copy dec/text checkpoints to /kaggle/working BEFORE the export so a later failure cannot lose training.
- k16 v3 pushed RUNNING (rows ~11 min + decoder ~17 + text ~15 + export/patch + 780 Whisper scores).
- Naming note from owner: project is being renamed to kecil-tts / kokoro (other agent running a Michael voice and rebuilding the Space). HF still has Suarify/suaradee-* and Suarify/suaradee-web with a two-speaker UI (Suaradee + Michael). publish_hf.py MUST NOT be run as-is: it would overwrite the other agent's Space. Plan: publish only the model repo Suarify/suaradee-v0.4 and hand over the one-line version option.

## 2026-11-16 k16 v3: training + export OK, died at webgpu_patch (script not in kernel); model salvaged
- Training: merged 1,281 rows; decoder 12,000 steps 17.2 min (held spec loss 4.46 -> 2.39); text early stop best step 3500 of 7,500, dur_mae 0.743 frames, length_ratio 0.920.
- Export OK: head check max |export - torch| 2.24e-07; paradee_int8.onnx 8.46 MB, 25.7x realtime 1 thread, length 64200 vs 64800 samples.
- Crash: k16 called scripts/webgpu_patch.py, but that file lives in the repo scripts/, not in paradee_ms, so the kernel's scripts/ copy lacks it. Checkpoints and the ONNX were already copied to /kaggle/working before the patch step, so the trained model was recovered from the kernel output instead of retraining.
- Salvage: patched locally -> runs/k16_final/ms_c4_int8.onnx (24 doubles removed, 0 left). corr 0.9968 vs unpatched, same length, rms 0.055 (louder than v0.3's 0.046). Web-stack probe OK, Python tests 11/11.
- k16 now inlines the patch (no external script) for future retrains.
- Scoring moved to kmb-18-score-c4 (dataset seanlon/kmb-c4-onnx): v0.2 + k14-clean + C4 scored on the same 200 test + 60 held rows, gate computed from v0.2 medians.
- Minor: Windows PowerShell wrote dataset/kernel metadata with a UTF-8 BOM, which the Kaggle CLI rejects ('Expecting value: line 1 column 1'); stripped the BOM before creating seanlon/kmb-c4-onnx.

## 2026-11-16 C4 RESULT (kmb-18, all three models scored in ONE session on the same rows)
- Overall (200 test rows, median / capped mean): v0.2 7.5/10.8 | v0.3 clean 5.9/10.5 | C4 5.5/9.5.
- Held-out 60: v0.2 4.9/8.9 | v0.3 5.6/10.3 | C4 4.8/8.5  (C4 best).
- Per category median (v0.2 / v0.3 / C4): general 9.9 / 6.4 / 5.9 | codeswitch 18.4 / 19.2 / 17.9 | conversational 5.2 / 4.9 / 4.3 | numbers 5.0 / 0.0 / 2.2 | question 1.7 / 0.0 / 0.0 | short 13.5 / 0.0 / 4.0.
- GATE (pre-registered): general >= 25% relative better than v0.2 -> 40.4% PASS. codeswitch -> 2.7% FAIL. Overall better -> PASS. So the C4 gate is NOT met.
- Per-row vs v0.3: 103 better / 95 worse / 62 same; vs v0.2: 115 better / 85 worse / 60 same. Big swings on individual sentences (e.g. te00093 0.00->0.44, te00092 1.00->0.14) show recogniser+model sentence-level variance dominates small-n categories.
- Interpretation: doubling the speech data gave a real but modest overall gain (capped mean -1.0 point vs v0.3, held-out -0.8) and cleared the general-text gate, but did NOT move code-switch (2.7% vs the 25% required). Code-switch is therefore not a data-volume problem; it is the English text/pronunciation side (C3) plus decoder rendering of English phonemes.
- Artifacts: runs/k16_final/ms_c4_int8.onnx (v0.4 candidate, webgpu-patched, verified), runs/k18_final/c4_report.json + test_scores_c4.csv.
- PENDING OWNER: publish v0.4 as a model repo (best overall + best held-out, but two small categories worse than v0.3 and the code-switch gate failed), or hold v0.3 and attack code-switch first (C3 lexicon + text side -> v0.5).

## 2026-11-16 v0.4 PUBLISHED (owner: 'just go with v0.4') - model repo only, Space untouched
- https://huggingface.co/Suarify/suaradee-v0.4 : 8.46 MB ONNX (0 float64 ops, webgpu-patched), 6 samples rendered from the v0.4 weights with its own k18 CERs (3 test 0%/0%/0%, 3 held 0%/5%/9%), wheel suaradee-0.4.0, card hf/model_card_v0.4.md (200-row table v0.2/v0.3/v0.4 + honest notes: code-switch unchanged at ~18%, muffled sound unchanged, 103/62/95 rows better/same/worse vs v0.3).
- Remote verification: hub ONNX re-downloaded -> 0 doubles; transformers.js loads the repo in 2.1 s and renders 3.08 s of audio.
- Loudness/brightness measured properly this time (24 sentences): v0.3 rms 0.0546 / 4 kHz share 0.0129; v0.4 rms 0.0456 / 0.0125. My earlier 'v0.4 is louder' line came from one sentence and was wrong; the card states the measured values.
- publish_hf.py gained --only and --skip-space so a model repo can be shipped without touching the Space (the Space is now Kecil-TTS 8M, rebuilt by another agent with a speaker picker + effects).
- Space still lists v0.3/v0.2/v0.1; its version list is JS data (suaradee: {versions: [...]}) and ?model= is ignored, so v0.4 needs one edit by the Space owner (handed over).

## 2026-11-16 Space redesign (owner request): EN/BM switch, dark mode, keywords, edge/IoT, Yati, v0.4
- Edited the live Kecil-TTS Space surgically (did not overwrite the other agent's effects lab / speaker picker): runs/space_live/index.html is the deployed copy, mirrored into hf/space_web/.
- EN/BM language switch (data-i18n dictionary, localStorage + ?lang=), dark mode (data-theme + CSS vars, localStorage + ?theme=), both in the header.
- Keyword cloud section + expanded meta keywords/schema (kecil tts, kokoro tts, smallest TTS, Malay TTS, edge/IoT/kiosk/vending voice, Yati...).
- New '3 - Runs anywhere: edge, IoT, kiosk, vending machine' card with two hand-drawn SVG scenes (kiosk with speech bubble 'Selamat datang', vending machine with 'Terima kasih') plus four honest capability bullets (10 MB, one CPU thread, Bahasa Melayu, same ONNX everywhere).
- Female -> Yati in the speaker picker; version list now offers v0.4 (latest) and defaults to it; footer links say 'Yati v0.4'; download filenames yati.wav / yati_effects.wav; module file also available as kecil.js (suaradee.js kept for compatibility), window.__kecil alias.
- Verified in the browser: loads v0.4 on WebGPU in 2.9 s, first audio 0.55 s (7.9x real time later), no console errors; BM and dark mode both render; fixed a dark-mode contrast bug where card headings and the Michael label used --navy (invisible on dark).
- Not done yet (needs owner): renaming the HF model repos (Suarify/suaradee-* -> Yati/kecil naming) and pushing source to github.com/suarify/kecil-tts (no suarify.my SSH profile available here).

## 2026-11-16 HF rename pass: suaradee-* -> kecil-tts-yati-* / kecil-tts-michael-*
- Moved (HfApi.move_repo, old URLs still 200 via redirect, verified): suaradee-v0.1/2/3/4 -> kecil-tts-yati-v0.1/2/3/4; suaradee-michael-v0.1/2 -> kecil-tts-michael-v0.1/2.
- Each card rewritten after the move: titles '# Kecil-TTS Yati vX' / '# Kecil-TTS Michael vX', all lseanlon/ links -> Suarify/, repo ids updated, prose Suaradee -> Kecil-TTS. pip package name and wheel filename stay suaradee-* (class Suaradee too) - renaming those is a breaking API change, left for an explicit decision.
- Space (Suarify/suaradee-web, URL unchanged on purpose) now offers Suarify/kecil-tts-yati-v0.4 (default), v0.3, v0.2, v0.1 and kecil-tts-michael-v0.2/v0.1; card links and footer follow the new ids; Space README models list updated.
- Keywords: added kokoro tts small, small kokoro tts, tiny kokoro tts, kokoro tts 8MB, kokoro voice Malay, smallest AI voice, low-resource TTS, CPU-only TTS, no-GPU TTS, Malaysian AI, suara AI Melayu, TTS Bahasa Malaysia (39 tags + meta keywords).
- Browser check after the rename: page loads Suarify/kecil-tts-yati-v0.4 on WebGPU, first audio 0.72 s, no console errors.
- Local sync: cards, publish_hf/build_hf repo naming, default repo ids in python/js/examples/gradio, hf/space_web mirror.
- STILL PENDING: GitHub push to github.com/suarify/kecil-tts (no suarify.my SSH profile in this session; only srv1745537.hstgr.cloud reachable).

## 2026-11-17 Muffle root-cause measurement (CPU only, no GPU)
- Fresh band-by-band comparison, same 25 k15 texts on three sides (scripts/muffle_diag.py, muffle_diag2.py, muffle_diag3.py; spectra n_fft 2048, dB relative to the 0-1 kHz band):
  - reference speaker (runs/k01/refs/ref_ms.wav, f0 med 189 Hz): 2-4k -23.8, 4-6k -22.9, 8-11k -20.3 -> BRIGHT
  - Kokoro Bella baseline (runs/k01/baseline, n=5): 2-4k -24.5, 4-6k -23.9, 8-11k -21.6 -> bright
  - Chatterbox raw (k15 raw2) and chosen cb_clips2 (the training targets): 2-4k -25.0/-26.6, 4-6k -24.1/-24.2, 8-11k -39.4/-40.2 -> ~2 dB darker in the presence band and ~19 dB darker above 8 kHz than the reference speaker
  - v0.4 synthesis of the same texts: within ~1 dB of its training target in EVERY band (2-4k -27.4, 4-6k -25.8, 8-11k -38.7), plus spectral flatness 2-6 kHz 0.022 vs target 0.011 (2x more noise-like).
- Conclusion: the training is NOT the (main) problem - the student faithfully imitates a dark teacher. The Chatterbox voice conversion step itself removes ~19 dB of 8-11 kHz and ~2 dB of 2-6 kHz present in the reference recording; every decoder loss is L1 to those dark clips, so the model is rewarded for muffling. Secondary effect: the source-filter head spends its 2-6 kHz energy as shaped noise (flatness 2x target; tonal_knob measured ~46-51% of voiced >1.5 kHz energy in the noise part), which reads as damp/smeared on top of dark. int8 export ruled out earlier (int8 = fp32).
- Fix options, cheapest first: (a) brighten targets 2-8 kHz with a matched high-shelf EQ before training (data-side, no arch change; do NOT boost 8k+ - Chatterbox has nothing there but noise floor); (b) run the skipped GAN/adversarial decoder stage (~1.6 s/step on T4, 5k steps ~2.2 h) - fixes noise-vs-harmonic and can restore brightness the L1 losses cannot ask for; (c) runtime noise-gain knob from tonal_knob.py (keep 0.25-0.5 pulls flatness 0.69 -> ~0.63, target 0.62) as an immediate palliative; (d) re-teacher with a different converter - most expensive, regenerates all data.

## 2026-11-17 Knob test page on real v0.4 weights (CPU, no GPU)
- Owner request: compare trained sound vs original Chatterbox ('watery / phaser, robot-like') and test the no-retrain tuning on a small sample. Built runs/listen/knob_v04.html + scripts/knob_v04.py (reuses tonal_knob's noise-to-harmonic move on the REAL v0.4 checkpoints runs/k16_final/dec_c4_last.pt + textA_c4_last.pt) + scripts/knob_v04_page.py.
- 10 sentences that have their original Chatterbox clip in cb_clips2.pt (all categories incl. codeswitch). Five versions per row, SAME noise seed per sentence: Chatterbox original / v0.4 as trained (keep=1) / knob 50% / knob 25% / knob 50% + 4 dB high-shelf 1.5-4 kHz. Blind-test shuffle + reveal built in.
- Measured over the 10 sentences (flatness 2-6 kHz -> target 0.623; >4 kHz energy -> target 0.084): v0.4 0.692/0.031; knob 50% 0.643/0.037; knob 25% 0.615/0.038; knob50+EQ 0.659/0.062. So the knob alone removes the watery/phaser smear toward target flatness, the shelf adds ~half the missing brightness; neither touches the >4 kHz harmonic detail (needs GAN stage or brighter targets).
- Michael: no training checkpoints in this repo (only his ONNX on HF, shared head) - knob can be baked into his next export once the other workspace's dec ckpt is available or re-exported with the knob inside. Noted on the page.

## 2026-11-17 Owner: knob test 'still not clear' -> head ablation isolates the phaser
- Owner listened to knob_v04: texture improved on paper but still not clear. Built scripts/ablate_head.py + runs/listen/ablate.html: 6 sentences x 7 versions (same noise seed, RMS-matched): Chatterbox target / v0.4 / harmonics-only (learned phase) / harmonics-only phase OFF / FIX A (phase OFF + keep 0.5) / FIX B (phase OFF + keep 0.25) / noise-only, with blind shuffle.
- Metrics (6 sentences): periodicity target 0.699, v04 0.745, harm .773, harm_pure .803, fix1 .783, fix2 .784, noise .148. Flatness 2-6k: target .621, v04 .698, harm .346, harm_pure .316, fix1 .624, fix2 .594. F0-curve smoothing changed nothing (rendered in first pass, dropped).
- Reading: v0.4 is MORE periodic than natural speech and F0 is fine, so the watery/phaser is NOT pitch jitter. Learned phase rotation u + pulse shift are the suspects (harm_pure > harm in periodicity). FIX A/B = disable phase machinery + energy-compensated noise move -> flatness lands on target (0.624/0.594) with higher periodicity than v0.4. Awaiting owner's ears on ablate.html; if FIX A/B wins, bake (phase OFF + chosen keep) into the next ONNX export, no retraining needed (pure weight-space/graph change).

## 2026-11-17 Ablation round 2 (owner listening verdict folded in)
- Owner verdict round 1: 'harmonics only (learned phase)' best but muffled; 'phase OFF' muffled + mild distortion. Conclusion: learned phase is GOOD (do not disable); the watery/phaser comes from the NOISE part smearing across harmonic bins; harmonics-only is muffled because the noise also carries the brightness fill.
- Round 2 (scripts/ablate_head.py rewritten): keep learned phase + keep noise energy but mask the noise spectrum out of the harmonic bins into the inter-harmonic gaps (voiced, >800 Hz, gaussian proximity frac/0.22, energy renormalised per frame, 3 new variants): fixC (all noise, gap-masked), fixC2 (gap-masked + keep 0.5), fixD (fixC2 + pulsed-noise sources removed).
- Metrics (6 sentences; target 0.699/0.621/0.080 periodicity/flatness/>4k): v04 0.745/0.698/0.040; harm 0.773/0.346/0.007; fixC 0.744/0.696/0.040; fixC2 0.742/0.650/0.047; fixD 0.746/0.629/0.035. Awaiting owner's ears (runs/listen/ablate.html, blind mode).
- If a FIX wins: bake (interharm mask + chosen keep, phase machinery kept) into the next ONNX export - no retraining; applies to Michael too.

## 2026-11-17 Ablation round 3: the burst path was the remaining watery source
- Owner verdict round 2: FIX C best but words still watery. scripts/measure_noise.py then showed the 3 ms BURST path carries ~100% of the voiced >1.5 kHz noise energy (plain ~1%, pulsed ~0 with cancellations); the burst fires noise 4x per trunk frame and never saw the gap mask -> that chopping is the residual watery/phaser.
- Round 3 variants (ablate_head.py: burst_uv option gates burst to unvoiced frames; comp_burst counts gated burst energy in the keep compensation): metrics (target 0.699/0.621/0.080): fixC 0.744/0.696/0.040; fixG (burst_uv+comp, keep .5) 0.746/0.511/0.043; fixH (burst_uv only, keep 1) 0.750/0.595/0.032 - flatness closest to target from below. fixE/F and fixH-with-keep-0 iterations dropped.
- Page runs/listen/ablate.html updated: target / v04 / harm / fixC / fixH / fixG. Awaiting owner's ears.

## 2026-11-17 FIX H baked into ONNX export (no retraining, CPU only)
- scripts/export_fixh.py ran clean: unvoiced no-op parity vs trained torch head max diff 1.79e-07; fp32 34.8 MB, int8 8.46 MB, both ~31x realtime on 1 thread; DFT bases generated at load (4), int8 weights 105 tensors, fp16 162.
- webgpu_patch applied -> runs/k16_final/fixh_export/paradee_int8_webgpu.onnx, 0 float64 ops left.
- scripts/verify_fixh.py: texture of the patched int8 ORT output on the 6 ablation sentences matches the PyTorch FIX H wavs (ort 0.754/0.598/0.029 vs torch 0.750/0.596/0.032; canonical targets 0.699/0.621/0.080). Sentence wavs saved as runs/listen/ablate/{id}_ortfixH.wav.
- suaradee package smoke test with SUARADEE_MODEL pointed at the new file: 3.9 s synth ok (runs/k16_final/fixh_export/smoke_yati.wav).
- Awaiting owner decision: overwrite kecil-tts-yati-v0.4 ONNX in place (browser cache risk, hard refresh needed) vs publish as v0.4.1 (Space dropdown edit needed by Space-owning agent). Michael rebake still blocked (no dec ckpt locally).

## 2026-11-17 v0.4.1 published (FIX H export)
- Owner verdict: ship as v0.4.1 (not in-place overwrite). hf/model_card_v0.4.1.md written (carries v0.4 CER tables, adds sound-texture table 0.750/0.598/0.032 vs targets 0.699/0.621/0.080; limitations updated: damp inherited from training, faint watery tail can remain).
- publish_hf.py: VERSIONS gained 0.4.1 (onnx = fixh paradee_int8_webgpu, samples-from runs/k16_final/fixh_export); wheel version regex keeps two-dot versions verbatim (suaradee-0.4.1 wheel).
- scripts/fixh_samples.py regenerated ALL 260 v5 sample clips with the new export into runs/k16_final/fixh_export/audio (+ v5_scores.csv copied); build_hf picked the same 6 sentences as v0.4.
- Published https://huggingface.co/Suarify/kecil-tts-yati-v0.4.1 (36 files, --skip-space as always). Remote re-download: 0 float64 ops, 8.46 MB. from_pretrained('Suarify/kecil-tts-yati-v0.4.1') end-to-end synth ok (smoke_v041_remote.wav).
- NEXT: Space version dropdown needs the v0.4.1 entry (Space-owning agent) before the owner can pick it in the demo; Michael rebake still blocked on his dec ckpt.

## 2026-11-17 Space updated for v0.4.1 (owner authorised the Space publish)
- runs/local_v041 (space page + g2p front end) got: v0.4.1 as Yati default dropdown entry ('v0.4.1 (latest)'), note updated (cleaner, less watery sound), Yati card link + footer link -> v0.4.1, Space README refreshed (models: yati-v0.4.1/0.4, michael-v0.2; old suaradee-* names gone).
- Uploaded to the Space. DISCOVERY: the Space-owning agent renamed the Space Suarify/suaradee-web -> Suarify/kecil-tts-web (rebrand); old subdomain 404s. Live URL now https://suarify-kecil-tts-web.static.hf.space (HF page redirects). My upload went through the redirect fine (suarify-logo.png from the other agent preserved).
- Browser-verified live: dropdown defaults v0.4.1, model loads WebGPU 8.2 s, synth 4.37 s audio in 0.98 s (4.5x realtime), only benign ORT warnings. runs/local_v041 now doubles as the deploy source.

## 2026-11-17 Michael v0.2.1: FIX H rebake without checkpoints (weights rebuilt from ONNX)
- No Michael checkpoints existed anywhere (only compiled ONNX). Rebuilt the torch model from runs/michael_v02.onnx: dequantized int8/fp16 tensors by their preserved torch names (234), mapped the 27 anonymous onnx::MatMul weights by consuming node path (ALBERT layers, head pw1/pw2/out/out_b, duration_proj, asr_proj, bert_encoder), recovered M/Mb warp matrices from the /head/MatMul Constants. Key gotcha: ONNX LSTM gate order is iofc vs torch ifgo -> gate reordering needed (durations were 5x too long before this).
- scripts/michael_rebuild.py: zero-noise parity (zeros noise constant baked into a copy of the graph after probing its shape) -> lengths identical, rms diff 0.00083 vs signal 0.035 (2.3%, float32-vs-float64 phase drift only). Saved runs/michael_v02_rebuilt.pt.
- export_fixh.py gained --state mode (load stripped state dict + M/Mb; pooler not in graph). Michael FIX H export: unvoiced no-op parity 3.28e-07, fp32 34.8 MB, int8 8.46 MB, 26x realtime; webgpu_patch -> 0 float64.
- Texture old->new (6 sentences): periodicity 0.635->0.651, flatness 2-6k 0.528->0.436, >4k share 0.105->0.082 (target 0.080). Wavs in runs/listen/michael/ (old vs fixh).
- Published Suarify/kecil-tts-michael-v0.2.1 (37 files; card carries v0.2 CER, adds texture table; all 6 samples re-rendered by the new export). Remote re-download: 0 float64. Space updated: Michael dropdown v0.2.1 (latest). Live verified: WebGPU load 6.5 s, 3.25 s audio in 0.85 s.

## 2026-11-17 Michael still watery/phasey: ablation round 1 for him
- Owner: v0.2.1 still watery + phasey. Built a Michael-specific ablation (scripts/ablate_michael.py) on his rebuilt weights, same six sentences, same noise seed, loudness-matched: old / fixH (current) / noPhase (u=1, shift kept) / noShift (shift=0, u kept) / keepHalf (0.5 noise->harmonic, energy-compensated) / noBurst / noMod.
- Metrics (periodicity / flatness / >4k): old .644/.516/.096; fixH .650/.436/.086; noPhase .583/.433/.093; noShift .590/.433/.091; keepHalf .648/.380/.085; noBurst .654/.430/.073; noMod .657/.393/.081. noPhase/noShift drop periodicity ~0.06 -> the learned phase and pulse shift both carry the pitch; don't expect them to be the fix.
- Listening page runs/listen/michael.html (served at localhost:8931/michael.html, also copied into runs/local_v041). 42 wavs in runs/listen/michael_ablate. Awaiting owner pick (two best buttons allowed).

## 2026-11-17 Michael watery root cause found: K=64 vs K=128 harmonics
- All round-1/2 Michael variants sounded watery to the owner. Diagnostics on the harmonic path (gh flicker, phase step/frame, F0 jitter, AM sidebands) showed no difference vs Yati. BUT inspecting the ONNX constants showed Michael's graph carries head.k of size **128** (card says 128 harmonics: v0.2 notes '128 sine harmonics instead of 64'), while the rebuild/export used VDecoder default K=64 -> harmonic bank truncated at ~7.7 kHz instead of 11 kHz, high band left to noise -> muffled + watery on EVERY variant.
- Fixed: michael_rebuild.py and export_fixh.py (new --k arg) now build with K=128. Zero-noise parity vs the original ONNX improved 0.0192 -> 0.0046 max, rms 0.00083 -> 0.00037 (4x), confirming K=128.
- Re-exported FIX H at K=128 (parity 2.4e-07, webgpu-safe 0 float64), republished Suarify/kecil-tts-michael-v0.2.1 in place (remote head.k = 128 verified; samples re-rendered). Note: same filename -> browsers may cache the old file; hard refresh needed.
- Michael ablation regenerated at K=128 with a decisive harmonics-only column: metrics old .643/.516/.096, fixH .649/.437/.088, harmOnly .700/.289/.013, noPhase .582/.433/.093, noShift .590/.433/.092, keepHalf .647/.381/.087, noBurst .654/.430/.074, noMod .656/.393/.082. Page runs/listen/michael.html (localhost:8931/michael.html). Owner to retest: if harmonics-only is ALSO watery the artefact is in Michael's training targets, not the export.

## 2026-11-17 Michael round 3: choppy characters from the burst gate; harmonics-only has no consonants
- Owner: published v0.2.1 'too choppy, some character chopped'; harmonics-only 'clean abit'. Measured consonant-band energy (noise-like frames, 2-8 kHz) vs published: fixH0 1.24x, hang7 1.13x, burstSoft 1.27x, keep25 1.13x, harmFill 1.20x, harmOnly 0.01x -> the 21-frame hangover eats consonants; harmonics-only removes fricatives entirely.
- scripts/ablate_michael3.py + runs/listen/michael3.html (7 variants: fixH21/fixH0/hang7/burstSoft/keep25/harmFill/harmOnly; served localhost:8931/michael3.html, synced to runs/local_v041). Awaiting owner pick.
- export_fixh.py parameterised: --gap-keep (fraction of gap noise kept, rest into harmonics with energy compensation), --hang (burst hangover frames), --burst gated|soft|off, --soft (voiced burst gain). Regression re-export with the published settings still matches (no-op parity 2.4e-07, lengths equal).

## 2026-11-17 Michael v0.2.1 final: 7-frame burst hangover baked (owner's pick)
- Owner verdict on round 3: 'gated, 7-frame hangover is better, the watery and phasey part less'. Baked exactly: export_fixh.py --state runs/michael_v02_rebuilt.pt --k 128 --gap-keep 1 --hang 7 --burst gated. No-op parity 2.68e-07, int8 8.46 MB, 18.3x realtime, webgpu patch 0 float64.
- Consonant energy vs the original v0.2 converter: 0.79x mean (published hangover-21 file was ~0.65x) -> characters no longer chopped.
- Republished Suarify/kecil-tts-michael-v0.2.1 in place (remote sha == local sha, head.k 128, 0 float64), samples re-rendered with the hang7 export. Same filename -> browsers/Space need a hard refresh or cleared site data.
- Yati v0.4.1 still uses the 21-frame hangover (owner approved it as-is); hang7 available for her too if wanted.

## 2026-11-17 Round 4: de-chop Michael, brighten Yati (owner: harmonics good, still choppy; Yati has room)
- Michael stays choppy even at hang7 -> added progressively less aggressive burst handling (soft 30%, free 100%) and 3-frame temporal gain smoothing; consonant energy rises 18.3 -> 20.5 (soft) -> 23.9 (free) on the same 4 sentences.
- Yati brightness measurement: energy share above 7 kHz only 0.004 (vs Michael 0.028) - the dark-target deficit left after FIX H. Two no-retrain levers rendered: air shelf (zero-phase high shelf 6.5 kHz +4/+6 dB) and a 2nd-harmonic exciter (5-10.5 kHz, 12%). >7k share: base 0.004 -> air4 0.011 -> air6x 0.015 (4x); consonant energy 4.8 -> 6.8 -> 8.4. Also rendered yati hang7, burstSoft, smooth, keep60 for comparison.
- Page runs/listen/air.html (localhost:8931/air.html, 24 rows, 56 wavs, all 200). Awaiting one pick per voice; the air shelf bakes into ONNX as a small linear-phase FIR conv.

## 2026-11-17 Michael focus: the 'III' vowel case and distortion-free smoothing
- Owner: the sentence with 'III' is muffled/robotic/watery there; 'smooth both gains' helped but adds mild distortion. g2p reads III as three stressed 'I' vowels -> long stressed vowels, where the noise floor shows.
- Round: scripts/ablate_iii.py renders the III sentence + a vowel-heavy and a mixed sentence with 6 variants: base (hang7), smooth3 (both gains, previous), smoothGn (noise gains only - harmonics untouched, no transient smear), smooth121 (light [1,2,1]), vowel (gap_keep 0.6 + noise smoothing), softGn (soft bursts + noise smoothing).
- Metrics (periodicity/flatness/>4k): base .673/.440/.044; smooth3 .688/.452/.043; smoothGn .671/.440/.045; smooth121 .689/.453/.042; vowel .669/.388/.047; softGn .671/.470/.050. Page runs/listen/iii.html (localhost:8931/iii.html, 18 wavs, all 200). Yati untouched per owner.

## 2026-11-17 Michael: 'atau' robotic+damp probe (III sentence better with noise-smooth)
- Owner: III sentence better now; 'atau' still robotic and damp. Rendered the III sentence + a second 'atau' sentence + a vowel-heavy one with 7 variants (scripts/ablate_atau.py): smoothGn (current best), air3/air5 (high shelf above 6.5 kHz = the damp), breath (55% gap noise moved into harmonics), air3breath, vib (0.4% pitch sway at 4.7 Hz = the robotic), air3vib.
- Metrics: smoothGn .631/.442/.067; air3 .630/.445/.079; air5 .630/.448/.084; breath .643/.390/.067; air3breath .643/.392/.076; vib .633/.442/.066; air3vib .632/.445/.079.
- Page runs/listen/atau.html (localhost:8931/atau.html, 21 wavs, all 200). Awaiting levers verdict (air vs breath vs vibrato); Yati untouched.

## 2026-11-17 Michael: timing is the real problem on III/atau (spectral levers all failed)
- Owner: air/breath/vibrato did not fix III or atau; more vibrato = more watery/damp/phase. Measured the model's own durations (scripts/michael_durations.py): on the III sentence 'atau' gets [16,6,2,3] frames (one vowel held 400 ms); in a clean sentence the same word gets [1,4,2,3] (one phoneme 25 ms); 15-17% of all phonemes get 1 frame. A held 400 ms vowel with frozen timbre reads as robotic/damp/watery; fast vibrato lands between the 25 ms phase frames and smears.
- scripts/ablate_dur.py renders 3 sentences with: base, clamp (2-8 frames = 50-200 ms), med3 (median filter + clamp 2-10), clampair (+3 dB shelf), trem (clamp + slow random amplitude sway 18% on the noise floor), vib03 (clamp + 0.3% vibrato at 3 Hz).
- Page runs/listen/dur.html (localhost:8931/dur.html, 18 wavs, all 200). If clamp works, bake as a Clip on the duration tensor in the ONNX (no weights, no spectrum change).

## 2026-11-17 Michael: grid around clamp+air (owner: clamp+3dB air better)
- Winner so far: durations clamped [2,8] frames + 3 dB high shelf above 6.5 kHz. Grid (scripts/ablate_grid.py, 3 sentences, 10 variants): air 2/3/4/5 dB, shelf corner 6.0/6.5/7.0 kHz, clamp [2,7]/[2,8]/[2,9]/[3,8], plus a slow amplitude sway.
- Metrics (periodicity/flat2-6k/>4k/>7k): ref .608/.468/.091/.041; air5 .607/.471/.103/.051; fc6k .608/.471/.096/.040; clamp38 .635/.466/.088/.041 (best periodicity - 75 ms floor removes staccato); others within noise.
- Page runs/listen/grid.html (localhost:8931/grid.html, 30 wavs, all 200). Awaiting: best air amount + best clamp range to bake.

## 2026-11-17 Michael v0.2.2 published (clamp 3-8 + air 3 dB + noise smoothing + hang7 + K=128)
- Owner picked clamp 3-8 (75-200 ms per phoneme) and asked for a minor version bump. Export recipe now: K=128, gap-masked noise (gap_keep 1), burst gated to unvoiced with 7-frame (22 ms) hangover, 3-frame noise-gain smoothing, phoneme durations clamped to [3,8] frames, +3 dB linear-phase high shelf above 6.5 kHz (127-tap FIR inside the graph).
- export_fixh.py gained --clamp-lo/--clamp-hi/--air-db/--air-fc/--gn-smooth; the head parity check now skips the no-op comparison when the recipe is modified and prints the expected length instead.
- Verified: torch-vs-ORT lengths identical (94200), 0 float64 after webgpu patch, 8.47 MB, 12-20x realtime. Metrics on the 3 test sentences: flat 0.472, >4k 0.094, >7k 0.043 vs grid reference 0.466/0.088/0.041.
- Published Suarify/kecil-tts-michael-v0.2.2 (37 files; card documents all five changes and the timing clamp caveat; samples rendered by the new export). Remote sha == local export, 0 float64, head.k = 128. Space dropdown: michael v0.2.2 (latest); live check WebGPU 10.4 s load, 4.5x realtime.

## 2026-11-17 Michael vs Yati on the latest published models; retrain feasibility
- Head-to-head (same 6 sentences, latest exports, scripts/compare_latest.py): yati v0.4.1 periodicity 0.751, flat 0.597, >4k 0.025, >7k 0.003, consonant energy 4.5, 13.3 chars/s; michael v0.2.2 periodicity 0.664, flat 0.444, >4k 0.099, >7k 0.045, consonant energy 29.3, 10.2 chars/s (clamped). Yati = cleaner/more periodic but darker; Michael = brighter and far more articulate, less periodic.
- Training asymmetry: Yati v0.4 1,281 clips / ~118 min (teacher shard on disk = 1,281 clips, 88.8 min after trimming) + two data-cleanup rounds; Michael ~750 clips / ~49 min, voice-converted twice.
- Retrain feasibility confirmed: data/teacher/shard_0000.pt holds the full teacher set (text, phonemes, durations, F0, N, 24 kHz audio); kaggle/k15_cbgen2 has the Chatterbox + ChatterboxVC pipeline; kaggle/k16_train_c4 trains decoder 12k + text 10k and exports (~1 h GPU). Two routes: (A) VC the 1,281 clips to a male reference (am_michael) then retrain -> true Michael voice, needs an am_michael reference clip + ~2 h GPU; (B) pitch/formant-shift the existing clips to male range locally (no GPU) then retrain -> new clean male voice, ~1 h GPU. GPU approval pending; budget resets 2026-11-21.

## 2026-11-17 Kaggle retrain (sm-07-delta) done; true checkpoints change the picture
- Kaggle account switched to suarifymalaysia (owner's token). Found the original Michael pipeline there: datasets sm-michael-clips (cb_michael.pt), sm-c4-in (Yati v0.4 inputs + yati dec), kernels sm-01..sm-06 all COMPLETE. sm-06 trained Michael on **1,261 rows** (746 old + 767 C4), decoder 12k steps, text 7.5k; report: held CER 2.2%/8.2%, test 6.9%/10.6%. So the published v0.2 weights are the C4 model (card's '750 clips/49 min' is stale; ONNX-vs-ckpt head corr 0.944 = int8 quant only).
- Uploaded dataset suarifymalaysia/sm-delta-train-in (merged shard + dec/text ckpts); kernel sm-07-delta trained 4,000 more decoder steps (held_spec_loss 2.545 -> 2.497) and 4,500 text steps (early stop; best at 500). Outputs: dec_delta_last.pt (step 4000), textA_delta_last.pt (step 4500).
- Key correction: the pathological durations (16-frame/400 ms holds) came from our ONNX-int8 rebuild, NOT the real model. True text model: max 7-8 frames, no >=10-frame holds, 1-frame share 17-19%.
- Exported v0.3.0 from the true checkpoints (K=128 + full recipe), webgpu patched (0 float64), 8.47 MB, 19x realtime. Metrics on 6 sentences: periodicity off .643 / min2 .656 / min3 .662 vs v0.2.2 .634.
- Listening page runs/listen/v03.html (localhost:8931/v03.html) compares v0.2.2 vs v0.3.0 with clamp off / 2-8 / 3-8. Awaiting owner pick before publishing v0.3.0.

## 2026-11-17 Training gap Michael vs Yati (measured from the shards)
- Data volume is now essentially equal: Yati v0.4 teacher 1,281 rows / 88.8 min / median F0 196 Hz; Michael C4 merged 1,261 rows / 87.3 min / median F0 120 Hz.
- Real gaps: (1) Michael's targets are voice-converted TWICE (Chatterbox -> female -> male) vs Yati's once, so his training audio carries two rounds of VC artefacts (the residual damp/phaser); (2) his first 746 clips come from the older single-seed generation round, only the 767 C4 additions used the improved recipe; (3) text side trained ~8k effective steps vs Yati's 10k (decoder 16k vs 12k, and his decoder started from Yati's v0.4 decoder).
- Scores: test CER yati 5.5/9.5 vs michael 6.9/10.6 (200 sentences); held median michael 2.2 vs yati 4.8 (capped 8.2 vs 8.5). Sound: periodicity yati 0.751 vs michael v0.3.0 0.662 -> pitch-clarity gap is timbre/VC driven, not data volume.
- Proposed next experiment (needs GPU approval): regenerate all 1,281 texts with Chatterbox using the am_michael reference directly (ONE conversion instead of two), rebuild rows, retrain text 10k + decoder 12k -> targets with half the VC artefacts. ~3 h GPU total.

## 2026-11-17 Michael v0.3.0 published + Space benchmarks; 15-clip single-conversion POC queued
- Published Suarify/kecil-tts-michael-v0.3.0 (37 files; remote sha == local export, 0 float64, head.k 128, 8.47 MB). Card now carries all benchmarks: CER (v0.2 8.7/14.3 -> C4/v0.3.0 6.9/10.6 test, held 2.2/8.2; Yati 5.5/9.5), texture/periodicity vs Yati and Chatterbox targets, size/speed/cost/offline table, and an explicit note that the exact v0.3.0 weights are still being re-scored and that no MOS/human comparison exists for this date.
- Space: intro .lead 14px -> 13px; new Benchmarks card (10 rows: file size, params, CPU/WebGPU speed, RM 0 cost, offline, CER test/held, >4k brightness, periodicity) with EN/BM i18n; Michael dropdown v0.3.0 (latest, retrained) default; footer/card links updated. Live check: WebGPU 1.11 s first audio, 4.10 s audio in 1.11 s (3.7x).
- Next experiment per owner (15 clips instead of 3 h full run): generated am_michael reference locally with Kokoro (scripts/make_michael_ref.py, 25.4 s), built dataset suarifymalaysia/sm-vc15-in (ref + 15 texts incl. the III/atau sentence), pushed kernel suarifymalaysia/sm-08-vc15: generates 15 clips with Chatterbox + am_michael prompt directly (ONE conversion) and exports the existing double-converted clips for A/B.

## 2026-11-17 15-clip single-conversion POC done (sm-08-vc15 v5)
- Kaggle image had a broken torchvision (mismatch with torch): transformers failed importing LlamaModel. Fix: uninstall torchvision + pin transformers 4.46.3; chatterbox import verified before generation. Kernel then completed: 15 single-conversion clips (Chatterbox with the Kokoro am_michael prompt directly) + the 15 old double-converted clips, zip 4.3 MB.
- scripts/vc15_page.py pulled the output and built runs/listen/vc15.html (localhost:8931/vc15.html, 30 buttons, all 200). A/B on the same 15 texts incl. the III/atau sentence.
- Raw-clip metrics (periodicity / flat / >4k / >7k): single 0.557/0.488/0.076/0.013 vs old 0.584/0.493/0.085/0.027 -> the single-conversion clips are darker at the top and less noisy; owner to judge by ear before the full 1,281-clip regeneration.

## 2026-11-17 Owner verdict on POC 1: old (2 conversions) more accurate -> regenerate-with-male-prompt route dropped
- POC 1 conflated two variables: generation prompt (am_michael English voice speaking Malay) and conversion count. Its 'new' clips were duller at the top (>7k 0.013 vs 0.027) and the owner judged the old double-converted clips more accurate.
- POC 2 (queued, sm-09-vc15b): same 15 raw Chatterbox clips (Malay prompt) converted ONCE to am_michael vs the current twice (raw -> female -> male). This isolates the conversion count. Includes the III sentence (c0571). Dataset suarifymalaysia/sm-vc15b-flat (raw_*, old_* wavs + ref + texts), kernel pushed v1.
- Metric snapshot for POC 1 (raw clips): single 0.557/0.488/0.076/0.013 vs old 0.584/0.493/0.085/0.027 (periodicity/flat/>4k/>7k).

## 2026-11-17 POC 2 done: raw vs one VC vs two VCs on identical clips
- sm-09-vc15b completed in 6 min (15 clips, zip 7.5 MB). Dataset had to be flattened (raw_*/old_* at root) because the Kaggle uploader zips or skips subfolders.
- Metrics (periodicity / flat 2-6k / >4k / >7k): raw 0.679/0.639/0.073/0.013; single (1 VC) 0.567/0.492/0.088/0.025; old (2 VCs) 0.573/0.481/0.091/0.028. The single conversion sits very close to the double conversion - no large texture difference, unlike POC 1 where the generation prompt changed pronunciation.
- Page runs/listen/vc15b.html (localhost:8931/vc15b.html, 45 buttons, all 200). Awaiting owner verdict: if new ~= old, keep v0.3.0 and stop; if new is cleaner, one VC for all 1,281 raw clips (~1 h GPU) + retrain.

## 2026-11-17 Conversion-quality gate before any full retrain (10 clips, 2 references)
- Student-vs-teacher measurement (scripts/student_fidelity.py, same 15 texts): teacher-old vs teacher-new spectral distance 0.97, student v0.3.0 vs teacher 3.97 (~4x) -> the student imitation gap dominates; conversion choice is secondary. Texture: teacher new 0.567/0.492/0.088/0.025, teacher old 0.573/0.481/0.091/0.028, student 0.645/0.434/0.086/0.038.
- Owner wants proof the conversion itself is not watery/phasey before spending quota. Kernel sm-10-vc10 (dataset sm-vc10-flat): 10 raw clips converted with TWO am_michael references - ours (Kokoro-generated, 25 s) and the original pipeline's ref_en_michael.wav; page will be runs/listen/vc10.html (raw / our-ref / pipeline-ref). Kernel pushed v1, currently QUEUED.

## 2026-11-17 Conversion gate done: both references behave the same
- sm-10-vc10 completed (6 min GPU, 10 clips, zip 5.0 MB). Page runs/listen/vc10.html (localhost:8931/vc10.html, 30 buttons, all 200).
- Metrics (periodicity / flat 2-6k / >4k / >7k): raw 0.667/0.640/0.081/0.015; our ref 0.570/0.490/0.100/0.030; pipeline ref 0.585/0.488/0.100/0.031. Conversion adds HF energy (>4k 0.081 -> 0.100, >7k 0.015 -> 0.031) and costs ~0.08 periodicity; the two references are within noise of each other.
- Decision pending owner's ear: if neither converted column is watery -> use pipeline ref (slightly higher periodicity) for the full 1,281 conversion + retrain; if both watery -> conversion is the ceiling, stop at v0.3.0.

## 2026-11-17 Student-vs-teacher preview page (answers 'will the final model sound like the clips')
- Rendered the student v0.3.0 on the same 10 texts as the conversion gate and built runs/listen/vc10s.html (localhost:8931/vc10s.html): raw / converted (pipeline ref) / student, 30 buttons all 200.
- Metrics (periodicity / flat / >4k / >7k): raw 0.667/0.640/0.081/0.015; converted 0.585/0.488/0.100/0.031; student 0.643/0.441/0.096/0.041. Student is more tonal, less flat, slightly brighter than the teacher, and 4x further from it than the two teacher variants are from each other -> a retrain on 1-VC data moves the student only slightly; it cannot equal the clips.
- Cheaper decisive test proposed instead of the 2 h full run: continue the existing v0.3.0 checkpoint ~2,000 steps (~10-15 min GPU) and listen whether quality moves at all; if it does not, data changes will not either.

## 2026-11-17 Student 'very bad, slower, phasey/watery': isolating our export from the model
- Phase-advance experiment (scripts/ablate_phase.py): adding inter-frame phase advance DESTROYS periodicity (0.673 -> 0.268 with +sign, 0.450 with -sign), so the current no-advance formulation is correct; phase advance is not the fix.
- Faithful-vs-processed test (scripts/ablate_faithful.py) from the true checkpoints: faithful (no gap mask/gate/smoothing/air/clamp) 4.58 s on the III sentence vs 5.85 s with our recipe -> the clamp is what made the voice slow (+28% length).
- Metrics (periodicity/flat/>4k): faithful 0.640/0.507/0.101; faithful+air 0.638/0.512/0.121; our fixH 0.682/0.440/0.112; fixH no-clamp 0.641/0.440/0.102; fixH bursts-ungated 0.678/0.509/0.122.
- Page runs/listen/faithful.html (localhost:8931/faithful.html) compares TEACHER clip vs faithful / faithful+air / our recipe (clamp) / our recipe no-clamp / our recipe ungated bursts, so the owner can say whether the model or our processing is at fault.

## 2026-11-17 Student-vs-teacher gap is architecture-level; clarity post-processing page
- Measured student-vs-teacher (log-mel distance) for both voices: Yati 3.29, Michael 3.84 (Michael teacher-vs-teacher 0.84). Both 8M students sit ~4x further from their teacher than teacher variants differ; Michael is only ~17% worse than Yati -> the murk is the 8M student rendering, not a broken Michael pipeline.
- Owner on faithful vs teacher: faithful is better (speed fixed) but still murky/watery/phaser vs the teacher's clarity.
- scripts/ablate_clarity.py built clarity options on the faithful render (no clamp/gate): air 3/5 dB, noise floor x0.6, combos, and +3 dB harmonics above 1.5 kHz. Metrics (periodicity/flat/>4k/>7k): faithful .640/.507/.101/.031; air3 .638/.512/.121/.046; air5 .634/.517/.136/.060; noise60 .650/.485/.088/.028; air3noise60 .648/.493/.110/.045; air5noise60 .646/.497/.129/.062; harmBoost .620/.479/.105/.032.
- Page runs/listen/clarity.html (localhost:8931/clarity.html, 16 buttons incl. TEACHER) - owner to pick the clearest; if none approaches the teacher, only longer/bigger training can move it.

## 2026-11-17 Bigger student for Michael (owner approved): head 320, 6 blocks (~11.9 MB)
- Step 0 (size_probe.py): params / estimated int8 size - current 7.88M/8.3MB; wide-224 8.62M/9.1MB; wide-320 10.10M/10.6MB; wide-320-6b 11.34M/11.9MB. Owner accepts the size change (dropdown variant, most models stay 8M).
- Step 1 pushed: kernel suarifymalaysia/sm-11-big - builds VDecoder(head v2, dim 320, blocks 6, K 128), copies the trunk from dec_C4_last.pt (head random-init), patches the trainer's VDecoder line, trains 16,000 steps (max 50 min) and saves dec_big_last.pt. Text side reused unchanged (validated durations/speed).
- Plan after training: local export + listening page (old v0.3.0 vs big), then CER scoring (~15 min) and publish as the bigger Michael variant.

## 2026-11-17 sm-11-big v1 failed on a patch that hit the wrong copy; v2 uses a regex patch
- v1: the trainer still built a dim=160 model ('size mismatch ... current model is torch.Size([160])'), so the init (320) could not load. Fix: regex-replace every VDecoder(...) call in the COPIED trainer (W/scripts/ms_train.py), print the patched line, and assert dim=320 before launching. Kernel v2 RUNNING.

## 2026-11-17 Bigger student trained (320/6 blocks) - no measurable gain
- sm-11-big v2 completed: 16,000 steps, 28 min, held_spec_loss 2.515 (the small model reached 2.497). Trunk transfer + random head.
- Export: paradee.onnx 48.6 MB, int8 11.96 MB, speed 9.1x fp32 / 5.3x int8 on one thread (vs 19x for the 8 MB model).
- Student-vs-teacher log-mel distance on the same 2 sentences: small 3.39, big 3.39 -> no gain, and 3-4x slower. Page runs/listen/big.html (localhost:8931/big.html) with teacher/small/big/big+air for an ear check; recommendation is to keep v0.3.0 unless the ear disagrees.

## 2026-11-17 Michael v0.3.1 published (speed fix) + third-voice candidate samples
- Root cause of 'Michael speaks slower': the v0.3.0 export clamped every phoneme to 3-8 frames (75-200 ms). New export: --no-gap-mask --burst free --clamp-lo 1 (faithful synthesis, native durations). Verified on the III sentence: 4.55 s vs 5.85 s (-22%); samples re-rendered (3.8/3.4/2.8/2.9/3.8/4.3 s).
- exporter gained --no-gap-mask and --burst free; HEAD passes gap_mask.
- Published Suarify/kecil-tts-michael-v0.3.1 (37 files, sha match, 0 float64, 8.46 MB) with a card explaining the clamp removal + faithful synthesis; Space dropdown michael v0.3.1 (latest), card/link/benchmark header updated. Live check: WebGPU load 16.4 s, III sentence 3.35 s of audio.
- Why Yati is clearer (measured): teacher periodicity yati 0.699 vs michael 0.585 (her target had ONE voice conversion, his two); pitch 196 vs 120 Hz (harmonics 196 vs 120 Hz apart vs the 80-point gain grid -> less smearing); teacher >7k 0.021 vs 0.031 (less inter-harmonic noise). Student follows the ordering: 0.751 vs 0.645.
- Next voice: generated Kokoro samples for af_heart/af_bella/af_nicole/af_sarah/bf_emma/am_puck/am_fenrir/bm_george -> runs/listen/voices.html (localhost:8931/voices.html). Proposal: pick one, 15-clip Malay preview (~6 min GPU), then full 1-VC build (~1.5 h).

## 2026-11-17 New voices (nicole, fenrir): staged plan approved, phase A+B queued
- Raw coverage check: local cb_raw_all has only 654 of 1,281 ids; sm-15-raw returns 403 for this account -> raws must be regenerated (voice-independent, reused by both new voices).
- Datasets: suarifymalaysia/sm12-in (texts.tsv for all 1,281 from the teacher shard + ref_ms.wav from runs/k01/refs + ref_af_nicole.wav) and suarifymalaysia/sm13-in (textA_c4_last.pt as the female text init).
- Phase A+B kernel sm-12-nicole-data (QUEUED): generates all 1,281 raw clips with ONE fixed Chatterbox config, then converts them ONCE to af_nicole. Fault tolerance: checkpoint every 100 clips (raws_partial.pt / nicole_partial.pt), 150 s per-clip timeout, failures logged to sm12_report.json, resume picks up partial files.
- Phase C kernel sm-13-nicole-train written and ready: rows via ms_rows --letters (cuda), decoder 12k from yati_c4_dec_last.pt, text 10k from textA_c4_last.pt, saves dec/text nitcole ckpts + rows. Waits on phase A+B output (kernel_sources).
- Fenrir reuses the generated raws afterwards (VC + rows + train only).

## 2026-11-17 sm-12 nicole data: two environment failures fixed
- v2: dataset sm12-in was created seconds before the push and did not mount (/kaggle/input empty -> texts.tsv IndexError). v3 re-push mounted it, then failed: chatterbox's pip install upgraded torchaudio to a CUDA-mismatched build (libcudart.so.12 missing, libtorchaudio cannot load). v4 adds a self-repair step: detect torch version/CUDA tag, force-reinstall torchaudio from the matching wheel index, verify import. Same fix applied to the fenrir kernel.
- Extra datasets for fenrir failed to create (403), so ref_fenrir.wav went into sm12-in (new version) and the male init checkpoints into sm13-in (new version).

## 2026-11-17 sm-12 v5: stub perth instead of fixing torchaudio
- The image's torch/torchaudio pair is broken (libtorchaudio.so will not load; a torchaudio==X.* reinstall did not match Kaggle's patched torch). chatterbox only needs 'perth' for watermarking, and perth imports torchaudio.
- v5 installs a no-op 'perth' package stub on PYTHONPATH (PerthNet / PerthImplicitWatermarker -> pass-through _Any), so torchaudio is never imported; imports are verified before generation. Fallback inside the same kernel: if the stub path fails, force-reinstall a matching torch+torchaudio pair from the PyTorch wheel index. Same stub added to the fenrir kernel. v5 QUEUED.

## 2026-11-17 Critical step-by-step gate plan for the new voice
- Gate 1 (kernel sm-12): raw generation + ONE VC to nicole, checkpoints every 100, per-clip timeout, resume.
- Gate 2 (after sm-12): listen to a few converted clips for watery/phase (page built by scripts/gate_check-style flow) BEFORE any training.
- Gate 3 (kernel sm-13a PROBE, written): rows + only 3,000 decoder steps + 2,500 text steps (~15 min) so the student's voice can be heard before the full run.
- Gate 4: scripts/gate_check.py measures durations (1-frame share, max frames), periodicity/flatness/>4k/>7k and renders faithful + air3 samples; benchmarks to beat: distance to teacher yati 3.29, michael 3.84; no clamp so native speed.
- Gate 5: only if the probe sounds clean -> full 12k run (sm-13), export faithful, listen again, then publish.

## 2026-11-17 Where we are (status)
- Live: Michael v0.3.1 (speed fix + faithful synthesis, 8.46 MB) on HF and the Space; Yati v0.4.1 untouched; Yati v0.4.1 remains the cleaner voice.
- Dropped after measurement: bigger Michael student (320/6, 11.96 MB, 3x slower, zero gain: distance 3.39 vs 3.39, held loss 2.515 vs 2.497); two-conversion regenerations; post-processing air/noise variants (owner preferred faithful).
- Running: sm-12-v8 (nicole data) - matched official torch+torchaudio pair first (2.6.0/cu124 -> cu121 -> 2.5.1), then extras removed (torchvision/torchao/torchtext), perth stubbed, transformers 4.46.3 with ladder, then generate 1,281 raws + one VC to af_nicole with checkpoints every 100.
- Environment battles so far: dataset mount timing (fixed by re-push), torchaudio libcudart/libtorchaudio (fixed by the matched pair), torchao torch._dynamo symbol (fixed by removal). Each failed attempt cost only ~4 min of setup, no generation.
- Ready to fire: sm-13a probe (3k steps, ~15 min) for an early listen; sm-13 full train (12k); sm-14 fenrir VC (reuses raws).
- Gate thresholds before publishing nicole: rows >= 900, 1-frame share <= 25%, max phoneme <= 10 frames, periodicity >= 0.60, distance to teacher <= 3.6 (michael 3.84, yati 3.29), native speed (no clamp).

## 2026-11-17 sm-12 v9/v10: environment root cause narrowed to missing libcudart.so.12
- v9 copied the BYOD docker_image from the working sm-05 kernel, but Kaggle ignored it (log still python 3.13.16) and failed with the same torchaudio/libcudart error.
- v10 installs nvidia-cuda-runtime-cu12 and preloads libcudart.so.12 with ctypes.CDLL(RTLD_GLOBAL) IN THE MAIN PROCESS before any chatterbox import (a child-process preload would not help dlopen). Running.
- Local page server on port 8931 was killed by a harness restart and restarted in the background.

## 2026-11-18 Gate 1 PASSED: 1,281 raws + 1,281 nicole conversions, 0 failures
- The long-running sm19/sm20 kernels were NOT hung - they were generating (272.7 min) and converting (34.4 min) but Kaggle's status/quota API did not reflect it. Both COMPLETE; sm19 output pulled.
- Gate 1: DONE raws 1281 nicole 1281 fails 0. nicole.pt (307 MB) downloaded, 15 sample wavs extracted to runs/listen/nicole_teacher.
- Gate 2 page runs/listen/nicole_teacher.html (localhost:8931/nicole_teacher.html): owner listens for watery/phase in the converted teacher clips before training.
- Note: generation was 272 min (4.5h) not the estimated 55 min - each clip ~13s at 7-9 it/s on 1000 diffusion steps; quota should be checked before committing fenrir (which reuses raws, VC only 34 min).

## 2026-11-18 Gate 2 PASSED (owner: clean enough) -> probe training launched
- Owner listened to 15 nicole converted teacher clips and approved.
- Kernel sm-13a-nicole-probe pushed (kernel_source sm-19 output for nicole.pt): builds rows from 1,281 clips, trains decoder 3,000 steps + text 2,500 steps (~15 min) to hear the student's voice cheaply before the full run. Background watcher started.
- After the probe: pull checkpoints, run scripts/gate_check.py for durations (1-frame share <=25%, max <=10 frames), periodicity >=0.60, flat/>4k/>7k, render faithful + air3 samples; only then decide on the full 12k run.

## 2026-11-18 Critical re-test of the nicole 500-step probe: FIX H vs raw head
- Owner: the tiny student sample was very watery; but noted Yati became acceptable only after changes we made along the way. Correct critical observation: the nicole tiny sample was rendered with the RAW training head (ms_train eval wavs), while Yati's accepted v0.4.1 uses the FIX H export (gap-masked noise + burst gate + gain smoothing + air shelf).
- Exported the SAME 500-step decoder checkpoint two ways: faithful (no processing) and FIX H (Yati recipe); rendered 8 sentences with teacher clips alongside.
- Page runs/listen/nicole_fixh_probe.html (localhost:8931/nicole_fixh_probe.html): TEACHER / student raw / student FIX H. Also note: probe = 500 steps + 624 rows vs Yati 12,000 steps + 1,281 rows, so it is badly undertrained - the earlier verdict was premature.

## 2026-11-18 Path B tested: Yati clean student -> one VC to nicole
- Diagnosed owner's 'lack of clarity' as the student's own synthesis quality (muffled+robotic), not loudness/knobs (those were amplitude-invariant metrics).
- Path B: synthesized 8 sentences with Yati v0.4.1 (12k-step clean student), converted once to nicole via ChatterboxVC on Kaggle CPU kernel sm-15b-yati-vc (4.8 min, 0 fails). Output RMS ~0.15 matches teacher ~0.147.
- Page runs/listen/pathB.html (localhost:8931/pathB.html): TEACHER vs PATH B vs YATI, loudness-matched. Awaiting owner verdict.
- Meanwhile rebuilt rows locally from all 1,281 nicole clips with correct flags (--max-last 1.0 --max-tail 99) to recover the 657 clips the defaults discarded; path A (full training) kept as fallback.

## 2026-11-18 Approved Path-B distillation; small train test inserted
- Owner approved ~1.5-2 h GPU. Plan: 1,281 Yati student clips (generated locally in 3 min) -> one VC to nicole (sm15c kernel) -> SMALL probe (2k dec + 1.5k text, sm15d kernel) -> owner listens for soft/watery -> full 12k run only if clean.
- Old-row rebuild with correct flags finished: 1252/1281 rows (vs 624 with defaults), confirming the tail-junk data loss.
- sm15c dataset upload running in background; sm15d probe kernel written and ready (kernel_sources sm15c VC output).

## 2026-11-18 Small probe gate: level OK, slight phasey on some words -> phase stabilization test
- Owner: probe student level OK, some words a bit phasey. No code until thinking done; decided (1) full training stabilizes phase, (2) free export-level phase knobs untested.
- export_fixh.py gained --phase-smooth N (circular-average u over N frames) and --phase-shrink F (multiply phase angles). Exported two variants from the 2k probe ckpts (FIX H + air6 base).
- Page runs/listen/probe_phase.html (localhost:8931/probe_phase.html): TEACHER vs probe current / phase smooth5 / phase x0.5, loudness-matched. Owner to pick; if a phase variant matches TEACHER, bake it into the full 12k export.

## 2026-11-19 Full 12k Path-B training approved and launched
- Owner verdict on phase stabilization: smooth5 and phase x0.5 both WORSE than current -> learned phase is useful; phasey texture is undertrained-head, not a knob. Explicit approval for full run.
- Kernel suarifymalaysia/sm-15e-full: same Path-B data (948 converted clips via sm15c kernel source), decoder 12,000 steps + text 10,000 steps from Yati init (~40 min GPU). Outputs dec_nicoleb_full.pt / text_nicoleb_full.pt.
- After: export FIX H + air6 (proven recipe), loudness-match to Path-B teacher, final listening gate -> publish as 8 MB voice only if clean.

## 2026-11-19 HANDOFF.md written for the other agent
- Comprehensive dated handoff: project, published state, Kaggle/env traps, full chronology 11-14..11-19, key decisions (FIX H, loudness matching, ms_rows flags, rejected no-training fixes), active files/kernels, pending work. Reviewer should read HANDOFF.md first.


## 2026-11-19 FULL 12k Path-B student exported — final gate open
- sm-15e-full COMPLETE: dec step 12000, text step 5000 (early stop). Pulled after freeing 3 GB of old kernel caches.
- Exported runs/nicole_full_export/paradee_int8_webgpu.onnx (8.47 MB, FIX H + air6 + gn smoothing, webgpu patch 0 float64).
- Final page runs/listen/nicole_full.html (localhost:8931/nicole_full.html): TEACHER (Path B) vs FULL 12k, 8 sentences, loudness-matched. Owner verdict needed: publish as 8 MB voice only if clean (no soft/watery/phasey).

## 2026-11-19 Nicole v0.1 published (owner allowed) + REVIEW.md adopted
- Published https://huggingface.co/Suarify/kecil-tts-nicole-v0.1 from the full 12k Path-B export (8.47 MB, sha match, 0 float64, 44 files); samples from the model, card documents the known residual watery/phasey limitation and that CER is not yet measured. Space dropdown gained nicole v0.1.
- Owner instructed: follow REVIEW.md phase by phase; it supersedes HANDOFF 7 items 2-3 (no fenrir / further nicole work until Phase 3). REVIEW root-cause: the residual murk is F0 prediction (17.4% voiced dropouts, 0/200Hz regression + decoder never sees predicted inputs), not the 8M head. Plan: P0 eval harness, P1 log-F0+uv text side, P2 GTA decoder, P3 real adversarial stage, P4 data, P5 rollout.

## 2026-11-19 REVIEW.md Phase 0 DONE: eval harness + baselines
- Adopted REVIEW.md v8 (supersedes HANDOFF 7 items 2-3: no fenrir/further nicole until Phase 3).
- Fixed 80-sentence eval set eval/sota_eval.tsv (40 general + 20 code-switch + 20 numbers, stable ids from data/text/test_200.tsv).
- scripts/sota_eval.py: renders with ORT CPU -> runs/sota/<tag>/<id>.wav, pYIN F0 health (fmin=70, same as ms_rows), UTMOS22-strong at 16 kHz, predicted-vs-measured F0 on 60 held rows, leaderboard.tsv.
- Baselines: TEACHER (15 Chatterbox-converted clips) UTMOS 3.5534/3.1623 (mean/p10); Yati v0.4.1 2.1447/1.7684, F0 dropout 16.4%, V/UV err 25.4%, jumps>25Hz 9.2%, span ratio 2.867; Michael v0.3.1 1.8132/1.4449.
- Confirms REVIEW's root-cause: the text side's F0 prediction, not the 8M head (bigger head gave zero gain). Next: Phase 1 log-F0 + uv head (CPU first, gate 1 automatic + owner A/B).

## 2026-11-19 REVIEW Phase 1 (CPU) DONE: log-F0 + voiced/unvoiced text head implemented + validated
- New `paradee_ms/student_uv.py` (copied to kaggle/ds_pms for Kaggle use): `TextStudentUV` subclasses TextStudent, adds `uv_proj` alongside F0_proj on the shared prosody features; `prosody()` is a line-for-line match of Kokoro `F0Ntrain` plus the UV head; `f0_hz()` converts normalised log-F0 + UV logit back to Hz with an ONNX-safe 5-frame majority filter (3D fix applied: 2D input made ONNX AveragePool reject the graph). Helpers `set_lf0_stats` (mu/sd from voiced frames of train rows) and `reinit_f0_proj` (old F0_proj emitted Hz ~200, must start near 0 for normalised log-F0) MUST be called after strict=False load of `runs/k16_final/textA_c4_last.pt`.
- `ms_train.py` gains `--uv-text`: loads TextStudentUV strict=False, reinit+stats, target lf0n=(log F0 - mu)/sd on voiced frames, l_f0 = weighted L1 (1.0 voiced / 0.1 unvoiced), l_uv = BCE-with-logits (voiced = F0>50) weight 1.0, l_n + l_dur + l_asr unchanged; checkpoint selection on l_dur + l_asr + l_uv; evaluate() and free-running call adapted to the 5-value forward. Old path untouched.
- `scripts/export_fixh.py` gains `--uv-text`: Paradee2.forward uses `ts.prosody` + `ts.f0_hz` instead of `predictor.F0Ntrain`; loads TextStudentUV strict=True (mu/sd buffers come from the checkpoint); --state+--uv-text guarded off.
- CPU validation: 2-step train smoke (losses fell l_f0 2.35->2.33, l_uv 0.644->0.639); 100-step timing at bs=16 = **1.41 s/step** (>> REVIEW's 0.35 threshold, so 10k steps ~3.9 h on CPU). Full UV export smoke on the untrained init: int8 ONNX 8.47 MB, head no-op parity 3e-07, ORT length matches torch, webgpu_patch -> 0 float64 Cast ops.
- NEXT: needs owner's explicit GPU yes before Kaggle kernel `sm-20-f0uv` (10k steps, Kaggle account suarifymalaysia). Then export + sota_eval + Gate 1 (dropout<=3%, V/UV<=7%, jumps>25<=3%, span 0.7-1.5, UTMOS>=2.1447, CER not >0.5 worse) + blind owner A/B page.

## 2026-11-19 REVIEW Phase 1 GPU run prepared (owner approved Kaggle GPU)
- Owner explicitly approved Kaggle GPU for the 10k-step UV text training (CPU timing was 1.41 s/step, 4x over REVIEW's 0.35 threshold).
- Dataset `suarifymalaysia/sm21-f0uv-in` uploaded: all 16 PMS scripts (incl. new `student_uv.py` + modified `ms_train.py`), exact 1,281-row `data/teacher/shard_0000.pt`, and `textA_c4_last.pt`. First upload silently kept only the root-level file; re-uploaded flat -> 17 files, status ready.
- Kaggle kernel `suarifymalaysia/sm-21-f0uv-text-head` pushed; first run ERROR at env-setup because the dataset had not finished processing (glob found no ms_train.py). Re-pushed after the dataset showed all 17 files; now RUNNING/QUEUED. Kernel: proven torch 2.11.0+cu130 LAST env recipe, kokoro 0.9.4 + misaki stubs, `--uv-text --held 60 --steps 10000 --bs 16 --eval-every 500`; saves `/kaggle/working/text_f0uv.pt`.
- Corrected `ms_train.py` UV target to REVIEW spec: `v = F0>0`, log-F0 linearly interpolated across unvoiced gaps (edges held constant), normalised by mu/sd. Added `--uv-unvoiced-weight` (0.1; Gate 1 retry 0.0) and `--uv-uv-weight` (1.0; retry 2.0). Smoke: held l_f0 dropped 2.35 -> 1.42, confirming the earlier constant-0 target was inflating the loss. NOTE: the running kernel used the earlier constant-0 target; if Gate 1 fails, retry with the corrected code + retry weights.
- `scripts/sota_eval.py` gained `--uv-text` (loads TextStudentUV, uses prosody+f0_hz) and now uses the FIRST 60 rows (the actual training held-out set) instead of the last 60 for dropout/V/UV/jumps/span.
- `scripts/make_sota_p1_page.py`: builds the blind, RMS-matched, shuffled 12-sentence owner page with label reveal, "pick cleaner" + "watery" ticks and P1>=8/12 tally. Smoke-tested.
- v0.4.1 export recipe verified from the published ONNX: default FIX H (K=64, hang=21, burst gated, gap_keep=1, no air/no smoothing); P1 will be exported with identical settings for a fair A/B.
- NEXT: when kernel completes -> download `text_f0uv.pt` -> export with `--uv-text` + FIX H -> webgpu patch (0 float64) -> sota_eval (Gate 1 automatic) -> if pass, build `runs/listen/sota_p1.html` for owner.

## 2026-11-19 Space UI polish (ready for the P1 voice rollout)
- `scripts/polish_space.py` (idempotent) updates `runs/local_v041/index.html`:
  - Sleeker type: Plus Jakarta Sans body + Outfit display (replaces Inter/Sora).
  - Live speed-highlight strip as the top row above the textarea: shows realtime factor, a progress bar, and a "+X% vs <previous version>" comparison. On every generation it updates with the measured speed for that run. Benchmarks are per-model in a SPEED map.
  - Sample-text chips now fill the textarea AND immediately trigger Speak.
  - Adds `Suarify/kecil-tts-yati-v0.5.0` ("v0.5 (SOTA P1)") to the Yati version dropdown (name is an assumption; confirm with owner before publish).
- `scripts/verify_space.py` smoke-checks all six changes. Space upload is deferred until P1 passes Gate 1 + owner A/B.
- Owner's rollout instruction: publish the P1 model, add the voice to the HF Space dropdown, and apply the UI polish. Publishing still gated on automatic + owner Gate 1.

## 2026-11-19 REVIEW Phase 1 GPU result: automatic Gate 1 FAILED (one retry permitted)
- Kernel `suarifymalaysia/sm-21-f0uv-text-head` COMPLETE in 12.8 min (train 8.7 min, 0.11 s/step on T4). Used the corrected interpolated log-F0 target (before l_f0 0.825 vs 2.35 old). Early-stopped with best step 500 (trained 4,500); held l_f0 0.825->0.679, held l_uv 0.620->0.542 at step 500, then overfit (held l_uv 0.87 at 3,500).
- Exported `runs/sota_p1_export/paradee_int8_webgpu.onnx` (8.47 MB, 0 float64) with identical v0.4.1 FIX H recipe, head parity 2.4e-07, length parity exact.
- `sota_eval.py --uv-text` on true held-out rows: UTMOS 2.1861/1.7750 (vs base 2.1579/1.7383, PASS), jumps>25Hz 5.05% (vs 10.28%, better but FAIL 3% gate), dropout 17.89% (vs 17.87%, FAIL), V/UV err 27.69% (vs 27.94%, FAIL), span ratio 3.35 (vs 3.32, FAIL).
- Diagnostic with TEACHER-FORCED alignment (removes free-running duration drift): base dropout 10.4%/uverr 17.9%, P1 dropout 11.7%/uverr 18.5% => the explicit UV head did not improve voicing prediction; free-running alignment adds ~7 points of error for both models.
- REVIEW.md rule: fail -> report numbers, try ONCE with l_uv weight 2.0 and unvoiced-F0 weight 0.0, then stop and report. Owner approval for that retry is pending (standing rule: every GPU step needs explicit yes).
- IMPORTANT: no publishing / no owner listening page until a gate passes. The Space UI polish (nicole speaker, chips auto-speak, big stats + waveform) is staged locally but NOT uploaded yet.

## 2026-11-19 REVIEW Phase 1 owner A/B page (owner chose to skip the automatic gate + retry)
- Owner explicitly chose "Skip to owner A/B" over the REVIEW-prescribed retry and over stopping Phase 1. The automatic Gate 1 remains FAILED; no publish decision yet.
- Built `runs/listen/sota_p1.html` via `scripts/make_sota_p1_page.py --baseline runs/sota/yati_v041 --p1 runs/sota/p1 --n 12`: 12 sentences (6 general / 3 codeswitch / 3 numbers), A/B shuffled per row, loudness-matched to equal RMS, "pick cleaner" + "watery" ticks, labels revealed only after pressing Reveal, P1>=8/12 tally. 24 wavs verified (24 kHz, non-empty).
- Page opened in the Review pane for the owner. Await owner verdict before ANY publish or Space upload.
- Space UI v2 staged locally (NOT uploaded): `scripts/polish_space2.py` adds Nicole as a real third speaker (version dropdown underneath), auto-speaking sample chips (already in v1), a large render-stats panel (realtime factor / time taken / first audio / audio length / device + comparison bar) and a live waveform canvas with a moving playhead; JS syntax checked with `node --check`.

## 2026-11-19 Phase 1 verdict + Phase 2 approved
- Owner listened to the blind page (`runs/listen/sota_p1.html`) and said "both almost the same" -> P1 is NOT published; v0.4.1 remains the released voice. Consistent with the flat objective numbers (UTMOS +0.03, voicing unchanged).
- Owner explicitly approved Phase 2 GTA with Kaggle GPU: fine-tune the Yati v0.4.1 decoder on its own text-model-predicted F0/N (the second root cause in REVIEW.md: the decoder only ever saw measured inputs). 3k steps, bs16, seg64, lr1e-4.

## 2026-11-19 REVIEW Phase 2 (CPU) DONE: GTA inputs + decoder trainer + Kaggle kernel prepared
- Owner chose "Phase 2 GTA" (after initially saying "retry" then clarifying), with explicit GPU approval already given for Phase 2.
- `scripts/make_gta.py`: ran the P1 text model (TextStudentUV, `runs/kaggle_sm21/text_f0uv.pt`) over all 1,281 rows with MEASURED durations, stored `gta_F0` (Hz via f0_hz), `gta_N`, `gta_asr` = asr_tok @ aln -> `data/teacher_gta/shard_0000.pt` (785 MB).
- `ms_train.py decoder` gains `--gta-p`: per batch item, with prob gta_p uses GTA inputs; c0 from the INPUT F0; harm_loss/mod_loss now per-item and weighted 0 for GTA items (their pitch no longer matches the target audio), mel+STFT unchanged. Intermediate checkpoints saved every 1k steps. Old path (gta_p=0) unchanged. Smoke-tested 2 steps OK (fixed a vm reduction shape bug).
- Dataset `suarifymalaysia/sm22-gta-in` uploaded and verified ready with all 17 files (GTA shard + dec_c4_last.pt + scripts).
- Kernel `suarifymalaysia/sm-22-gta` pushed and QUEUED: `dec --gta-p 0.7 --held 60 --steps 3000 --bs 16 --seg 64 --lr 1e-4 --eval-every 500`, saves `dec_gta.pt`.
- `scripts/make_sota_p1_page.py` parameterized with `--label-base`/`--label-p1` so it can build the P2-vs-P1 page; JS syntax checked.
- NEXT on completion: download dec_gta.pt -> export with `--uv-text` + identical FIX H -> webgpu patch (0 float64) -> sota_eval tag p2 -> Gate 2 (UTMOS >= P1+0.1 OR owner picks P2 >=8/12) -> publish only after owner gate.

## 2026-11-19 REVIEW Phase 2 result: Gate 2 FAILED (P2 worse than P1)
- Kernel `suarifymalaysia/sm-22-gta` COMPLETE; `dec_gta.pt` downloaded after a disk-full retry (freed the upload staging folder). Training was 3k steps, bs16, seg64, lr1e-4, gta_p 0.7.
- Exported `runs/sota_p2_export/paradee_int8_webgpu.onnx`: 8.47 MB, 0 float64, head parity 1.5e-07, identical FIX H recipe, 13.5x realtime.
- `sota_eval.py` p2: UTMOS 2.0101/1.5395 vs P1 2.1799/1.7647 (and v0.4.1 2.1579/1.7383). Gate 2 automatic threshold >= P1+0.1 (2.28) FAILS decisively. F0 health unchanged (text model unchanged: dropout 17.89%, uv_err 27.69%, span 3.348).
- Insight: GTA fine-tuning trains the decoder on the text model's own predicted F0/N, but those inputs still carry ~18% voicing dropout, so the decoder learned to reproduce the artifacts and got worse. Phase 2 was doomed by Phase 1's unresolved F0 problem; GTA only helps once the text-side F0 is genuinely clean.
- REVIEW says if both Gate 2 paths fail, keep P1's decoder and still go to Phase 3 (use GTA inputs there anyway). Nothing published; v0.4.1 remains the live voice. Owner decision pending (blind P2 A/B vs Phase 3 vs stop).

## 2026-11-19 REVIEW Phase 3 (CPU) DONE: GAN stage implemented + kernel submitted
- Owner approved Phase 3 after both P1 and P2 were judged "almost the same" in blind A/Bs. Nothing published; v0.4.1 remains live.
- REVIEW Phase 3 recipe implemented in `ms_train.py --gan`: discriminator warm-up 500 steps (G frozen), then 8k joint steps; G lr 1e-4, D lr 2e-4 (`--dlr`), betas (0.8,0.99); fp16 autocast + GradScaler on CUDA; GTA prob 0.7 with harm/mod losses zeroed on GTA items (per-item weights); s/step printed at step 50 for the profiling gate; checkpoints every 1k; 12 held sentences rendered at 2k/4k/8k. Smoke-tested 4 steps on CPU.
- Init from the **v0.4.1 decoder** (`dec_c4_last.pt`), per REVIEW: "if neither Gate 2 path passes, keep P1's decoder" — P2's decoder was objectively worse (UTMOS 2.01), so starting from it would be a regression. Documented deviation from the literal "from the Phase-2 decoder" wording.
- `suarifymalaysia/sm22-gta-in` dataset re-versioned with the GAN-capable trainer (verified Kaggle ms_train.py bytes == local 20416).
- Kernel `suarifymalaysia/sm-21-gan` pushed and RUNNING: 8500 steps total (500 warmup + 8000 joint), bs16, seg32, max 100 min; saves dec_gan.pt + step2k/4k/6k/8k checkpoints for UTMOS selection.
- Profiling gate: if step-50 s/step > 0.8, stop and report instead of burning quota.
- Next on completion: pick checkpoint by UTMOS on the 80-sentence set -> export with --uv-text + FIX H -> webgpu patch -> Gate 3 3-way blind page {v0.4.1, P2, P3} (pass = P3 >=8/12 and watery <=2/12) -> publish Yati v0.5 only after owner gate.

## 2026-11-19 Phase 3 first run FAILED (code bug), fix implemented + verified
- Kernel `sm-21-gan` COMPLETE (0.34 s/step at step 50, within the <=0.5 target), but the discriminator loss went NONFINITE (NaN) right after the 500-step warm-up and stayed NaN for all 8k joint steps. GradScaler skipped every optimizer step, so `dec_gan*.pt` are BYTE-IDENTICAL to v0.4.1 (mean abs weight diff 0.0) - the generator never updated. This is a numerical bug in my GAN implementation, NOT a genuine result that adversarial training fails.
- Root cause: autocast wrapped the whole loss computation including the squared LSGAN terms `((r-1)**2)` / `f**2`, which overflow fp16.
- Fix in `ms_train.py` (smoke-tested 6 CPU steps, generator now changes: mean abs diff 1.3e-05): fp16 autocast now covers ONLY the generator forward; `y = y.float()` before all losses; D and G adversarial losses computed in fp32; D grad clipping after `scaler.unscale_(optD)`; explicit NONFINITE detection with abort after 20 consecutive bad steps.
- Need owner's explicit GPU yes before re-uploading the fixed trainer + re-running the ~50 min kernel. No publishing; v0.4.1 remains live.

## 2026-11-19 Phase 3 rerun submitted (owner approved)
- Re-uploaded the fixed trainer to suarifymalaysia/sm22-gta-in (verified Kaggle ms_train.py bytes == local 20920, NONFINITE guard present). Kernel suarifymalaysia/sm-21-gan re-pushed and RUNNING with the corrected fp32 GAN losses. Same recipe: 500 D warmup + 8k joint steps, bs16 seg32, 0.34 s/step measured previously.
- Watch for NONFINITE lines in the log; if the fp32 fix failed to stop the divergence, report rather than burning more quota.

## 2026-11-19 Yati v0.5 PUBLISHED + Space updated (owner approved despite modest gain)
- Owner: "the difference here not significant. but ok publish n go phase4".
- Published `Suarify/kecil-tts-yati-v0.5.0` from `runs/p3_dec_gan_export/paradee_int8_webgpu.onnx` (best P3 checkpoint, step 5407, UTMOS 2.2748/1.7932). 42 files, remote sha == local, 0 float64, 8.47 MB. Card is honest: +0.12 UTMOS over v0.4.1, owner did not find the difference clearly perceptible, F0 voicing unchanged, large gap to teacher remains.
- Space `Suarify/suaradee-web` updated with the polished UI (verified remotely): Yati v0.5 dropdown, Nicole as a third speaker with its own version dropdown, auto-speaking sample chips, large render-stats panel + live waveform, sleeker fonts.
- Phase 1/2 checkpoints were NOT published.

## 2026-11-19 Phase 4 step 1 (CPU) DONE: 3,400 new sentences selected
- Owner said "publish n go phase4" after the modest v0.5 gain. v0.5 published + Space updated (see prior entry).
- `scripts/make_p4_texts.py`: loaded 7,500 corpus rows, excluded the 1,281 already used, greedy token-bigram coverage (40k bigram space; char bigrams saturated at 254 sentences so switched to token bigrams) -> 2,500 sentences covering 30,121 token bigrams. 300 numbers from the numbers category.
- `scripts/make_p4_codeswitch.py`: corpus had only ~218 unused code-switch sentences, so generated 600 Malay-frame + English-loanword sentences from 20 templates x word lists (REVIEW says "English loanwords in Malay frames, use the lexicon from C3").
- `scripts/merge_p4_texts.py` -> `eval/p4_new_texts_final.tsv`: 2,500 greedy + 600 codeswitch (204 corpus + 396 synthetic) + 300 numbers = 3,400 rows, asserted.
- NEXT (GPU, needs owner yes): 50-clip timing kernel testing batched Chatterbox generation + fewer sampling steps; pick fastest config with CER <=2.5% on those 50; report clips/hour and total GPU-h for owner approval BEFORE generating all 3,400.

## 2026-11-19 Phase 4 step 2 (GPU, owner approved): 50-clip timing kernel submitted
- Owner explicitly approved the timing kernel. Dataset suarifymalaysia/sm23-p4-timing-in uploaded (50 texts from eval/p4_new_texts_final.tsv + ref_ms.wav), status ready.
- Kernel suarifymalaysia/sm-23-p4-timing pushed and QUEUED. Tests 3 configs on 15 clips each: A baseline (audio_prompt_path per call), B prepare_conditionals once, C conditionals once + top_p 0.9. Measures generation s/clip, clips/hour, projected GPU-h for all 3,400, and Whisper large-v3 CER (language=malay, capped 1.0, median + capped mean). Fastest config with CER median <=2.5% wins.
- IMPORTANT: owner must approve the reported total GPU-h BEFORE the full 3,400-clip generation is committed (REVIEW rule).

## 2026-11-19 Phase 4 step 2 result: timing done, CER gate NOT met
- Kernel suarifymalaysia/sm-23-p4-timing COMPLETE. Results on 15 clips/config: A baseline 12.14 s/clip, CER median 0.0949 / capped mean 0.1243 -> 297 clips/h, 11.5 GPU-h for 3,400. B conditionals-once 12.58 s/clip, same CER -> 11.9 GPU-h. C cond+top_p0.9 12.61 s/clip, CER 0.1136/0.1366 -> 11.9 GPU-h. NONE meets the REVIEW <=2.5% CER gate; conditionals-once gave NO speedup (actually slightly slower).
- Warning in the log: Chatterbox T3 hit token-repetition and forced EOS on some clips - a known source of truncation/poor CER on long dense sentences. The 15 test sentences are the greedily-selected dense ones (percentages, proper nouns), which are harder than the conversational teacher set.
- CER may be a scorer artifact: my kernel used direct WhisperForConditionalGeneration.generate(language=malay), whereas the project-validated k18 scorer uses the HF sr pipeline with generate_kwargs. Teacher clips normally score ~1.9% median. Need to re-score the SAME generated wavs with the k18 scorer before concluding generation is bad.
- Owner approval needed before any full 3,400-clip generation (REVIEW rule). No generation committed.

## 2026-11-19 Phase 4 CER scorer verification (owner approved) - submitted
- Kernel suarifymalaysia/sm-24-cer-verify QUEUED. Re-scores the SAME 15 generated wavs (from sm23 config A) two ways: the project-validated k18 HF sr pipeline scorer (generate_kwargs language=malay) and the direct WhisperForConditionalGeneration scorer used in sm23. Prints per-clip hypotheses + CERs and aggregate medians/means.
- Dataset suarifymalaysia/sm24-cer-verify-in: 15 flat wavs + refs.tsv, status ready.
- If the pipeline scorer gives ~2% CER, the sm23 gate failure was a scorer artifact -> proceed with full generation planning (11.5 GPU-h for baseline). If it still gives ~9%, generation quality is genuinely below the 2.5% gate -> stop and report.

## 2026-11-19 Phase 4 CER verification result: gate genuinely FAILED
- suarifymalaysia/sm-24-cer-verify COMPLETE. Both the validated k18 pipeline scorer and the direct scorer give IDENTICAL results on the same 15 wavs: median CER 0.0949, capped mean 0.1243. So the sm23 failure was NOT a scorer artifact.
- Error analysis: errors concentrate on the dense greedily-selected sentences (percentages spelled as words vs Whisper digits, English proper nouns). Number-token stripping changed median only 0.0949 -> 0.0811 (some clips got worse), so it is not purely a formatting artifact. Chatterbox T3 also hit token-repetition / forced-EOS on some clips (truncation).
- No timing config met the REVIEW <=2.5% CER gate; none was faster than baseline. Baseline generation cost ~12.14 s/clip => 297 clips/h => 11.5 GPU-h for 3,400 clips (batching / conditionals-once gave no speedup).
- Owner decision pending before any full generation. No generation committed.

## 2026-11-19 Phase 4 representative 50-clip CER retest (owner approved) - submitted
- The 15-clip gate failed (9.5% CER) but was biased: all 15 were dense greedy-selected sentences. Owner chose a representative retest.
- scripts/make_p4_cer50.py -> eval/p4_cer50.tsv: 20 general / 10 conversational / 10 codeswitch / 10 numbers, stratified from the 3,400 final sentences.
- Dataset suarifymalaysia/sm25-cer50-in uploaded (50 texts + ref_ms.wav). Kernel suarifymalaysia/sm-25-cer50 QUEUED: baseline Chatterbox config on all 50, CER via the validated k18 pipeline scorer, per-category medians, s/clip and projected GPU-h for 3,400.
- Decision rule: if CER median <=2.5% -> owner approves the reported GPU-h then full generation. If still high -> stop Phase 4 and report.

## 2026-11-19 Phase 4 representative CER retest result - overall gate still FAILED
- Kernel suarifymalaysia/sm-25-cer50 COMPLETE: 50 stratified sentences, baseline config. Speed 8.59 s/clip, 419 clips/h, 8.11 GPU-h for 3,400.
- Raw CER (my scorer omitted normalize()): general 1.69%, conversational 1.93%, codeswitch 14.9%, numbers 56.9%, overall median 5.05%.
- Re-scored locally WITH the correct k18 clean_txt (normalize() first): general 2.16%, conversational 1.93%, codeswitch 10.75%, numbers 43.78%, overall median 4.1%. Still above the 2.5% gate. Some number errors are genuine mis-transcriptions (e.g. 'lima perpuluhan lapan peratus' -> '5 masan proper pilihan'), not just digit/word formatting.
- Conclusion: the 2,551 general+conversational new sentences are clean (~2% CER); the 849 numbers+codeswitch sentences are noisy and would teach wrong pronunciations if used as teacher data.
- Owner decision pending: generate only the clean 2,551 (~6 GPU-h), generate all 3,400 (~8.1 GPU-h) accepting noise, or stop Phase 4.

## 2026-11-20 FINAL DECISION: stop here, keep Yati v0.5 as the live voice
- Owner considered a balanced GAN continuation (~4-5h) and a small-data retrain (~6h) within the remaining ~7 GPU-hours, and chose: STOP. No further GPU spend.
- Final state: `Suarify/kecil-tts-yati-v0.5.0` published (42 files, sha match, 0 float64, 8.47 MB, UTMOS 2.275/1.793); Space `Suarify/suaradee-web` updated with the polished UI (Yati v0.5, Nicole as 3rd speaker, auto-speak chips, stats + waveform, sleeker fonts). Nicole v0.1 and Michael v0.3.1 remain published.
- REVIEW phases 1-4 all exhausted: P1 (log-F0+UV) and P2 (GTA decoder) failed gates and were judged "almost the same" by the owner; P3 (GAN, after a NaN bug fix) gave +0.12 UTMOS and was published as v0.5 despite the owner noting the difference was not clearly significant; P4 (more data) failed the CER gate (general/conversational clean at ~2%, numbers/codeswitch noisy) and was not pursued given the owner's 7h budget.
- Route A (fine-tune Kokoro-82M + distil, 10-20 GPU-h) remains available ONLY with a fresh owner decision (REVIEW section 4).

## 2026-11-20 Space bug fixed + UTMOS results posted on model card
- Owner reported the HF Space was broken ("model not loaded"). Diagnosed via browser console: two `ReferenceError: Cannot access 'tts'/'lastAudio' before initialization` in `index.html` from the UI polish (speedStrip() called by drawSpeakers() before `let tts`, and a resize listener before `let lastAudio`). This aborted the whole module script before load() ran.
- Fix: declared `let tts, lastAudio;` at the top of the module and removed the later declarations. JS syntax checked (`node --check`), Space re-uploaded, verified LIVE in the browser: "Model ready on webgpu in 15.8 s", Speak button active, 3 speakers rendered. Remaining console entries are benign ONNX Runtime warnings only.
- Owner asked to post the UTMOS result on the model: updated `Suarify/kecil-tts-yati-v0.5.0` README.md with the full development leaderboard (teacher 3.553, v0.5/P3 2.275, P1 2.180, v0.4.1 2.158, P2 2.010) plus the Phase-4 CER finding (general 2.2% / conversational 1.9% pass; numbers 43.8% / codeswitch 10.8% fail). Verified remotely.

## 2026-11-20 Space UI tuning v3 (5 owner requests) - uploaded + verified live
- (1) Stats panel + waveform hidden by default (`#stats,#wave{display:none}`), revealed only after speech is generated (`.on` class added in the Speak handler).
- (2) The '3 · Runs anywhere: edge, IoT, kiosk...' card is now a modal popup (`.modal-wrap#edgeModal`), opened by a new ⓘ info icon beside the Kecil-TTS header, closed via × / backdrop click / Escape.
- (3) On mobile the BM/light toggles stay in the top-right row (`.headrow{flex-wrap:nowrap}`, `.topbar{margin-left:auto}`, smaller tg/badge/h1 instead of wrapping).
- (4) Keywords card is visually hidden (`.seo-hidden`: height .1px, opacity .001) with all 37 keywords written as one sentence inside for SEO.
- (5) The 'Need custom software...' CTA section card was removed; its content now sits in a footer row (`.cta-row`) and 'Contact Suarify → suarify.my/contact-form' is a plain sentence with an inline link, not a button.
- Verified live in browser: model loads ("Model ready on webgpu in 5.5 s"), Speak active, 3 speakers, stats/wave hidden, modal opens/closes, keywords opacity .001, CTA row present, no CTA button, zero JS errors.

## 2026-11-20 Space UI v4: smaller info icon + streaming timing clarified
- Owner: info icon 30% smaller and floating top-right after the Kecil-TTS word -> .infobtn 30->21px, font 17->12px, translateY(-4px), margin 8px (verified live 21px).
- Owner: "why time taken >2s even in streaming" -> two causes fixed: (a) the timer t0 was started before `await ensureAudio()` (AudioContext resume + FX chain build) and phonemize(), so setup time was counted as synthesis; moved t0 to just before generation. (b) Streaming only speeds FIRST audio, not total synthesis; the stats panel now swaps labels in streaming mode: "Time taken" -> "Time to audio" (shows firstT) and "First audio" -> "Total gen" (shows total), with an explanatory note in the info line.
- Verified live after generation in streaming mode: "Time to audio 554 ms", "Total gen 554 ms", render speed 5.6x, no JS errors. Longer sentences will show first-audio < total gen.

## 2026-11-20 Space UI v6 (5 owner requests) - uploaded + committed to GitHub
- (1) Nicole speaker sub no longer says "af_nicole"; note simplified.
- (2) "Runs anywhere" modal title no longer prefixed with "3 ·" (EN + BM).
- (3) The two modal SVGs now gently float/rotate via @keyframes floaty (3.2s/3.8s, staggered), disabled under prefers-reduced-motion.
- (4) Removed the old Yati note "Understandable and the most accurate so far...".
- (5) Footer gets a header-like gradient separator: 3px teal->navy->teal bar on top, 4px navy->teal bar at bottom, subtle tint background.
- Verified live via browser: 3 speakers, Nicole sub "female voice", modal title clean, svgAnim "floaty", footer gradient present, zero JS errors.

## 2026-11-20 Phase 5 rollout STARTED: Fenrir first (owner approved, 6.5h GPU budget)
- Owner: "train fenrir and nicole from yati now... yati is clearest, michael/nicole volume very soft... prioritize fenrir first... 6.5hr GPU left". Approved plan: full fenrir pipeline (~1.8h) -> export/eval/blind page -> owner gate BEFORE Nicole.
- Root cause of softness measured: Nicole output RMS 0.018-0.035 vs Yati v0.5 0.036-0.052; the VC output was quiet and the model learned that level. Fix: normalize every VC clip to RMS 0.07 (Yati teacher-row level) with peak limiting at 0.95 in the VC kernel.
- Step A (CPU, done): `scripts/gen_yati_v05_clips.py` generated all 1,281 Yati v0.5 student clips from `runs/p3_dec_gan_export/paradee_int8_webgpu.onnx` -> `runs/pathb_yati_v05_clips` (2.1 min).
- Dataset `suarifymalaysia/sm26-fenrir-vc-in` uploaded (bundled ONE .pt of 1,281 clips, ref_fenrir.wav, texts.tsv - avoids the 948/1281 wav-folder mount bug). Kernel `sm-26-fenrir-vc` RUNNING: ChatterboxVC to am_fenrir + loudness normalization -> `fenrirB.pt`.
- Kernel `sm-27-fenrir-train` prepared: rows (ms_rows with `--voice am_fenrir --f0-min 50 --f0-max 300`, male settings) -> text 10k UV from Yati v0.5 text -> GTA rows -> decoder 12k GTA from Yati v0.5 decoder -> short GAN 2k (D warmup 100, D lr 1e-4 to avoid the D-wins problem). Saves fenrir_text/dec/gan.pt.
- Inits: text from `runs/text_f0uv.pt`, decoder from `runs/kaggle_sm21gan/dec_gan.pt` (the Yati v0.5 weights).

## 2026-11-20 Space UI v7 (owner request while Fenrir trains)
- Effects lab is now a modal popup (`.modal-wrap#fxModal`), opened only by a new ⚙ gear icon beside the speed/stream row; closes via ×, backdrop click, or Escape.
- Speak + Replay buttons moved to their own row BELOW the speed/stream row.
- The status sentence ("first audio after ... on webgpu ...") and Phonemes line moved ABOVE the textarea (right under the waveform) and shrunk to 10px micro text.
- Verified live: gear opens/closes the modal, info font-size 10px, info above textarea, model loads normally, zero JS errors.

## 2026-11-20 Space UI v8 (owner request while Fenrir trains)
- (1) The "first audio... / Phonemes" micro line hidden (elements kept with `hidden` attr so JS refs stay valid - removing them broke load() with a null textContent error, fixed and re-verified).
- (2) Benchmarks card is now a `<details id="benchd"><summary>` collapsible, collapsed by default (no `open` attr).
- Verified live: infoVisible false, phVisible false, benchOpen false, model loads, 3 speakers, zero JS errors.
