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
