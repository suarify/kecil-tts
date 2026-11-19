# REVIEW + PLAN v8 — from "decent" to the best tiny Malay TTS (Kecil-TTS)

Written 2026-11-19 by a reviewing agent after reading HANDOFF.md, STATUS.md, `kaggle/ds_pms/ms_train.py`,
`student.py`, `ms_rows.py`, `discriminators.py`, `scripts/export_fixh.py`, and `suaradee-michael/PLAN*.md`.
**This file is the live plan.** It supersedes HANDOFF.md §7 items 2–3 (publishing nicole, starting fenrir) — those
wait until Phase 3 here is done. Everything else in HANDOFF.md (env facts, rules, lessons) still applies.

Owner's goal, in his words: "sota kokoro bm" — the best-sounding tiny Malay TTS, still ~8.5 MB, Python + browser.

---

## 0. Standing rules (unchanged, repeat them to yourself)

- Every GPU step needs the owner's explicit "yes" in chat. CPU work does not.
- Kaggle account `suarifymalaysia` only. Quota resets 2026-11-21 (30 h/week).
- Publish only with `scripts/publish_hf.py ... --skip-space`. Never publish without an owner gate.
- Scoring: Whisper large-v3, `language='malay'`, CER after `normalize()`, capped at 1.0 per sentence; report
  **median AND capped mean**. Find the scorer used for the v0.4.1 card (`grep -rl large-v3 kaggle scripts`).
- Every listening page is **loudness-matched** (equal RMS to the reference clip). Blind, shuffled, labels revealed after picking.
- After each step: append to STATUS.md, commit (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
- Local torch: `.venv/Scripts/python.exe`, `PYTHONUTF8=1`, misaki stubs on `PYTHONPATH` (`.stubs` or `runs/k09/stubs`).
- After any ONNX export: `scripts/webgpu_patch.py IN.onnx OUT.onnx` (0 float64 ops).

---

## 1. Review: what is right, what is wrong

### Right
- Single voice conversion (Path B) is cleaner than double conversion; generating teacher audio from Yati avoids trailing junk.
- FIX H, loudness matching, `--max-tail 99`, checkpoint selection on `l_dur + l_asr`: all correct, keep them.
- Owner-ear gates before spending: correct.

### Wrong (the root cause is misnamed)
HANDOFF §4 (11-17) concludes "the residual murk is the 8M parametric head". The project's own evidence says otherwise:

| Evidence (date, source) | What it shows |
|---|---|
| damp_where test, 2026-11-15 (STATUS l.103) | Decoder fed **measured** F0/N: damp 1/5. Same decoder fed **predicted** F0/N: damp 4/5. Chatterbox/Kokoro 0. |
| f0n_diag2, 2026-11-15 (STATUS l.104) | Predicted F0 drops out (<50 Hz → noise) on **17.4%** of voiced frames; 10–90% span 227 Hz vs 69 Hz measured. |
| Bigger Michael head, 2026-11-17 | 320-dim/6 blocks: distance 3.39 vs 3.39, held loss 2.515 vs 2.497 → **zero gain**. If the head were the limit, a bigger head would help. |
| Text-side logs k10/k12 | held F0 loss 0.56 vs train 0.12 → the F0 predictor overfits ~2 h of data. |

The code explains why:

1. **F0 is trained as one Hz number with 0 for unvoiced** (`ms_train.py` text side:
   `l_f0 = L1(pF0, F0)` over all frames; `ms_rows.measure_f0` writes 0 for unvoiced). The decoder/head switches
   harmonics off when `f0 < 10` (`export_fixh.py` l.140, l.153). A regressor asked to output either ~0 or ~200 Hz
   hedges at voicing edges and in uncertain frames → dropouts mid-vowel and wild swings → noise bursts and
   wrong-pitch harmonics = "watery / phasey / damp".
2. **The decoder never sees the text model's outputs during training.** It trains only on measured F0/N and on
   Kokoro-teacher `t_en` features aligned by measured durations. At synthesis it gets *predicted* F0, N and the
   student's `asr_proj` features — three inputs it has never seen. `--aug` jitter is multiplicative and keeps the
   voicing mask, so it never teaches dropouts.
3. **The adversarial stage was never really run.** k11 did 843 steps with an untrained discriminator (d-loss ~3.9);
   sm-07-gan (Michael PLAN.md 2026-11-17) was written but no result exists. Spectral-only losses (mel + STFT + harm +
   mod) are the textbook cause of "swirly / phasey" fine detail. "GAN gave no gain" is **not** an established result.

Why did post-hoc F0 repair lose 8/8 (STATUS l.108)? Because repairing after the fact cannot know where the real
voicing is, and the decoder was never trained on such contours. The fix is in **training**, not in post-processing.

### Smaller issues
- **Path-B lost 26% of its data**: only 948/1,281 clips were visible on the Kaggle mount and the assert was lowered.
- **Path-B retrains the text side 10k steps on 948 clips.** VC keeps timing, so Yati's text side already has the
  right durations; retraining risks more overfit (see held/train F0 gap).
- **Path-B's ceiling is Yati.** The new voice is trained on Yati's own output, so it inherits Yati's prosody and
  artifacts. Improve Yati first, then regenerate Path-B data from the improved Yati.
- `measure_f0` uses pYIN `fmin=70`. Fine for female voices; for male voices (Michael 55–300 Hz) use `fmin=50`.

---

## 2. The plan (phases, each with a gate; do not skip gates)

Work on **Yati first** (data local: `data/teacher/shard_0000.pt`, 1,281 rows; the owner knows her sound).
Every improvement then flows to nicole/fenrir/michael in Phase 5.

| Phase | What | GPU | Owner gate |
|---|---|---|---|
| 0 | Eval harness + baseline numbers for v0.4.1 | 0 (CPU) | none |
| 1 | Text side: log-F0 + voiced/unvoiced head | 0–0.3 h | A/B 12 sentences |
| 2 | Decoder fine-tune on the text model's own outputs (GTA) | ~0.3 h | A/B 12 sentences |
| 3 | Real adversarial stage on short crops | ~1–1.5 h | A/B 12 sentences → ship Yati v0.5 |
| 4 | More data (pronunciation, English words) | 6–10 h | A/B + CER |
| 5 | Roll out to nicole / fenrir / michael | ~1–1.5 h per voice | per voice |

GPU numbers are estimates; measure s/step in the first 200 steps of every run and report before continuing.

### In-flight: `sm-15e-full` (nicole, running now)
1. Let it finish. Pull checkpoints into `runs/kaggle_sm15e/`.
2. Export FIX H + air6 + webgpu patch as planned, but build the page as **3-way, loudness-matched**:
   Path-B TEACHER / nicole-full / **Yati v0.4.1** (judge against Yati, not only the teacher).
3. Also export a variant **"nicole decoder + Yati text side"** with a log-F0 affine shift to nicole's pitch:
   compute `mu, sd` of log-F0 over voiced frames for Yati rows and nicole Path-B rows; at export
   `F0' = exp((log F0 − mu_y)/sd_y · sd_n + mu_n)` on voiced frames, 0 elsewhere. Put it on the same page.
4. Owner picks. **Do not publish nicole yet** unless the owner explicitly wants it now; tell him Phase 1–3 should make
   every voice better, and nicole's data will be regenerated from the improved Yati.

### Phase 0 — eval harness (CPU, ~1 h of agent time)
Create `scripts/sota_eval.py ONNX_PATH --tag NAME`:
- Fixed eval set (write once to `eval/sota_eval.tsv`): 40 general from `data/text/test_200.tsv`, 20 code-switch,
  20 numbers/dates/currency (from `eval/test_40.tsv`, `eval/e_test.tsv`), ids stable. 80 sentences total.
- Renders all 80 with onnxruntime CPU → `runs/sota/<tag>/<id>.wav`.
- **F0 health** (no Whisper needed): run pYIN (same settings as `ms_rows.measure_f0`) on each output.
  For held rows of `shard_0000.pt` (the 200 held ids), also compare the text model's predicted F0 to the
  measured one (reuse `scripts/f0n_diag2.py` logic): dropout % (measured voiced, predicted <50 Hz),
  V/UV error %, % of voiced→voiced jumps > 25 Hz, 10–90 span ratio.
- **Naturalness proxy**: UTMOS22-strong via `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)`
  at 16 kHz. Language-agnostic but sensitive to phasey/buzzy artifacts. Report mean and 10th percentile.
- Writes `runs/sota/<tag>/metrics.json` and appends one line to `runs/sota/leaderboard.tsv`.
- CER is done in the GPU kernels (Whisper large-v3); locally only if a run is cheap enough. Do not block on it.
Run it on: Yati v0.4.1 ONNX, Michael v0.3.1 ONNX, and 20 Chatterbox teacher clips (UTMOS only) for the ceiling.
Commit. This leaderboard is the automatic yardstick for every later phase.

### Phase 1 — text side: log-F0 + voiced/unvoiced head
Code (new file `kaggle/ds_pms/student_uv.py` + flags in `ms_train.py`; keep the old path working):
1. `TextStudentUV(TextStudent)`:
   - `self.uv_proj = nn.Conv1d(C, 1, 1)` with `C = self.predictor.F0_proj.in_channels`;
     init bias = `logit(voiced_fraction)` (≈ logit(0.6)), weight ~0.
   - Re-implement `F0Ntrain` inline (verify against the installed `kokoro/modules.py ProsodyPredictor.F0Ntrain`):
     `x = shared(en)`; F0 branch blocks → features `h`; `lf0n = F0_proj(h)`, `uv = uv_proj(h)`; N branch unchanged.
   - Buffers `lf0_mu`, `lf0_sd` (per voice, computed from training rows' voiced frames).
   - `forward` returns `dur, lf0n, uv_logit, N, asr_tok`. Helper `f0_hz(lf0n, uv_logit)`:
     `mask = sigmoid(uv) > 0.5`, then majority filter `avg_pool1d(mask, 5, 1, 2, count_include_pad=False) > 0.5`,
     `F0 = exp(lf0n·sd + mu) · mask`. **Do not smooth the contour itself.**
2. Targets (precompute once when loading rows): `v = F0 > 0`; `lf0c` = log-F0 linearly interpolated across unvoiced
   gaps (edges held constant); `lf0n_t = (lf0c − mu)/sd`.
3. Losses: `l_f0 = L1(lf0n, lf0n_t)` weighted 1.0 on voiced frames + 0.1 on unvoiced frames (masked by `fm`);
   `l_uv = BCE_with_logits(uv, v)` masked by `fm`, weight 1.0; `l_dur`, `l_n`, `l_asr` unchanged.
4. Init from the **Yati v0.4.1 text checkpoint** (the `--text` used for the v0.4.1 export; find it in STATUS "v0.4"),
   `strict=False` for the new layer. Re-init `F0_proj` bias to 0 (output is now normalised log-F0, not Hz).
5. Train 10k steps, bs 16, `--eval-every 500`, select on `l_dur + l_asr + l_uv`. Log held voiced-F0 MAE in cents and V/UV error.
   **Try CPU first:** time 50 steps locally; if ≤ 0.35 s/step, run locally in the background (no GPU approval needed).
   Otherwise one Kaggle kernel `sm-20-f0uv` (~15 min).
6. Export: in `export_fixh.py`, add `--uv-text` so `Paradee2.forward` calls `ts.f0_hz(...)` instead of
   `predictor.F0Ntrain`. Same decoder (Yati v0.4.1), same FIX H recipe, webgpu patch.
   The decoder needs no retraining for this phase: its input format (Hz, 0 = unvoiced) is unchanged.

**Gate 1 (automatic, before the owner hears anything):** held dropout ≤ 3% (from 17.4%), V/UV error ≤ 7%,
jumps > 25 Hz ≤ 3%, span ratio within 0.7–1.5 of measured, UTMOS ≥ v0.4.1, CER (when run) not worse than v0.4.1 by
> 0.5 points. Fail → report numbers, try once with `l_uv` weight 2.0 and unvoiced-F0 weight 0.0, then stop and report.
**Gate 1 (owner):** page `runs/listen/sota_p1.html`: 12 sentences × {v0.4.1, P1}, blind, loudness-matched,
"pick the cleaner one" + "watery" tick. Pass = P1 picked ≥ 8/12.

### Phase 2 — decoder fine-tune on the text model's own outputs (GTA)
1. `scripts/make_gta.py`: for every row, run the Phase-1 text model in eval mode with **measured durations**
   (`aln = alignment(r['pred_dur'])`) and store `gta_F0` (Hz with uv, via `f0_hz`), `gta_N`, `gta_asr`
   (`asr_tok @ aln`) into a copy of the shard (`data/teacher_gta/shard_0000.pt`). Audio stays aligned because durations are measured.
2. `ms_train.py dec --gta-p 0.7`: each batch item uses GTA inputs with prob 0.7, measured inputs otherwise.
   For GTA items: compute `c0` from the **input** F0 (gta_F0); set harm and mod loss weights to **0**
   (those losses lock to the input pitch, which no longer equals the target audio's pitch). Keep mel + STFT.
3. Init from the Yati v0.4.1 decoder, lr 1e-4 (cosine), 3k steps, bs 16, seg 64, save every 1k.
4. Export with `--uv-text` + FIX H + webgpu patch → `sota_eval.py` → page `runs/listen/sota_p2.html` (P1 vs P2).

**Gate 2:** UTMOS ≥ P1 + 0.1 or owner picks P2 ≥ 8/12. If neither, keep P1's decoder and still go to Phase 3
(use GTA inputs there anyway — it costs nothing).

### Phase 3 — adversarial stage done right
1. **Profile first** (k11 ran at 1.57 s/step, ~10× slower than plain steps). On Kaggle T4: `torch.cuda.amp.autocast`
   (fp16, GradScaler) for G and D; `--seg 32` (0.8 s crops) for GAN batches; bs 16. Print s/step at step 50.
   Target ≤ 0.5 s/step. If > 0.8 s/step, stop, profile (`torch.profiler`, 20 steps) and report before burning quota.
2. Recipe (from the Phase-2 decoder):
   - **D warm-up 500 steps**: update D only (G frozen, `no_grad`). This fixes k11's "untrained discriminator".
   - Then joint 8k steps: G lr 1e-4, D lr 2e-4, AdamW betas (0.8, 0.99); loss as in `ms_train.py --gan`
     (`mel_weight·(lm+ls) + adv + 2·fm + 2·harm + 5·mod` on measured-input items; harm/mod = 0 on GTA items).
   - Inputs: GTA prob 0.7, as Phase 2. Save every 1k. Render the 12 gate sentences at 2k / 4k / 8k.
   - Kernel `sm-21-gan`, `--max-minutes 100`. Estimated 1–1.5 GPU-h.
3. Pick the checkpoint by UTMOS on the 80-sentence set (CER must not rise > 0.5 points), export, webgpu patch.

**Gate 3 (owner):** 3-way page {v0.4.1, P2, P3}, 12 sentences, blind. Pass = P3 picked ≥ 8/12 and "watery" ≤ 2/12.
Pass → publish **Yati v0.5** (`Suarify/kecil-tts-yati-v0.5`, `--skip-space`), card with leaderboard numbers.
Fail on all three phases → report honestly; then (and only then) consider Route A (§4).

### Phase 4 — more data (for pronunciation and English words)
Current data ≈ 118 min (1,281 sentences); English-word CER ~18%. Strong small TTS models use 5–20 h.
1. Select 2,500 new sentences: greedy phoneme-bigram coverage from `data/text/corpus_train.tsv`, plus 600 code-switch
   (English loanwords in Malay frames, use the lexicon from C3), plus 300 numbers/dates/RM/percent/abbreviations.
2. Generation is the cost (12.7 s/clip measured on sm19). Before committing: a 50-clip timing kernel testing batched
   Chatterbox generation and fewer sampling steps; pick the fastest config whose CER on those 50 ≤ 2.5%.
   Report clips/hour and total GPU-h; owner approves the number.
3. Same pipeline as Yati's original data (Chatterbox → one VC to Bella timbre, `ms_rows --letters --max-last 1.0
   --max-tail 99`). **Bundle inputs/outputs as one `.pt` or `.npz`, never a folder or a zip of wavs** (the 948/1,281 bug).
   Assert the full row count; never lower an assert to make a run pass.
4. Retrain text (Phase-1 recipe, 15k steps) and decoder (Phase 2 + Phase 3 recipe from the v0.5 weights) on all rows.
**Gate 4:** CER median ≤ 3.5% and capped mean ≤ 6% on the test set, code-switch ≤ 10%, owner A/B vs v0.5 ≥ 8/12.

### Phase 5 — roll out to the other voices
For nicole (and fenrir): regenerate Path-B data **from Yati v0.5** (1,281 or more clips, generated locally on CPU),
one VC per clip (bundle as one file!), then: text side = Phase-1 recipe (or Yati text + log-F0 affine shift if the
owner preferred it in the in-flight test), decoder = GTA + 2k GAN steps from Yati v0.5's decoder.
Michael: pYIN `fmin=50` when rebuilding rows; same Phase 1–3 recipe from his v0.3.1 weights.

---

## 3. Final targets ("SOTA" for this project, measured, not felt)
- Size ≤ 8.6 MB int8 ONNX; browser speed no worse than v0.4.1.
- CER (Whisper large-v3, malay, capped): median ≤ 3.5%, capped mean ≤ 6%; code-switch ≤ 10%. (v0.4: 5.5% / 9.5%.
  Chatterbox teacher clip ≈ 1.9% median — the ceiling.)
- UTMOS mean within 0.3 of the Chatterbox teacher clips.
- Owner blind A/B vs v0.4.1: ≥ 9/12 preferred, "watery" ≤ 2/12.

## 4. Route A (only if Phases 1–3 all fail)
Fine-tune Kokoro-82M itself on Malay (kokoro-deutsch / Thorsten-Voice recipe, see `suaradee-michael/PLAN_QUALITY.md`
§5) and distil with Paradee's own trainers. 10–20 GPU-h and a real risk of the "foreign accent" the owner rejected at
Gate R. Do not start it without a fresh owner decision.

## 5. Do not
- Do not propose post-hoc F0 contour edits, EQ, phase knobs, or a bigger head again (all tried, all rejected).
- Do not judge a model by amplitude-invariant metrics alone; always loudness-match.
- Do not lower a data-count assert to make a run pass.
- Do not publish any voice before its owner gate.
