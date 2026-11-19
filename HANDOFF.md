# HANDOFF — Kecil-TTS (Malay TTS) — full progress, challenges, dates, decisions

Prepared 2026-11-19 for a reviewing agent. This is the authoritative handoff: read this first, then
`STATUS.md` (chronological log), `continue.plan*.md` (plans), and the scripts/pages referenced below.
Working directory: `C:\Users\Acer\Downloads\kokoro-malay-bella`.

---

## 1. What the project is

A tiny (originally 8.5 MB, 7.9M-parameter) Malay text-to-speech student distilled from **Chatterbox Multilingual**
speech, with a rule-based Malay front end. Architecture: Kokoro-style text side (durations/F0/loudness) +
Paradee-derived decoder/head (sine-bank harmonics + noise, 128 harmonics for the male voice, 64 for female),
exported as one ONNX for Python/Node/browser (WebGPU or WASM). Branding was renamed mid-project from
"Suaradee" to **Kecil-TTS**; pip package name stayed `suaradee`.

## 2. Current published state (as of 2026-11-19)

| Model | HF repo | Notes |
|---|---|---|
| Yati v0.4.1 | `Suarify/kecil-tts-yati-v0.4.1` | Female, 8.46 MB, **the cleaner voice**, FIX H export, default in Space |
| Michael v0.3.1 | `Suarify/kecil-tts-michael-v0.3.1` | Male, 8.46 MB, timing-clamp removed (22% faster), faithful synthesis |

Space: `Suarify/kecil-tts-web` (renamed from `suaradee-web`), URL
`https://suarify-kecil-tts-web.static.hf.space` — Yati v0.4.1 / Michael v0.3.1 selector, EN/BM, dark mode,
keyword cloud, benchmark card, 13px intro. **Nicole/Fenrir are NOT published.**

**Rule from owner:** every GPU step needs explicit yes. Always publish with `--skip-space` (Space is owned by
another agent). Scoring: Whisper large-v3, `language='malay'`, CER after `normalize()`, capped 1.0/sentence,
report median AND capped mean; append STATUS after steps; commit after steps.

## 3. Kaggle / environment facts (critical)

- Kaggle account: **`suarifymalaysia`** (token `~/.kaggle/access_token`). Older kernels are under `seanlon`
  (403 for this token) — do not use those.
- GPU quota resets **2026-11-21**; ~13 h remained on 2026-11-18 (17 h/30 used).
- Python on Kaggle is **3.13.16**; the default image's torch/torchaudio pair is ABI-broken
  (`libcudart.so.12` missing, `libtorchaudio.so: undefined symbol`). Fixes that worked, in order of
  success: install chatterbox-tts → remove torchvision/torchao/torchtext → transformers==4.46.3 →
  **install matched torch+torchaudio 2.11.0 from `https://download.pytorch.org/whl/cu130` LAST**
  (so no later pip step restores the broken pair). Perth watermarking can be stubbed, but the matched pair
  is what finally worked.
- Kaggle API traps: dataset created seconds before a kernel push is not mounted → re-push the kernel.
  `dataset_list_files` is paginated (default 20). Queued script kernels can go stale — pushing a **new kernel
  id** cancels the stale one (status becomes CANCEL_ACKNOWLEDGED). GPU quota/status API may report RUNNING
  while a job is actually generating; verify with output files/logs. 1,283 separate file uploads hang —
  bundle into one zip (Kaggle auto-expands it).
- Local: `.venv\Scripts\python.exe` with `PYTHONUTF8=1`; misaki stubs in `.stubs/`; local machine has **no GPU**
  (~1.9× realtime CPU forward). Local page server on port 8931 keeps dying on harness restarts — restart with
  `.venv\Scripts\python.exe -m http.server 8931 --directory runs\local_v041`.

## 4. Chronological progress and challenges

### 2026-11-14 — Teacher search, Gate A/R failed
- Chatterbox voice-converted clips were consistently preferred over Kokoro/Dora blends; Dora-based voices
  "not so accurate" on code-switch/numbers. Owner: "the Chatterbox reference sounds Malay, most Kokoro options
  do not". Decision: abandon Kokoro/Dora timbre; train the student directly on Chatterbox voice-converted audio
  with measured timings/pitch/loudness. Fixed a normaliser bug where "RM87.20" lost digits (commit 60b83c5).
- Transplant (Chatterbox word timings/pitch) lost to plain Chatterbox (Gate R failed).

### 2026-11-15 — v0.1/v0.2 shipped; damp/phaser diagnosed
- k09 (686 clips, 12k steps): test CER 13.9%, 8.46 MB. Owner: "phasey/wet/damp". Measured: student energy
  >4 kHz 0.034 vs Chatterbox target 0.055; flatness 0.69 vs 0.62. int8 = fp32 (not quantisation).
- k10 (early stopping + jitter) WORSE (median CER 22.4%): select text checkpoints on `l_dur + l_asr`, not sum.
  Always report median + capped mean (Whisper hallucinations inflate means).
- k11 GAN stage: no real gain (15k steps needed ≈ 6.5 GPU-h). Abandoned.
- Alignment diagnosis: within-word phoneme timings off by median 27 ms / mean 42 ms vs letter-level forced
  alignment; 53% off by >1 frame. Added `--letters` to ms_rows: 6,050/6,318 words timed from letters.
- **v0.2 shipped** (letter timings): test median CER 4.8%, capped mean 6.5%, 70% ≤8%. Owner accepted ("Yes,
  acceptable"). Deployed under `Suarify/`; v0.1/v0.2 model repos + static Space (Gradio not possible on free HF).
- Damp diagnosis: owner ticks blamed **predicted F0** (version 3 from text damp 4/5; decoder with measured
  inputs 1/5; Chatterbox/Kokoro 0). f0_fix contour surgery (gap-fill/smooth/span-cap) **rejected 8/8** — the
  shipped contour always won. Conclusion: no-training fix for the damp NOT FOUND; needs training.

### 2026-11-15/16 — v0.3/v0.4 (Yati), packages A–C
- C3 lexicon grown; C4 cleaning: dropped unreadable clips, added 767 new clips (69 min), double-seed selection;
  1,281 aligned sentences (~118 min). Decoder 12k steps, text 10k. **Yati v0.4** overall median/capped CER
  5.5%/9.5% (v0.2 7.5%/10.8%); held 4.8%/8.5%. Published through `Suarify/kecil-tts-yati-v0.4`.
- Package A: damp located in the **predicted F0/N** again; contour surgery rejected again. Package B: streaming
  (Python/JS/Node parity pass), card, keyword cloud. Renamed all HF repos `suaradee-v*` → `kecil-tts-yati-v*`
  (old URLs redirect 200); Space renamed to `kecil-tts-web`.

### 2026-11-17 — FIX H breakthrough (Yati), Michael, bigger-student rejection
- **Root cause of muffled/watery (decoder)**: Chatterbox VC removes ~19 dB at 8–11 kHz; student faithfully
  matched its dark target. Fix at export level: **FIX H** = (1) gap-mask noise out of harmonic bins into the gaps
  in voiced frames >800 Hz, energy preserved, (2) gate 3 ms "burst" noise path to unvoiced frames (+hangover),
  (3) learned phase untouched. Owner verdict: "FIX H is the best one altho abit watery at the back". Baked into
  Yati v0.4.1 ONNX, no retraining. Metrics: periodicity 0.750 / flatness 0.595 / >4k 0.032 (targets 0.699/0.621/0.080).
- Michael retrain: his checkpoints were on the `suarifymalaysia` Kaggle account. Rebuilt from ONNX first
  (int8 dequant, weight corr 0.944), gate-order bug ONNX iofc vs torch ifgo fixed. sm-07 delta (+4k dec/+4.5k
  text), exported from true checkpoints → **Michael v0.3.0** with clamp 3–8, FIX H, air shelf, benchmarks in card.
- **Slowness root cause**: v0.3.0's clamp 3–8 frames (75–200 ms) made speech 22% slow; removed in **v0.3.1**
  (4.55 s vs 5.85 s on the III sentence), faithful synthesis, published + Space updated.
- Bigger Michael student (head 320, 6 blocks, 11.96 MB): trained 16k steps, **no gain** (teacher distance
  3.39 vs 3.39, held loss 2.515 vs 2.497), 3–4× slower. Rejected.
- Phase advance experiment: destroys periodicity (0.673 → 0.268/+ or 0.450/−). Learned phase is good.
- Student-vs-teacher log-mel distance: Yati 3.29, Michael 3.84 (teacher variants differ by only 0.84) → the
  residual murk is the 8M parametric head, not the data pipeline or export.
- Air/noise/clarity knobs: owner preferred faithful over processed for Michael; air6/noise variants rejected.

### 2026-11-17/18 — New voices: nicole/fenrir, the long environment battle
- Owner picked `af_nicole` and `am_fenrir` from 8 Kokoro candidate samples (`runs/listen/voices.html`).
- Plan: one-VC data (V Bella → target) instead of Michael's double conversion. Raw clips were missing locally
  (654/1,281) and `sm-15-raw` dataset 403 → regenerate 1,281 raws with Chatterbox (one fixed config).
- sm-12 kernel failed ~10 times on Kaggle env: dataset mount timing; torchaudio libcudart; torchao torch._dynamo;
  libtorchaudio ABI; BYOD docker image ignored; stale QUEUED sessions. Final working recipe = matched torch pair
  installed last (see §3). A single CPU-local torch test (`nvidia-smi` absent) confirmed no local GPU.
- **Gate 1 passed** (2026-11-18): sm19/sm20 were NOT hung — they were generating (272 min!) and converting
  (34 min); `DONE raws 1281 nicole 1281 fails 0`. Generation cost was 4.5 h not the estimated 55 min.
- **Gate 2 passed**: owner listened to 15 converted teacher clips — clean enough.
- Small probe (sm13a/14a/14b GPU queue dead → local CPU 500-step probe on 624 rows): owner said **very watery**.
- Critical correction from owner: "Yati became quite ok after changes along the way" — the tiny probe was
  rendered with the RAW training head, not FIX H. Re-rendered same checkpoint with FIX H → owner: "too soft".
  Loudness measurement revealed the real issue: **student RMS −36 dB vs teacher −17 dB (19 dB quieter)** —
  my amplitude-invariant metrics were blind to it. After loudness matching, FIX H + air6 was "ok but lacks clarity".

### 2026-11-18 — Path B and data-loss discovery
- **Path B**: synthesize with Yati's clean 12k student → one ChatterboxVC pass to nicole. 8 clips, 4.8 min CPU,
  0 fails; page `runs/listen/pathB.html`. Owner: **"path b sound good"**.
- Data-loss discovery: `ms_rows` defaults `--max-last 1.0 --max-tail 0.8` discarded 657/1,281 clips for
  trailing junk; the old working kernels used `--max-tail 99`. Rebuilding locally with correct flags gave
  **1,252/1,281 rows** (vs 624). This was the real reason the first probe was undertrained.
- Owner approved ~1.5–2 h GPU for **Path-B distillation**: 1,281 Yati clips → one VC each → small probe →
  full 12k run. Generated 1,281 Yati clips locally in 3 min (`scripts/pathb_gen_all.py`).
- sm15c dataset upload: 1,283 files hung; bundled to zip; Kaggle auto-expanded; only 948 wavs visible per
  kernel mount → lowered the assert. VC pass: **948 clips, 0 fails, 25 min** (`DONE 948 fails 0`).
- sm15d small probe: 2k dec + 1.5k text steps, rows with `--max-tail 99`, checkpoints saved. Exported with
  FIX H + air6, loudness-matched → `runs/listen/probe_small.html`. Owner: **level OK, some words a bit phasey**.
- Phase stabilization (export knob `--phase-smooth 5`, `--phase-shrink 0.5`): owner verdict — **both WORSE
  (more watery/phasey) than current**. Learned phase is useful; residual phasey = undertrained head.

### 2026-11-19 — Full 12k Path-B run (ACTIVE)
- Owner explicitly approved full run. Kernel `suarifymalaysia/sm-15e-full`: same Path-B data (948 converted
  clips via sm15c kernel source), decoder **12,000 steps** + text **10,000 steps** from Yati init, ~40 min GPU.
  Outputs `dec_nicoleb_full.pt` / `text_nicoleb_full.pt`. **RUNNING / watcher active.**
- After it finishes: export FIX H + air6 (proven recipe), loudness-match to Path-B teacher, final listening
  page → publish as an 8 MB voice **only if clean**.

## 5. Key technical decisions and lessons (do not re-litigate without evidence)

1. **FIX H is the export recipe that made Yati acceptable**; bake into graph, no retraining.
2. **Loudness matching is mandatory for listening tests** — RMS/peak must be equal; periodicity/flatness are
   scale-invariant and hide "too soft" (student was 19 dB quieter).
3. **Text checkpoints select on `l_dur + l_asr`**, never summed held loss (f0/N overfits; sum picks step 250).
4. **ms_rows needs `--max-tail 99 --max-last 1.0`** for these clips; defaults discard half the data.
5. **No-training fixes for damp/phaser were exhaustively tried and rejected**: EQ, flatness knobs, F0 contour
   surgery, phase advance, phase smooth/shrink, bigger head. Do not propose them again without new evidence.
6. **Dataset quality matters**: single VC (Path B) is cleaner input than double VC; generating from Yati's
   student avoids trailing-junk entirely.
7. Every failed Kaggle attempt cost ~4 min of setup (no generation) — keep checkpoints every 100 items.
8. Owner rejects voice blends; B2 comparison kernel stays paused. Do not run publish_hf.py without `--skip-space`.

## 6. Active files / pages

- Exporter: `scripts/export_fixh.py` (FIX H, `--gap-keep`, `--hang`, `--burst`, `--no-gap-mask`, `--air-db`,
  `--gn-smooth`, `--clamp-*`, `--phase-smooth`, `--phase-shrink`). Post-export **mandatory**:
  `scripts/webgpu_patch.py IN.onnx OUT.onnx` (0 float64 ops).
- Data: `data/teacher/shard_0000.pt` (1,281 Yati rows); `runs/kaggle_sm15c/nicoleB.pt` (948 Path-B clips);
  `runs/kaggle_sm15d/*_probe.pt` (2k probe ckpts); `runs/listen/vc15.html`, `pathB.html`, `probe_small.html`,
  `probe_phase.html` (local listening pages; `runs/local_v041/` is the served mirror).
- Kaggle kernels: `kaggle/sm15c_nicole_vc`, `kaggle/sm15d_probe`, `kaggle/sm15e_full` (active).
- Heartbeat: `STATUS.md` appended after every step; all steps committed.

## 7. Immediately pending

1. **sm-15e-full result** → pull checkpoints → export FIX H + air6 + webgpu patch → loudness-matched page → owner gate.
2. If clean → publish `Suarify/kecil-tts-nicole-v0.1` (name TBD by owner) + Space dropdown entry.
3. Fenrir: reuse the 948 Path-B raws → one VC to `am_fenrir` → same probe/full gates.
4. Older deferred items: v0.5 code-switch attack, GitHub push (no suarify.my SSH profile), B3/B4 card work,
   Michael bigger-variant card (rejected), pip package rename `suaradee` → `kecil-tts` (owner undecided).
