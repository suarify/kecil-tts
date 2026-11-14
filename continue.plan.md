# kokoro-malay-bella — continuation plan (v2, low-compute route)

Written 2026-11-14 for an agent that has **no memory of the earlier sessions**. Read sections 0–5
before touching anything. `plan.md` is the original plan; this file **replaces its Phases 4–12**.
Phases 0–3 of `plan.md` are done and their results are in section 2.

---

## 0. One page

**Goal.** A Malay text-to-speech voice ("Bella speaking Malay") that ships as:

1. a **tiny single-file ONNX model, about 8–9 MB** (the size class of
   [Paradee-8M](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0)),
2. a **Python package** (onnxruntime + numpy only, no torch),
3. a **browser / Node version** that loads through **transformers.js** (with plain
   onnxruntime-web as the fallback), using the Malay G2P that already exists in JavaScript.

**What changed.** The original plan was: generate ~11 h of Malay audio with Chatterbox
(~18 GPU-hours), fine-tune the 82M Kokoro on it (~15–25 GPU-hours), then export a ~90 MB model.
The owner does not want that much compute or checking, and wants a model ten times smaller.

**New route — distil, don't fine-tune.** Paradee shows that Kokoro-82M can be distilled into an
8M-parameter single-voice model by training a small student against the **frozen** Kokoro teacher.
A Brazilian-Portuguese port ([eraldohug/Paradee-8M-PtBr](https://huggingface.co/eraldohug/Paradee-8M-PtBr))
did it with 12,000 sentences, 8,000 text-side steps and 5,000 decoder steps. So:

| Step | What | GPU cost (Kaggle T4, estimate) |
|---|---|---|
| A | Find the best **zero-training Kokoro teacher** for Malay: Bella blended with a non-English Kokoro voice, plus G2P variants. Owner picks by ear. | ~0.7 h |
| M1 | Ship that teacher as the "big" model (Kokoro-82M + new voice file + Malay G2P). **No training at all.** | 0 |
| B | Kokoro teacher reads 12,200 Malay sentences and saves its internals | ~1 h |
| C | Train the 8M student: text side, then decoder | ~2.5–4 h |
| D–E | Export ONNX (fp32 + int8), Python + JS runtimes, tests | CPU only |
| F | Owner's final listen, release | 0 |

**Total: about 5–6 GPU-hours, hard cap 8.** The old route had 35–45 left.

**The owner is needed exactly twice:** Gate A (about 10 minutes of listening) and Gate F
(about 5 minutes). Everything else is automatic and gated by numbers. Do not ask anything else.

**The one real risk.** The student can only be as good as its teacher, and the teacher is Kokoro
reading Malay phonemes with no Malay training. Plain Bella doing this was **rejected by the owner
(0 of 60 picks)** because it sounds English. Step A exists to fix that for free by changing the
voice vector and the phoneme conventions. If Step A fails its gate, section 12 has the fallback.
Do not start Step B before Gate A passes.

---

## 1. Definition of done

- [ ] `export/ms_bella_8m/onnx/model_quantized.onnx` ≤ 10 MB, `onnx/model.onnx` (fp32) ≤ 40 MB.
- [ ] Round-trip CER (Whisper large-v3, language `malay`) of the int8 model on `data/text/test_200.tsv`
      ≤ teacher CER + 1.5 points, and ≤ 6% absolute.
- [ ] ≥ 15× real time on one CPU thread (onnxruntime, int8).
- [ ] `python -m kokoro_malay "Selamat pagi, boleh saya bantu?" -o out.wav` works with only
      `onnxruntime`, `numpy`, `soundfile` installed.
- [ ] Node test: transformers.js loads the model and produces the same waveform as Python for the
      same `input_ids` (correlation ≥ 0.99, see E3). A static web page speaks typed Malay text.
- [ ] Python G2P and JS G2P still produce identical phonemes (`g2p/tests/parity.mjs`: all rows equal).
- [ ] Owner accepted at Gate F.
- [ ] `STATUS.md` records every gate result. Everything committed.

---

## 2. What exists and what was measured

Project root: `C:\Users\Acer\Downloads\kokoro-malay-bella` (git repo, branch `main`,
head `00f7b27`). Windows 11, Git Bash + PowerShell. Kaggle CLI is logged in as **`seanlon`**.

### 2.1 Repo map (only what matters now)

| Path | What it is |
|---|---|
| `g2p/ms_norm.py`, `g2p/ms_norm.js` | Malay text normaliser (numbers, RM, dates, times, phone/OTP digits, abbreviations). Python and JS are line-for-line ports. |
| `g2p/ms_g2p.py`, `g2p/ms_g2p.js` | Malay G2P → Kokoro phoneme strings. `G2P(stress=True, r='r', diph='glide')`. `g(text)` returns phonemes, `tokens()` returns ids with `0` at both ends, `chunk()` splits at ≤ 510. |
| `g2p/lexicon_e.json` | 24,044 words: which `e` is pepet `ə` and which is taling `e` |
| `g2p/lexicon_en.json` | ~430 English words Malaysians mix in, with misaki phonemes |
| `g2p/lexicon_exceptions.json` | direct phoneme strings (malaysia, kuala, celsius, …) |
| `g2p/teacher_respell.json`, `scripts/build_teacher_respell.py` | English → Malay-style spelling. **Only for Chatterbox. Not needed on the new route.** |
| `g2p/vocab.json`, `kokoro_config.json` | Kokoro's 178-token vocab |
| `g2p/tests/test_gate1.py`, `g2p/tests/parity.mjs`, `g2p/tests/golden.jsonl` | 5 pytest tests; Node parity check over 2,000 golden rows |
| `scripts/build_corpus.py` | builds `data/text/corpus_train.tsv` (7,500), `test_200.tsv`, `val_100.tsv`, `OOD_texts.txt` |
| `scripts/listen_pilot.py` | builds a blind listening page — copy its pattern for Gate A |
| `eval/test_40.tsv`, `eval/e_test.tsv`, `eval/pilot_150.tsv`, `eval/en_ref.tsv` | small test sets |
| `kaggle/ds_g2p/` | staging folder of Kaggle dataset `seanlon/kokoro-malay-g2p` |
| `kaggle/k01_baseline`, `k02_teacher`, `k03_pilot` | the three kernels already run (read them for working patterns) |
| `runs/` (**git-ignored, local only**) | downloaded kernel outputs: `runs/k01` (Kokoro baseline clips, `refs/ref_en.wav`, `refs/bella_en/`), `runs/k02/teacher` (Chatterbox T1/T2, untrimmed), `runs/k03/pilot/{raw,T2,T2v}` + `pilot_scores.csv` |

Kaggle assets: dataset `seanlon/kokoro-malay-g2p`, dataset `seanlon/kmb-pilot-raw`, kernels
`seanlon/kmb-01-baseline`, `kmb-02-teacher`, `kmb-03-pilot`.

### 2.2 Results so far

| Experiment | Result |
|---|---|
| Gate 1 (G2P) | 5/5 tests pass, JS = Python on 2,000 rows, zero out-of-vocab symbols |
| Stock Kokoro + `af_bella` reading Malay, 8 G2P variants, 40 sentences | best = **no stress, `r`, glide diphthongs** (`s0_r_glide`): CER **4.1%**, median 0%. `ɾ` and stress marks were worse **with an English voice**. |
| Same, 20 "e-test" words | Whisper recognised 14/20 |
| Chatterbox Multilingual teacher, 60 sentences | T2 (Malay reference clip, `cfg_weight=0.3`): CER 6.8%, similarity to Bella 0.49, muffled (roll-off 3.9 kHz), 2 of 3 clips had babble at the end |
| **Owner's blind pick, 60 sentences** | **Kokoro baseline 0 · Chatterbox-EN-ref 8 · Chatterbox-MS-ref 52.** Comment: Malay character is clearly there with Chatterbox. Problems: English loanwords read with Malay letters, junk at the end. |
| Pilot 150 with fixes (respelling + trim at last word) | CER 20.9% → **1.1%**; 148/150 pass. Voice-converted copy (toward real Bella): similarity 0.48 → 0.58, CER 2.8% |
| Owner's pick on pilot (34 rated) | voice-converted 22 · trimmed-only 12; 6 clips flagged "mispronounced" (all English loanwords) |
| Speed | Kokoro on T4 ≈ 23× real time. Chatterbox ≈ 0.7× real time (1.45 s GPU per second of audio). |

Similarity scale (ECAPA cosine to a true-Bella centroid): Bella vs Bella 0.86, Sarah vs Bella 0.17.

### 2.3 What the evidence says

1. **The owner judges "does it sound Malay" far above "is it exactly Bella".** A voice at
   similarity 0.49 beat a voice at 0.80 by 52 to 0. So the Bella share of the blend in Step A may
   go well below 50% if that is what sounds Malay. A pleasant female voice that is recognisably
   related to Bella is enough.
2. **Kokoro already says Malay words correctly** (4.1% CER, fully clear audio, no babble, no
   watermark, 30× cheaper to generate). Its problem is accent, not intelligibility.
3. **Chatterbox's strengths are accent and rhythm; its weaknesses are reliability and cost.**
   On the new route it is used only as a **reference for choosing the teacher**, never as
   training audio. The 150 trimmed pilot clips the owner approved are that reference.
4. CER alone does not measure accent. Step A therefore adds an accent score and ends with ears.

---

## 3. Background facts the plan relies on

Verified on 2026-11-14 by reading the sources (repo commit `2de547e` of
[sahilmahendrakar/paradee](https://github.com/sahilmahendrakar/paradee), the fork
[astrowar/paradee_pt](https://github.com/astrowar/paradee_pt), the model cards and the paper).

**Kokoro**
- Voice file `voices/<name>.pt`: tensor `[510, 1, 256]`. Row index = `len(phonemes) - 1`.
  **Columns `0:128` go to the decoder (timbre and articulation). Columns `128:256` go to the
  prosody predictor (durations, pitch, loudness).** The two halves can be blended independently.
- Voice vectors blend linearly. Non-English voices exist: Spanish `ef_dora`, Italian `if_sara`,
  Brazilian Portuguese `pf_dora`, Hindi `hf_alpha` `hf_beta`, Japanese `jf_alpha`, French
  `ff_siwis`, Mandarin `zf_xiaoxiao` (list the repo's `voices/` folder to confirm names).
- `KModel` runs from a phoneme string. It does not need misaki or eSpeak.

**Paradee**
- Two halves trained separately against the frozen teacher, then joined with no joint training:
  - **Text side, 4.23M params** (`TextStudent("s+mlp")`): phoneme ids → durations, F0, energy
    `N`, and 512-channel phoneme features. Trained with L1 losses, teacher-forced durations.
    Published recipe: 8,000 steps, batch 32, lr 5e-4.
  - **Decoder, 3.65M params** (`VDecoder(head="v2", dim=160, blocks=4)`): teacher features + F0
    + N → waveform. Spectral losses, then adversarial. Runs at 80 frames/s, never upsamples.
- Teacher corpus = shards of 500 sentences, each row a dict:
  `ids, dur, pred_dur, d, t_en, F0, N, audio(int16), text, ps`. The first 200 rows of shard 0
  are the held-out set in every trainer.
- **The teacher is driven by a phoneme string.** `gen_teacher.py` only calls
  `pipe.g2p(text)` to get it. Replacing that with our Malay phonemes is the whole adaptation.
- The student keeps Kokoro's 178-token vocab (`n_token=178`), so Malay needs no new symbols.
- Released PyTorch weights: `pytorch/text_side.pt` and `pytorch/decoder.pt`, each
  `{"model": state_dict}`. v1.1 has one voice row; v1.2 has four; **v1.3 has eight**
  (`af_heart, bf_isabella, am_puck, bm_fable, hf_alpha, hm_omega, ff_siwis, em_alex`) and has
  been trained on Hindi, French and Spanish phonemes.
- The pt-BR fork's working recipe on CUDA: text side from scratch 8,000 steps (final loss 0.2835:
  dur 0.040, f0 0.044, n 0.139, asr 0.059); decoder `--init-from` the released decoder, **5,000
  spectral-loss steps, no adversarial stage** (DTW log-mel L1 0.5754). The fork did not re-render
  audio on CPU; teacher features and audio from the same CUDA run are self-consistent.
- Export (`export_paradee_v1_1.py DEC --text TEXT --out DIR`): one graph, opset 17,
  inputs `input_ids` int64 `[1,T]` (0 at both ends, T ≤ 512) and `speed` float32 `[1]`,
  output `waveform` float32 `[1,S]` at 24 kHz. It writes fp32 (~35 MB) and a weight-only int8
  copy (~8.5 MB). It reads the first 4 rows of `data/teacher/shard_0000.pt` for its self-check.
- Browser: the existing demo ([KingNish/paradee-web](https://huggingface.co/spaces/KingNish/paradee-web))
  runs the int8 file with **onnxruntime-web on WASM** and feeds `{input_ids, speed}`.
  WebGPU is not needed at this size; the community WebGPU build had to re-quantise the model
  ([danyalmalik/Paradee-8M-ONNX-WebGPU](https://huggingface.co/danyalmalik/Paradee-8M-ONNX-WebGPU)).
- Licences: Kokoro-82M and Paradee are Apache-2.0.

**Not verified — check while executing**
- Seconds per training step on a T4 (the estimates in section 0 assume 0.2–0.4 s for the text
  side and 0.3–0.6 s for the decoder). Step C measures first and scales the step counts.
- Fine-tuning the text side **from v1.3 weights** instead of from scratch is how the author made
  v1.2 and v1.3, but the public trainer has no flag for it. C1 adds one; from-scratch is the fallback.
- That `StyleTextToSpeech2Model` in transformers.js runs a graph without a `style` input (E3 tests it).

---

## 4. Rules for the executing agent

1. **Gates are hard.** Do not start a phase before the previous gate passes. If a gate fails,
   follow its "if it fails" line. Do not lower a threshold to pass.
2. **Compute cap: 8 GPU-hours total** on this plan. Log GPU time per kernel in `STATUS.md`.
   If the cap would be exceeded, stop and report.
3. **Never train on Chatterbox audio** and never fine-tune Kokoro-82M on this route.
4. **Do not edit** `g2p/tests/golden.jsonl` by hand. Regenerate it with `scripts/make_golden.py`
   only when the G2P default behaviour changes on purpose (step A6), then re-run parity.
5. Ask the owner only at Gate A and Gate F. Report in plain language: what was done, the numbers,
   what is next.
6. After every phase: update `STATUS.md` (create it in the repo root: date, phase, kernel version,
   GPU minutes, gate numbers, decision), then commit. Commit messages end with
   `Co-Authored-By:` for the model doing the work.
7. Heavy work runs on Kaggle. Local work is CPU only (G2P, corpus building, page building, export
   checks, runtimes).

---

## 5. Environment and known traps (all of these already cost a failed run)

| Trap | Fix |
|---|---|
| Kaggle image is **Python 3.13**; `kokoro==0.9.4` declares `<3.13` | `pip install -q --no-deps --ignore-requires-python "kokoro==0.9.4"` then `pip install -q loguru`. |
| `import kokoro` imports misaki (not installed) | Create a fake package before anything imports kokoro, and put it on `PYTHONPATH` so **subprocesses** see it too: `stubs/misaki/__init__.py`, `stubs/misaki/en.py`, `stubs/misaki/espeak.py`, each containing only `def __getattr__(name): return type(name, (), {})`. Then `os.environ["PYTHONPATH"] = "/kaggle/working/stubs:" + os.environ.get("PYTHONPATH","")` and `sys.path.insert(0, "/kaggle/working/stubs")`. |
| Dataset / kernel-output mount paths vary | Never hard-code. `glob.glob('/kaggle/input/**/<known file>', recursive=True)[0]`. |
| Whisper large-v3 ran out of the T4's 15 GB | `batch_size=4`, `generate_kwargs={'language':'malay','task':'transcribe','max_new_tokens':200}`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. With `return_timestamps='word'` use `batch_size=1`. Delete other models and `gc.collect(); torch.cuda.empty_cache()` first. |
| Kaggle CLI on Windows crashes on IPA characters | `export PYTHONUTF8=1` before every `kaggle` command. |
| Two `kaggle kernels output` runs into one folder corrupt the log | One download at a time. Use `--file-pattern` to fetch only what is needed. |
| `kaggle datasets create -p kaggle/ds_x` fails | `cd kaggle && kaggle datasets create -p ds_x` (add `--dir-mode zip` when the folder has subfolders). New version: `kaggle datasets version -p ds_x -m "msg"`. Wait for `kaggle datasets status <slug>` = `ready` before pushing a kernel. |
| Bash heredocs mangle backslashes and quotes | Write a script file and run it. |
| Installing `chatterbox-tts` downgrades torch and breaks torchvision | Not needed on this route. (If ever needed: `pip uninstall -y torchvision` after install.) |
| A kernel's output is kept only from `/kaggle/working` (≤ 20 GB, keep ≤ ~400 files) | Write shards and checkpoints there. Never write thousands of WAVs; pack clips into a few `.pt` or `.zip` files. |
| Kernel killed at 12 h, or weekly quota (about 30 GPU-h) runs out | Every trainer call gets a step budget computed from measured s/step (C0). Split work across kernels as in section 8. |
| Waiting on a kernel | Poll `kaggle kernels status <slug>` every 30–60 s in a background shell; do not block the session. |
| Kernel metadata | Copy `kaggle/k03_pilot/kernel-metadata.json`. Script kernel, `enable_gpu: true`, `enable_internet: true`, `machine_shape: "NvidiaTeslaT4"`, `is_private: true`. `kernel_sources` attaches an earlier kernel's output. |

Local Python has no GPU. Local Node works (`g2p/package.json` is `{"type":"module"}`).

---

## 6. Phase A — find the teacher (≈ 0.7 GPU-h, then the owner listens)

**Aim.** A Kokoro voice vector + G2P options that read Malay in a way the owner accepts, with no
training. Output: `voices/ms_bella.pt` and a frozen G2P option set.

### A0. Prepare references (local, CPU)

1. Build a reference set from the pilot: rows of `runs/k03/pilot_scores.csv` with `cer ≤ 0.02`
   and `category == 'general'`, shortest-to-longest spread, pick **24**; plus **8** code-switch
   rows with `cer ≤ 0.05`. Save `eval/ref_32.tsv` (`id, category, text, norm`).
2. Stage a Kaggle dataset `seanlon/kmb-ref` from `kaggle/ds_ref/`: the 32 `runs/k03/pilot/T2/<id>.wav`
   and `T2v/<id>.wav` files (keep the two folder names, use `--dir-mode zip`), `ref_32.tsv`,
   and `runs/k01/refs/bella_en/*.wav` (12 true-Bella clips for the similarity centroid).
3. Add a `final_a` option to `g2p/ms_g2p.py` **only** (not JS yet), default unchanged:
   `G2P(..., final_a='a')`. With `final_a='ə'`: in `word()`, on the Malay path only (not
   `lex_x`, not `lex_en`, not acronyms), if the phoneme list has ≥ 2 vowels and ends in `a`,
   replace that last `a` with `ə` — unless the lowercased word is in a keep-list
   (`ya, data, drama, kamera, formula, agenda, media, visa, pizza, opera, asia, indonesia, astro,
   sofa, kuota, zebra, panda, lava`) or the token is capitalised and is not the first
   token of the sentence (names). This is the Johor-Riau standard Malaysian reading ("saya" →
   `sajə`); `final_a='a'` is the spelling ("baku") reading. Default stays `'a'`, so
   `pytest g2p/tests/test_gate1.py` must still pass. Run it.
4. Copy the updated `ms_g2p.py` to `kaggle/ds_g2p/` and version the dataset.

### A1. Kernel `kmb-04-teacher-search` (`kaggle/k04_teacher_search/`)

Inputs: datasets `seanlon/kokoro-malay-g2p`, `seanlon/kmb-ref`. Reuse the install, stub, model
loading, `synth()`, `lev()`, `clean()` code from `kaggle/k01_baseline/k01_baseline.py`.

**Blend.** `B` = `af_bella`, `D` = a donor, both `[510,1,256]`:

```python
def blend(B, D, wa, wp):          # wa, wp = Bella's share of the acoustic / prosody half
    return torch.cat([wa * B[..., :128] + (1 - wa) * D[..., :128],
                      wp * B[..., 128:] + (1 - wp) * D[..., 128:]], -1)
```

Donors: `ef_dora, if_sara, pf_dora, hf_alpha, hf_beta, jf_alpha, ff_siwis, zf_xiaoxiao`, and
`mix3` = mean of `ef_dora, if_sara, hf_alpha`. Skip any name missing from the repo.

**Scores per clip.**
- `cer`: Whisper large-v3 round trip against the normalised text (as in kernel 01).
- `acc` (accent distance, lower is better): Whisper large-v3 **encoder** output for the clip and
  for the owner-approved Chatterbox clip `T2/<id>.wav` of the same sentence; keep the first
  `ceil(seconds / 0.02)` frames of each, L2-normalise, DTW with cosine distance, divide by path
  length. (16 kHz input, features from the pipeline's feature extractor; encoder =
  `asr.model.get_encoder()`; run in fp16; DTW in numpy on CPU.)
- `sim`: ECAPA cosine to the true-Bella centroid (kernel 02 has the code; `run_opts={'device':'cuda:0'}`).
- `rate`: phonemes per second.

**Check the accent score before trusting it.** On the 32 reference sentences compute
`acc(B0)` for stock Bella (`wa = wp = 1`, G2P `s0_r_glide`) and `acc(T2v)` for the voice-converted
Chatterbox clip. Expect `acc(T2v) < acc(B0)` on ≥ 80% of sentences. Write the fraction to the
report. If it is below 80%, the score is not usable: rank by `cer` only and send **6** finalists
to the owner instead of 4.

**Stage 1 — voice grid** (G2P fixed at `stress=False, r='r', diph='glide', final_a='a'`), on 16 of
the 32 sentences (12 general + 4 code-switch): for each donor, `wa ∈ {0.7, 0.5, 0.3}` ×
`wp ∈ {0.5, 0.0}`, plus each pure donor (`wa = wp = 0`) and stock Bella. About 64 configs × 16
clips ≈ 1,000 clips (about 3 min to synthesise, about 20 min to score).
Rank by `acc` among configs with `cer ≤ cer(B0) + 0.02`. Keep the **top 6**, at most 2 per donor.

**Stage 2 — phoneme conventions**, for each of the 6: `r ∈ {r, ɾ}` × `stress ∈ {False, True}` ×
`final_a ∈ {a, ə}` on the same 16 sentences (48 configs). For the `final_a='ə'` variants the
Chatterbox reference stays valid for `acc` but compute `cer` as usual (Whisper writes "saya"
either way). Keep each blend's best convention by `acc` (ties → lower `cer`).

**Finalists.** Top 4 by `acc` (or 6, see above). Re-synthesise all 32 reference sentences plus
`eval/test_40.tsv` and `eval/e_test.tsv` for each finalist and score them.

**Outputs** (`/kaggle/working`): `search_scores.csv` (every config: donor, wa, wp, r, stress,
final_a, mean cer/acc/sim/rate), `finalists.json`, `voices/<finalist>.pt` (the blended
`[510,1,256]` tensors), `clips_finalists.zip` (WAVs: `<finalist>/<id>.wav`),
`search_report.json` (accent-score check, B0 numbers, runtime).

### A2. Listening page (local) — **Gate A**

Build `runs/listen/teacher.html` with a new `scripts/listen_teacher.py` modelled on
`scripts/listen_pilot.py`. 16 sentences (10 general, 3 code-switch, 3 from `test_40` numbers).
Per sentence, shuffled and unlabelled: each finalist, **stock Bella** (`B0`), and the
**Chatterbox voice-converted clip** where one exists (anchor: "this is the Malay you liked").
If the top finalist's best convention used `final_a='ə'`, also include the same blend with
`final_a='a'` (and vice versa) so the owner hears that one difference in isolation.
Controls per sentence: pick the best one; tick "not acceptable" on any clip.
A "Copy results" button like the pilot page.

Tell the owner, in the chat, to run:

```bash
python -m http.server 8765 --directory C:/Users/Acer/Downloads/kokoro-malay-bella/runs
```

then open `http://localhost:8765/listen/teacher.html`, rate at least 12 sentences, and paste the
copied results. (Starting that server from the agent side was blocked earlier; the owner runs it.)

**Gate A passes when** one Kokoro finalist has (a) the most picks among the Kokoro systems,
(b) "not acceptable" on ≤ 25% of its rated clips, and (c) `cer ≤ 6%` on `test_40`.
It does **not** have to beat the Chatterbox anchor.

**If it fails:** go to section 12. Do not continue to M1 or B.

### A3. Freeze

1. Copy the winner to `voices/ms_bella.pt`; also write `voices/ms_bella.bin` (raw little-endian
   float32, 510 × 256 = 522,240 bytes — the format kokoro-js voices use).
2. Record the winning options in `configs/teacher.json`:
   `{donor, wa, wp, stress, r, diph, final_a, cer_test40, acc, sim}`.
3. If the frozen options differ from the current defaults (`stress=True, r='r', diph='glide'`,
   no `final_a`): make them the **defaults** in `ms_g2p.py`, port `final_a` and the keep-list to
   `ms_g2p.js`, regenerate `golden.jsonl` with `scripts/make_golden.py`, and get
   `pytest g2p/tests/test_gate1.py` and `node g2p/tests/parity.mjs` green again.
   (`test_gate1.py` has an e-vowel list and normaliser cases that do not depend on these options;
   only the golden file changes.)
4. Commit. Update `STATUS.md`.

---

## 7. Milestone M1 — the big model, for free (local, CPU, ~1 hour of work)

The teacher is already a usable Malay Kokoro. Ship it now so the owner has something working
while the small model trains, and so there is a quality ceiling to compare against.

- `python/kokoro_malay_big.py`: `KModel` (PyTorch, `hexgrad/Kokoro-82M`) + `voices/ms_bella.pt`
  + `ms_norm` + `ms_g2p`. One function `tts(text) -> np.ndarray` at 24 kHz, sentence chunking
  with `chunk()`.
- `web/big.html` + `web/kokoro-malay-big.js`: transformers.js with the stock ONNX Kokoro
  (`onnx-community/Kokoro-82M-v1.0-ONNX`, `dtype: 'q8'`, about 92 MB). Use
  `StyleTextToSpeech2Model.from_pretrained` and `AutoTokenizer.from_pretrained` from
  `@huggingface/transformers`, phonemes from `g2p/ms_g2p.js`, and feed
  `{input_ids, style, speed}` where `style` is one 256-float row of `ms_bella.bin` as a
  `[1,256]` float32 tensor. Python indexes the row by `len(phonemes) - 1`; kokoro-js computes
  its own index from the token count. Read the kokoro-js source for the exact row index and
  tensor shapes before writing this; do not guess.
- Check: the same sentence through Python and through Node gives clips Whisper transcribes the
  same way. This is a smoke test, not a gate.

Skip M1 only if it takes more than half a day; it is not on the critical path.

---

## 8. Phase B — teacher corpus (≈ 1 GPU-h)

### B0. Text (local, CPU)

1. Parametrise `scripts/build_corpus.py`: `--target N` (default 7500) and read the G2P options
   from `configs/teacher.json` instead of the hard-coded `G2P(stress=False, r='r', diph='glide')`.
2. Run with `--target 12000`. The template categories saturate (about 1,900 conversational,
   370 code-switch, 230 questions); the existing top-up from the general pool fills the rest.
   That is fine. Expect about 19 hours of teacher audio (72.7 phonemes per sentence on average).
3. Write `data/text/teacher_manifest.tsv` = **`test_200.tsv` rows first (exactly 200, in order),
   then the 12,000 training rows, shuffled with a fixed seed.** Columns `id, category, text,
   norm, phonemes`. The first 200 rows become Paradee's held-out set automatically.
   `val_100.tsv` stays out of the manifest (used in Phase F).
4. Assert: every phoneme character is in `g2p/vocab.json`; 8 ≤ `len(phonemes)` ≤ 400 for every
   row (drop rows outside and refill). Upload as dataset `seanlon/kmb-teacher-text`
   (manifest + `voices/ms_bella.pt` + `configs/teacher.json`).

### B1. Vendor the training code (local)

`git clone https://github.com/sahilmahendrakar/paradee third_party/paradee` and check out commit
`2de547e`. Keep its `LICENSE`. Create `paradee_ms/` in the repo with **copies** of
`third_party/paradee/training/scripts/*.py` plus the changes below, and upload `paradee_ms/` as
dataset `seanlon/kmb-paradee-ms`. Read each script fully before patching it.

| File | Change |
|---|---|
| `common.py` | `load_teacher(device, voice=None, fuse=True)` builds the teacher with `from kokoro import KModel; model = KModel(repo_id="hexgrad/Kokoro-82M").to(device).eval()` and `pack = torch.load(os.environ["KMB_VOICE"], map_location=device)`. Return a tiny object with `.model = model` and `.load_voice = lambda *_: pack` in place of the pipeline, then `model`, then `items = []`. Nothing in the trainers uses `items`. Remove the `pipe.g2p` loop. |
| `gen_teacher_ms.py` (new, from `gen_teacher.py`) | Read `teacher_manifest.tsv`; phonemes come from the `phonemes` column (no G2P call). Voice = `torch.load(KMB_VOICE)`. Style row `ref[min(max(0, T - 3), ref.shape[0] - 1)]`. `torch.manual_seed(row_index)` before each sentence. Save the same dict keys **except `d`** (no trainer reads it; it is the largest item). Shards of 500, atomic write (`.tmp` then `os.replace`), skip shards that exist. Args `--device cuda --start-shard --end-shard`. Print audio hours and ×RT per shard. |
| `train_text.py` | Add `--init-from PATH` (load `{"model": …}` into the fresh model when no `last.pt` exists; fresh optimiser) and `--max-minutes M` (stop the loop cleanly, save `last.pt`, run the final `evaluate(save=True)`). Default `--device cuda`. The two `pipe.load_voice("af_heart")` calls keep working through the shim. |
| `train_vhead.py` | Add `--max-minutes M` the same way. Default `--device cuda`. `--init-from` already exists. |
| `slice_voice.py` (new) | `st = torch.load(src, map_location="cpu")["model"]; st["style"] = st["style"][row:row+1].clone(); torch.save({"model": st}, dst)` |
| `export_paradee_v1_1.py` | No change expected. It needs `data/teacher/shard_0000.pt`. |

Local check before uploading: `python -m py_compile paradee_ms/*.py`.

### B2. Kernel `kmb-05-teacher-corpus` (`kaggle/k05_teacher_corpus/`)

Inputs: `seanlon/kmb-teacher-text`, `seanlon/kmb-paradee-ms`.

1. Install kokoro as in section 5, write the misaki stubs, set `PYTHONPATH`, set `KMB_VOICE`.
2. Copy `paradee_ms/` to `/kaggle/working/train/scripts/`, `cd /kaggle/working/train`.
3. Smoke: first shard only, print ×RT. Expect ≥ 15× on a T4. If < 8×, something is wrong
   (model on CPU?) — stop.
4. Run all shards → `/kaggle/working/train/data/teacher/shard_0000.pt … shard_0024.pt`
   (about 5 GB).
5. Sanity sample: decode 300 random clips from the shards, Whisper CER against `text`
   (normalised). Also save 20 clips as WAV in one zip for a spot listen.
6. Write `corpus_report.json`: sentences, audio hours, ×RT, sample CER, phoneme coverage
   (count per symbol), any skipped rows.

**Gate B:** ≥ 12,000 rows, audio 15–24 h, sample CER ≤ `cer_test40` of the teacher + 1.5 points,
no symbol with zero occurrences that appears in `corpus_train.tsv`.
**If it fails:** CER too high → list the 30 worst sentences, look for a G2P or normaliser bug in
the worst words, fix locally, rebuild the manifest, re-run. Do not continue with a bad corpus.

---

## 9. Phase C — train the 8M student (≈ 2.5–4 GPU-h, cap 5)

Kernel `kmb-06-student` (`kaggle/k06_student/`). Inputs: `seanlon/kmb-paradee-ms`,
`seanlon/kmb-teacher-text`, `kernel_sources: ["seanlon/kmb-05-teacher-corpus"]`.
Symlink the shards: find `shard_0000.pt` by glob and `os.symlink(<its folder>, "data/teacher")`.
One kernel runs C0–C3 in order and saves everything under `/kaggle/working/train/`.
If the measured timings in C0 say the whole thing will not fit in 5 hours, split the decoder
into a second kernel `kmb-07-decoder` that attaches kernel 06's output.

### C0. Initial weights and timing

```bash
hf download sahilmahendrakar/Paradee-8M-v1.3 pytorch/text_side.pt pytorch/decoder.pt config.json --local-dir init/v13
python scripts/slice_voice.py init/v13/pytorch/text_side.pt init/text_row0.pt 0
python scripts/slice_voice.py init/v13/pytorch/decoder.pt  init/dec_row0.pt  0
```

Row 0 is `af_heart` (American female, nearest to Bella). Confirm both sliced files load:
`TextStudent("s+mlp").load_state_dict(...)` and `VDecoder(head="v2", dim=160, blocks=4).load_ckpt(...)`
with no missing or unexpected keys (apart from what `load_ckpt` itself allows).
If the v1.3 files do not load into the code at commit `2de547e`, use
`sahilmahendrakar/Paradee-8M-v1.1` (single voice, no slicing) and record that in `STATUS.md`.

Timing probe: run each trainer for 100 steps with a throwaway `--tag _probe`, read `s/step`
from the log. Set the step counts below so that **text ≤ 60 min, decoder spectral ≤ 75 min,
decoder adversarial ≤ 75 min**, never below the minimums in brackets.

### C1. Text side

```bash
python scripts/train_text.py s+mlp --device cuda --init-from init/text_row0.pt --steps 8000 --max-minutes 60 --tag _ms
```

(minimum 5,000 steps). The trainer evaluates on the 200 held-out rows every 2,000 steps and at
the end, and writes `checkpoints/text_s+mlp_ms/log.jsonl` plus student/teacher WAV pairs.

**Gate C1** (final `EVAL` line): `l_dur ≤ 0.06`, `l_f0 ≤ 0.07`, `l_asr ≤ 0.09`,
`dur_mae_frames ≤ 1.0`, and `dtw_l1` reported (keep the number). These are about 1.5× the pt-BR
fork's final losses.
**If it fails:** train from scratch — same command without `--init-from`, new tag `_ms_scratch`,
8,000 steps. Take whichever run has the lower `l_dur + l_f0 + l_asr`. If both fail, the corpus is
the suspect: go back to Gate B's diagnosis.

### C2. Decoder, spectral stage

```bash
python scripts/train_vhead.py --head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 \
  --device cuda --init-from init/dec_row0.pt --steps 8000 --eval-every 2000 --utmos --max-minutes 75 --tag _ms
```

(minimum 5,000 steps — the fork's proven number). `--shards 999` matters: the default is 8.
Each eval prints `dtw_l1` and `utmos` on 8 held-out clips and saves `step<N>.pt`.

**Gate C2:** final `dtw_l1 ≤ 0.62` and not rising over the last two evals.
**If it fails:** continue for another 4,000 steps (same command, higher `--steps`; it resumes).
If still failing, restart from `init/dec_row0.pt` with `--lr 1e-4`.

### C3. Decoder, adversarial stage

```bash
python scripts/train_vhead.py --head v2 --fine-stft --harm-weight 2 --mod-weight 5 --shards 999 \
  --device cuda --gan --mel-weight 3 --bs 8 --init-from checkpoints/vh_ms/last.pt \
  --steps 6000 --eval-every 1500 --utmos --max-minutes 75 --tag _msgan
```

(minimum 3,000 steps). Pick the checkpoint (`step<N>.pt` of either stage) with the best `utmos`
among those with `dtw_l1 ≤ 0.65`. UTMOS was trained on English ratings, so use it only to
compare checkpoints with each other and with the teacher's own held-out clips.

**Gate C3:** chosen checkpoint's `utmos ≥` (teacher UTMOS on the same 8 clips − 0.30).
**If the adversarial stage makes things worse** (UTMOS falls, or audible artefacts in the eval
WAVs): ship the C2 checkpoint. The pt-BR model shipped without an adversarial stage.

### C4. Join and score in PyTorch (same kernel)

Adapt `eval_full.py` (or write `eval_ms.py`): text side → rounded durations → alignment →
decoder, for the 200 held-out rows; save WAVs in one zip; Whisper CER for student and teacher.

**Gate C4:** student CER ≤ teacher CER + 1.5 points on the 200 held-out sentences.
**If it fails** while C1–C3 passed: the mismatch is between the features the text side predicts
and the teacher features the decoder was trained on. Train the text side for 2,000 more steps
with `--through-decoder` (adds a log-mel loss through the **frozen teacher decoder**; raise
`--steps` by 2,000 and keep the same tag so it resumes), then re-run C4. Do not use
`--through-student-decoder`: it expects the v1.0 decoder class, not the v2 head.
Report honestly if the CER stays above the gate.

Kernel outputs to keep: `checkpoints/text_s+mlp_ms/last.pt`, the chosen decoder checkpoint copied
to `checkpoints/final_decoder.pt`, logs, eval zips, `student_report.json` (all gate numbers,
s/step, GPU minutes).

---

## 10. Phase D — export (CPU; do it at the end of kernel 06 and repeat locally)

1. `python scripts/export_paradee_v1_1.py checkpoints/final_decoder.pt --text checkpoints/text_s+mlp_ms/last.pt --out models/ms_bella_8m`
   → `paradee.onnx` (fp32) and `paradee_int8.onnx`. The script checks the ONNX head against the
   PyTorch head itself; read its printed deltas.
2. Score **both ONNX files** with onnxruntime on `test_200`: Whisper CER, UTMOS, and ×RT on one
   CPU thread (`intra_op_num_threads=1`).
3. Assemble the release folder locally (download the two ONNX files from the kernel output):

```
export/ms_bella_8m/
  onnx/model.onnx              # = paradee.onnx (fp32)
  onnx/model_quantized.onnx    # = paradee_int8.onnx
  config.json                  # {"model_type":"style_text_to_speech_2","sample_rate":24000,"n_token":178,"vocab":{…},"voices":["ms_bella"],"teacher":"hexgrad/Kokoro-82M","g2p":{…frozen options…}}
  tokenizer.json               # copy from sahilmahendrakar/Paradee-8M-v1.3 (Kokoro's vocab in transformers.js format)
  g2p/                         # ms_norm.js, ms_g2p.js, lexicon_e.json, lexicon_en.json, lexicon_exceptions.json, vocab.json
  LICENSE  NOTICE  README.md
```

   The `onnx/model.onnx` + `onnx/model_quantized.onnx` names are what transformers.js looks for
   (`dtype: 'fp32'` and `dtype: 'q8'`).

**Gate D:** int8 file ≤ 10 MB; int8 CER ≤ fp32 CER + 0.5 points; int8 CER ≤ teacher CER + 1.5
points and ≤ 6%; ≥ 15× real time on one thread.
**If int8 fails only on quality:** ship fp32 as the default and report the size (about 35 MB).

---

## 11. Phase E — runtimes (local, CPU)

### E1. Python: `python/kokoro_malay/`

`__init__.py`, `__main__.py`, `tts.py`, and the G2P files + lexicons as package data.
Dependencies: `onnxruntime`, `numpy`, `soundfile` (for the CLI only).

```python
class KokoroMalay:
    def __init__(self, model_path=None, quantized=True, threads=1): ...
    def phonemize(self, text) -> list[str]      # normalize → G2P → sentence split → chunk(≤510)
    def __call__(self, text, speed=1.0) -> np.ndarray   # float32, 24 kHz
```

- Split text into sentences at `. ! ? …` and newlines first; a sentence longer than 300 phonemes
  is split again at `, ; :`; then `chunk()` as the last resort. The student was trained on single
  sentences of ≤ 400 phonemes, so never feed it a paragraph in one call.
- ids = `[0] + [vocab[c] for c in phonemes if c in vocab] + [0]`; run
  `{"input_ids": int64 [1,T], "speed": float32 [1]}`; join chunks with 120 ms of silence.
- Tests (`python/tests/`): 10 sentences from `test_200` synthesise without error; output length is
  within ±25% of 80 ms per phoneme; an empty string returns an empty array; a 1,500-character
  paragraph works.

### E2. JavaScript, primary path: transformers.js (`web/kokoro-malay.js`)

```js
import { AutoTokenizer, StyleTextToSpeech2Model, Tensor } from '@huggingface/transformers';
// model = await StyleTextToSpeech2Model.from_pretrained(repoOrPath, { dtype: 'q8', device: 'wasm' });
// const { waveform } = await model({ input_ids, speed });   // no `style` input: single voice
```

- `input_ids`: build them from `g2p/ms_g2p.js` (`loadG2P(base, opts)` then `tokens()`), as a
  `Tensor('int64', BigInt64Array, [1, T])`. `speed`: `Tensor('float32', [1.0], [1])`.
  Using the G2P's own `tokens()` avoids any difference between the tokenizer and Python.
- Same sentence splitting as Python (port the function; keep the two side by side).
- `device: 'webgpu'` is optional and must use `dtype: 'fp32'`; test it only if WASM works first.

### E3. Verify the transformers.js path in Node — and the fallback

`web/tests/node_parity.mjs`: load the local `export/ms_bella_8m` folder with transformers.js
(`env.allowLocalModels = true; env.localModelPath = …`), run 10 fixed sentences, and compare with
Python: (1) identical `input_ids`; (2) waveform lengths equal; (3) Pearson correlation ≥ 0.99
between the two waveforms for the fp32 model. The decoder uses random noise for unvoiced sound,
so if correlation is lower with identical ids, compare log-mel spectrograms instead (mean
absolute difference ≤ 0.15) and say so in the report.

**If transformers.js refuses the graph** (missing `style` input, unknown model type, file-name
rules): keep the same public API in `web/kokoro-malay.js` but create the session with
`onnxruntime-web` / `onnxruntime-node` directly (`InferenceSession.create(url, {executionProviders:['wasm']})`),
exactly as the Paradee web demo does. Record which path shipped. The owner asked for a
transformers.js version, so try it properly first and explain plainly if the fallback is used.

### E4. Demo page `web/index.html`

Text box, Speak button, speed slider, shows the phonemes. Loads the model from `../export/ms_bella_8m/`.
Serve with `python -m http.server` (the owner runs the command). No build step, ES modules from
a CDN allowed by default browsers (`https://cdn.jsdelivr.net/npm/@huggingface/transformers`), pin
the exact version tested in E3.

**Gate E:** Python tests pass; `node_parity.mjs` passes on one of the two JS paths;
`pytest g2p/tests/test_gate1.py` and `node g2p/tests/parity.mjs` still pass.

---

## 12. If Gate A fails (the teacher does not sound Malay enough)

Do these in order. Stop at the first that passes Gate A's rule. Each needs one more short listen.

1. **Widen the search (≈ 0.5 GPU-h).** Three-way blends (Bella + two donors), `wa ∈ {0.2, 0.1, 0.0}`
   (almost no Bella in the timbre), `speed ∈ {0.92, 1.0}`, and the taling-`e` written as `ɛ`
   instead of `e`. Then a 150-evaluation random walk around the best config: perturb `wa`, `wp`
   and per-donor weights by ±0.1, keep a step if `acc` improves and `cer` stays within 1 point.
2. **Ask the owner one question:** accept the best available teacher (name its weaknesses in one
   sentence), or spend more compute. Offer exactly these two options.
3. **Only if the owner chooses more compute — prosody transplant (experimental, ≈ 6 GPU-h).**
   Generate 2,000 corpus sentences with Chatterbox T2 + trim + voice conversion using
   `kaggle/k03_pilot/k03_pilot.py` (about 4 GPU-h). For each clip: Whisper word timestamps →
   rescale Kokoro's own per-phoneme durations word by word to match; F0 from the clip
   (e.g. `torchcrepe`), time-aligned to the new frames and shifted to the teacher voice's median
   F0; energy `N` from Kokoro, time-stretched. Run Kokoro's **decoder** on
   (text features with the new alignment, new F0, new N) to get teacher audio with Chatterbox's
   rhythm and melody and Kokoro's sound. Train the text side on these targets after C1
   (fine-tune 3,000 steps); the decoder training is unchanged. This has never been tried; build
   it on 50 clips first and let the owner hear them before generating 2,000.
4. The original route in `plan.md` (Chatterbox corpus + Kokoro fine-tune) stays the last resort
   and needs the owner's explicit go-ahead because of its cost.

---

## 13. Phase F — acceptance and release

1. Listening page `runs/listen/final.html`: 12 sentences from `val_100.tsv` (never trained on) —
   4 general, 3 call-centre, 3 numbers, 2 code-switch. Per sentence: teacher (Kokoro-82M +
   `ms_bella`), student int8, and the Chatterbox clip where one exists. Blind. Owner marks each
   student clip acceptable / not, and may flag words.
2. **Gate F:** owner accepts, with ≤ 2 of 12 student clips "not acceptable".
   Flagged words: fix in the lexicon or normaliser (both Python and JS, parity must stay green).
   Lexicon fixes need **no retraining** unless they introduce a phoneme sequence the corpus never
   had.
3. Write `export/ms_bella_8m/README.md` (model card): what it is, sizes, the gate numbers,
   usage in Python and JS, limitations (one voice; accent comes from a blended Kokoro voice, not
   from Malay recordings; English words only as well as the 430-word list; names and rare loans
   may be wrong), how it was made, credits.
4. `NOTICE`: Kokoro-82M (Apache-2.0, hexgrad), Paradee (Apache-2.0, Sahil Mahendrakar),
   misaki English dictionary entries (Apache-2.0), WikiPron-derived `e` labels (source data
   CC-BY-SA — attribute), g2p-id schwa dictionary (MIT), Tatoeba and Wikipedia sentences were
   used only as prompts.
5. Publishing to the Hugging Face Hub is an outward-facing step: prepare the folder and the
   exact `hf upload` command, then **ask the owner before running it**.

---

## 14. Budget and schedule

| Kernel | Phase | GPU estimate | Cap |
|---|---|---|---|
| `kmb-04-teacher-search` | A | 0.7 h | 1.2 h |
| `kmb-05-teacher-corpus` | B | 1.0 h | 1.5 h |
| `kmb-06-student` (+ `kmb-07-decoder` if split) | C, D | 3.0 h | 5.0 h |
| **Total** | | **4.7 h** | **8 h (stop and report beyond)** |

Earlier sessions used roughly 2 GPU-hours of this week's Kaggle quota (check the Kaggle account
page for the real figure before starting).
Elapsed time with the owner available for Gate A: about one working day.

---

## 15. Risks

| Risk | Likelihood | What limits the damage |
|---|---|---|
| No blend sounds Malay enough | medium | Gate A before any training; section 12; M1 and the student cost nothing until then |
| Accent score does not track what the owner hears | medium | It is checked against known clips first, and the gate is the owner's ears, not the score |
| A blended voice sounds unstable (two-speaker effect) | low–medium | `sim` and UTMOS in the search table; the owner's "not acceptable" ticks |
| `final_a='ə'` breaks loanwords and names | medium | keep-list, capitalisation rule, owner hears both variants; it ships only if chosen |
| Student noticeably worse than teacher | low | Proven on English, pt-BR, Hindi, French, Spanish; gates C1–C4 and D |
| v1.3 weights do not load into the public code | low | v1.1 single-voice weights, or from scratch (the pt-BR recipe) |
| T4 slower than assumed | medium | C0 measures first; step counts have floors and time caps |
| transformers.js cannot run the graph | medium | onnxruntime-web fallback with the same API (E3) |
| Kaggle image changes again | medium | Section 5; read the first 50 log lines of every run before waiting on it |

## 16. Explicitly not doing

- Fine-tuning Kokoro-82M. Training on Chatterbox audio. Generating the 18 GPU-hour Chatterbox corpus.
- More than one voice, voice cloning, emotion control, streaming inside the graph.
- eSpeak in any runtime (the Malay G2P is rule- and lexicon-based and already ported to JS).
- A WebGPU-specific quantised build, unless the owner asks after release.

## 17. Sources

- Paradee model cards: <https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0>, <https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.3>
- Paradee code and training README: <https://github.com/sahilmahendrakar/paradee> (commit `2de547e`)
- Paper: <https://huggingface.co/papers/2610.06817>
- Brazilian-Portuguese port: <https://huggingface.co/eraldohug/Paradee-8M-PtBr>, <https://github.com/astrowar/paradee_pt>
- Browser demos: <https://huggingface.co/spaces/KingNish/paradee-web>, <https://huggingface.co/danyalmalik/Paradee-8M-ONNX-WebGPU>
- Teacher: <https://huggingface.co/hexgrad/Kokoro-82M>
