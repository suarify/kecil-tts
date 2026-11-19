import subprocess

msg = """
## 2026-11-19 REVIEW Phase 2 (CPU) DONE: GTA inputs + decoder trainer + Kaggle kernel prepared
- Owner chose "Phase 2 GTA" (after initially saying "retry" then clarifying), with explicit GPU approval already given for Phase 2.
- `scripts/make_gta.py`: ran the P1 text model (TextStudentUV, `runs/kaggle_sm21/text_f0uv.pt`) over all 1,281 rows with MEASURED durations, stored `gta_F0` (Hz via f0_hz), `gta_N`, `gta_asr` = asr_tok @ aln -> `data/teacher_gta/shard_0000.pt` (785 MB).
- `ms_train.py decoder` gains `--gta-p`: per batch item, with prob gta_p uses GTA inputs; c0 from the INPUT F0; harm_loss/mod_loss now per-item and weighted 0 for GTA items (their pitch no longer matches the target audio), mel+STFT unchanged. Intermediate checkpoints saved every 1k steps. Old path (gta_p=0) unchanged. Smoke-tested 2 steps OK (fixed a vm reduction shape bug).
- Dataset `suarifymalaysia/sm22-gta-in` uploaded and verified ready with all 17 files (GTA shard + dec_c4_last.pt + scripts).
- Kernel `suarifymalaysia/sm-22-gta` pushed and QUEUED: `dec --gta-p 0.7 --held 60 --steps 3000 --bs 16 --seg 64 --lr 1e-4 --eval-every 500`, saves `dec_gta.pt`.
- `scripts/make_sota_p1_page.py` parameterized with `--label-base`/`--label-p1` so it can build the P2-vs-P1 page; JS syntax checked.
- NEXT on completion: download dec_gta.pt -> export with `--uv-text` + identical FIX H -> webgpu patch (0 float64) -> sota_eval tag p2 -> Gate 2 (UTMOS >= P1+0.1 OR owner picks P2 >=8/12) -> publish only after owner gate.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_gta.py', 'scripts/make_sm22_dataset.py', 'scripts/wait_sm22.py',
         'scripts/read_gta_log.py', 'scripts/make_sota_p1_page.py', 'scripts/check_page_labels.py',
         'scripts/check_p2_page.py', 'paradee_ms/ms_train.py', 'kaggle/sm22_gta/sm22_gta.py',
         'kaggle/sm22_gta/kernel-metadata.json', 'scripts/log_p2_start.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 2 CPU: GTA inputs, decoder trainer, sm-22 kernel', '--quiet'])
print('committed')
