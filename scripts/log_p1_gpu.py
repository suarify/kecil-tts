import subprocess

msg = """
## 2026-11-19 REVIEW Phase 1 GPU run prepared (owner approved Kaggle GPU)
- Owner explicitly approved Kaggle GPU for the 10k-step UV text training (CPU timing was 1.41 s/step, 4x over REVIEW's 0.35 threshold).
- Dataset `suarifymalaysia/sm21-f0uv-in` uploaded: all 16 PMS scripts (incl. new `student_uv.py` + modified `ms_train.py`), exact 1,281-row `data/teacher/shard_0000.pt`, and `textA_c4_last.pt`. First upload silently kept only the root-level file; re-uploaded flat -> 17 files, status ready.
- Kaggle kernel `suarifymalaysia/sm-21-f0uv-text-head` pushed; first run ERROR at env-setup because the dataset had not finished processing (glob found no ms_train.py). Re-pushed after the dataset showed all 17 files; now RUNNING/QUEUED. Kernel: proven torch 2.11.0+cu130 LAST env recipe, kokoro 0.9.4 + misaki stubs, `--uv-text --held 60 --steps 10000 --bs 16 --eval-every 500`; saves `/kaggle/working/text_f0uv.pt`.
- Corrected `ms_train.py` UV target to REVIEW spec: `v = F0>0`, log-F0 linearly interpolated across unvoiced gaps (edges held constant), normalised by mu/sd. Added `--uv-unvoiced-weight` (0.1; Gate 1 retry 0.0) and `--uv-uv-weight` (1.0; retry 2.0). Smoke: held l_f0 dropped 2.35 -> 1.42, confirming the earlier constant-0 target was inflating the loss. NOTE: the running kernel used the earlier constant-0 target; if Gate 1 fails, retry with the corrected code + retry weights.
- `scripts/sota_eval.py` gained `--uv-text` (loads TextStudentUV, uses prosody+f0_hz) and now uses the FIRST 60 rows (the actual training held-out set) instead of the last 60 for dropout/V/UV/jumps/span.
- `scripts/make_sota_p1_page.py`: builds the blind, RMS-matched, shuffled 12-sentence owner page with label reveal, "pick cleaner" + "watery" ticks and P1>=8/12 tally. Smoke-tested.
- v0.4.1 export recipe verified from the published ONNX: default FIX H (K=64, hang=21, burst gated, gap_keep=1, no air/no smoothing); P1 will be exported with identical settings for a fair A/B.
- NEXT: when kernel completes -> download `text_f0uv.pt` -> export with `--uv-text` + FIX H -> webgpu patch (0 float64) -> sota_eval (Gate 1 automatic) -> if pass, build `runs/listen/sota_p1.html` for owner.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['paradee_ms/ms_train.py', 'scripts/sota_eval.py', 'scripts/make_sota_p1_page.py',
         'scripts/wait_sm21.py', 'scripts/make_sm21_dataset.py', 'scripts/fix_sm21_upload.py',
         'scripts/log_p1_cpu.py', 'scripts/parse_sm21_log.py', 'kaggle/sm21_f0uv/sm21_f0uv.py',
         'kaggle/sm21_f0uv/kernel-metadata.json', 'STATUS.md']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 1: GPU kernel prepared, UV eval + blind page builder', '--quiet'])
print('committed')
