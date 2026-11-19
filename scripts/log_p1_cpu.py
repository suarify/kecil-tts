import subprocess

msg = """
## 2026-11-19 REVIEW Phase 1 (CPU) DONE: log-F0 + voiced/unvoiced text head implemented + validated
- New `paradee_ms/student_uv.py` (copied to kaggle/ds_pms for Kaggle use): `TextStudentUV` subclasses TextStudent, adds `uv_proj` alongside F0_proj on the shared prosody features; `prosody()` is a line-for-line match of Kokoro `F0Ntrain` plus the UV head; `f0_hz()` converts normalised log-F0 + UV logit back to Hz with an ONNX-safe 5-frame majority filter (3D fix applied: 2D input made ONNX AveragePool reject the graph). Helpers `set_lf0_stats` (mu/sd from voiced frames of train rows) and `reinit_f0_proj` (old F0_proj emitted Hz ~200, must start near 0 for normalised log-F0) MUST be called after strict=False load of `runs/k16_final/textA_c4_last.pt`.
- `ms_train.py` gains `--uv-text`: loads TextStudentUV strict=False, reinit+stats, target lf0n=(log F0 - mu)/sd on voiced frames, l_f0 = weighted L1 (1.0 voiced / 0.1 unvoiced), l_uv = BCE-with-logits (voiced = F0>50) weight 1.0, l_n + l_dur + l_asr unchanged; checkpoint selection on l_dur + l_asr + l_uv; evaluate() and free-running call adapted to the 5-value forward. Old path untouched.
- `scripts/export_fixh.py` gains `--uv-text`: Paradee2.forward uses `ts.prosody` + `ts.f0_hz` instead of `predictor.F0Ntrain`; loads TextStudentUV strict=True (mu/sd buffers come from the checkpoint); --state+--uv-text guarded off.
- CPU validation: 2-step train smoke (losses fell l_f0 2.35->2.33, l_uv 0.644->0.639); 100-step timing at bs=16 = **1.41 s/step** (>> REVIEW's 0.35 threshold, so 10k steps ~3.9 h on CPU). Full UV export smoke on the untrained init: int8 ONNX 8.47 MB, head no-op parity 3e-07, ORT length matches torch, webgpu_patch -> 0 float64 Cast ops.
- NEXT: needs owner's explicit GPU yes before Kaggle kernel `sm-20-f0uv` (10k steps, Kaggle account suarifymalaysia). Then export + sota_eval + Gate 1 (dropout<=3%, V/UV<=7%, jumps>25<=3%, span 0.7-1.5, UTMOS>=2.1447, CER not >0.5 worse) + blind owner A/B page.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
subprocess.run(['git', 'add', 'paradee_ms/ms_train.py', 'paradee_ms/student_uv.py',
                'scripts/export_fixh.py', 'STATUS.md'], check=True)
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 1 (CPU): log-F0 + UV text head, train + export integration', '--quiet'])
print('committed')
