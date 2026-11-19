import subprocess

msg = """
## 2026-11-19 REVIEW Phase 2 result: Gate 2 FAILED (P2 worse than P1)
- Kernel `suarifymalaysia/sm-22-gta` COMPLETE; `dec_gta.pt` downloaded after a disk-full retry (freed the upload staging folder). Training was 3k steps, bs16, seg64, lr1e-4, gta_p 0.7.
- Exported `runs/sota_p2_export/paradee_int8_webgpu.onnx`: 8.47 MB, 0 float64, head parity 1.5e-07, identical FIX H recipe, 13.5x realtime.
- `sota_eval.py` p2: UTMOS 2.0101/1.5395 vs P1 2.1799/1.7647 (and v0.4.1 2.1579/1.7383). Gate 2 automatic threshold >= P1+0.1 (2.28) FAILS decisively. F0 health unchanged (text model unchanged: dropout 17.89%, uv_err 27.69%, span 3.348).
- Insight: GTA fine-tuning trains the decoder on the text model's own predicted F0/N, but those inputs still carry ~18% voicing dropout, so the decoder learned to reproduce the artifacts and got worse. Phase 2 was doomed by Phase 1's unresolved F0 problem; GTA only helps once the text-side F0 is genuinely clean.
- REVIEW says if both Gate 2 paths fail, keep P1's decoder and still go to Phase 3 (use GTA inputs there anyway). Nothing published; v0.4.1 remains the live voice. Owner decision pending (blind P2 A/B vs Phase 3 vs stop).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/download_sm22.py', 'scripts/gate2_check.py', 'scripts/log_p2_cpu.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 2 Gate 2 FAILED: P2 UTMOS 2.01 vs P1 2.18', '--quiet'])
print('committed')
