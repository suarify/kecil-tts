import subprocess

msg = """
## 2026-11-19 Yati v0.5 PUBLISHED + Space updated (owner approved despite modest gain)
- Owner: "the difference here not significant. but ok publish n go phase4".
- Published `Suarify/kecil-tts-yati-v0.5.0` from `runs/p3_dec_gan_export/paradee_int8_webgpu.onnx` (best P3 checkpoint, step 5407, UTMOS 2.2748/1.7932). 42 files, remote sha == local, 0 float64, 8.47 MB. Card is honest: +0.12 UTMOS over v0.4.1, owner did not find the difference clearly perceptible, F0 voicing unchanged, large gap to teacher remains.
- Space `Suarify/suaradee-web` updated with the polished UI (verified remotely): Yati v0.5 dropdown, Nicole as a third speaker with its own version dropdown, auto-speaking sample chips, large render-stats panel + live waveform, sleeker fonts.
- Phase 1/2 checkpoints were NOT published.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_gate3_page.py', 'scripts/verify_gate3.py', 'scripts/extract_gate3_js.py',
         'scripts/verify_space_remote.py', 'scripts/diagnose_rerun.py', 'scripts/check_live_gan.py',
         'scripts/check_runs.py', 'scripts/make_p1_repo.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Publish Yati v0.5 + Space UI update; Phase 4 next', '--quiet'])
print('committed')
