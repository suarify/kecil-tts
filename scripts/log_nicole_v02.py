import subprocess

msg = """
## 2026-11-20 Nicole v0.2 PUBLISHED (+25% gain, correct version numbering) + Space updated
- Owner: "make the 0.5 louder 25%... abit too soft" and "why no 0.2/0.3 in space, straight 0.5". Clarified: the new Nicole was mislabeled v0.5 (after its Yati v0.5 source); semantically it is Nicole v0.2 (next after v0.1). Renamed accordingly.
- Added `--gain` to `scripts/export_fixh.py` (constant multiplier baked into HeadX output, so it flows through ONNX). Exported `nicole_gan.pt` with --gain 1.25 -> `runs/nicole_v02_gain_export/paradee_int8_webgpu.onnx`. Verified: RMS 0.0575 -> 0.0719 (exactly x1.25), peak 0.405 safe, 0 float64, 8.47 MB. Note the head no-op parity check now legitimately differs (gain is intentional).
- sota_eval tag nicole_v02: UTMOS 1.939/1.514, F0 dropout 28.1%, uv_err 32.5%. Card honest: objective UTMOS lower than v0.1 (2.18) but loudness fixed (v0.1 was RMS 0.02-0.035; now ~0.052-0.072).
- Published `Suarify/kecil-tts-nicole-v0.2` (38 files, sha match, 0 float64, new `kecil_tts-0.2.0` wheel). Space: Nicole card link -> v0.2, dropdown v0.2 (latest) + v0.1, speed map + footer updated, verified live (correct order, no JS errors).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/export_fixh.py', 'scripts/make_nicole_v02_repo.py', 'scripts/space_nicole_v02.py',
         'scripts/read_nicole_v02_eval.py', 'scripts/check_live_versions.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Publish Nicole v0.2 with +25% gain; Space dropdown fixed', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
r2 = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
print('push rc', r2.returncode)
