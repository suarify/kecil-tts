import subprocess

msg = """
## 2026-11-20 Nicole v0.5 training + export DONE - blind gate open
- Kernel sm-29-nicole-train COMPLETE (rows 1,281, text early-stop, decoder 12k, GAN 2.1k, no NaN; held spec 3.37->2.45). Checkpoints downloaded.
- Exported both with identical FIX H + webgpu patch:
  - nicole_v05_gan: UTMOS 1.9669/1.4944, F0 dropout 28.1%, uv_err 32.5%, output RMS 0.052
  - nicole_v05_dec (no GAN): UTMOS 1.7387/1.3988 -> GAN kept.
- Compare: old Nicole v0.1 UTMOS 2.1799, Yati v0.5 2.2748. New Nicole is objectively LOWER UTMOS but the loudness fix worked (RMS 0.052 vs old ~0.02-0.035) and F0 dropout improved vs Firdaus (28% vs 33%). Owner's ears decide publish vs keep old.
- Built `runs/listen/nicole_v05_gate.html` blind 3-way {Yati v0.5, Nicole v0.1 old, Nicole v0.5 new}, 12 loudness-matched shuffled sentences, rendered old Nicole wavs from runs/nicole_v010_repo for the comparison. Opened for owner.
- Note: GitHub source was pushed to suarify/kecil-tts main earlier this session (after stripping large .pt blobs).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/export_nicole.py', 'scripts/make_nicole_gate.py', 'scripts/verify_nicole_gate.py', 'scripts/log_github_push.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Nicole v0.5 export + blind gate', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
r2 = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
print('push rc', r2.returncode)
