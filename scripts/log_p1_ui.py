import subprocess

msg = """
## 2026-11-19 Space UI polish (ready for the P1 voice rollout)
- `scripts/polish_space.py` (idempotent) updates `runs/local_v041/index.html`:
  - Sleeker type: Plus Jakarta Sans body + Outfit display (replaces Inter/Sora).
  - Live speed-highlight strip as the top row above the textarea: shows realtime factor, a progress bar, and a "+X% vs <previous version>" comparison. On every generation it updates with the measured speed for that run. Benchmarks are per-model in a SPEED map.
  - Sample-text chips now fill the textarea AND immediately trigger Speak.
  - Adds `Suarify/kecil-tts-yati-v0.5.0` ("v0.5 (SOTA P1)") to the Yati version dropdown (name is an assumption; confirm with owner before publish).
- `scripts/verify_space.py` smoke-checks all six changes. Space upload is deferred until P1 passes Gate 1 + owner A/B.
- Owner's rollout instruction: publish the P1 model, add the voice to the HF Space dropdown, and apply the UI polish. Publishing still gated on automatic + owner Gate 1.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['scripts/polish_space.py', 'scripts/verify_space.py', 'runs/local_v041/index.html', 'STATUS.md']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Space UI polish: sleek fonts, live speed strip, chips speak, P1 dropdown', '--quiet'])
print('committed')
