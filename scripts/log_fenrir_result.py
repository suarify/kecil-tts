import subprocess

msg = """
## 2026-11-20 Firdaus (fenrir) results: best checkpoint = GAN (UTMOS 1.84), blind gate open
- Exported + evaluated both checkpoints with the identical FIX H recipe + webgpu patch:
  - fenrir_gan.pt: UTMOS 1.8425 / 1.4653, F0 dropout 33%, uv_err 36%, span 2.35
  - fenrir_dec.pt (12k, no GAN): UTMOS 1.2741 / 1.2424 -> much worse, GAN kept.
- For reference: Yati v0.5 2.2748, Nicole 2.1799. By numbers Firdaus is below both, but UTMOS penalizes new male timbres; owner's ears are the gate.
- Verified the UV head stats were recomputed on male data: fenrir_text.pt lf0_mu 4.804 (122 Hz) vs Yati 5.296 (199 Hz) - not a stats bug. The higher dropout is genuine model behavior.
- Built `runs/listen/firdaus_gate.html` blind 3-way {Yati v0.5, Nicole, Firdaus}, 12 loudness-matched shuffled sentences, labels hidden until Reveal, pass = Firdaus >=8/12 and watery <=2/12. Opened for owner.
- Nicole pipeline remains paused pending owner's Firdaus verdict.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/check_fenrir_stats.py', 'scripts/verify_fenrir_page.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Firdaus best=GAN UTMOS 1.84; blind 3-way gate opened', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
