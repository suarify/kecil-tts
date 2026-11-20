import subprocess

msg = """
## 2026-11-20 Fenrir training COMPLETE - export/eval in progress
- Kernel sm-27-fenrir-train COMPLETE in 84.8 min. Stages: rows 16.6m (1,281 rows), text 8.6m (early-stopped best step 4,500), GTA 1.4m, decoder 16.3m (12k steps), GAN 38.1m (2.1k steps). Checkpoints downloaded: fenrir_text.pt (step 4500), fenrir_dec.pt (12000), fenrir_gan.pt (2100).
- GAN log: no NONFINITE; D loss ~3.8 vs G adv ~2.0 (D still winning but finite); held spec loss before 3.747 -> after 2.527 (improved substantially during GAN).
- Owner: rename fenrir -> FIRDAUS for the public voice name (keep internal ids as fenrir). Build `scripts/make_fenrir_page.py` blind 3-way page {Yati v0.5, Nicole, Firdaus}.
- Exporting fenrir_gan.pt with the identical FIX H recipe + webgpu patch + sota_eval (running). Next: pick best checkpoint by UTMOS, build the blind page, owner gate before Nicole.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_fenrir_page.py', 'scripts/wait_sm27.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Fenrir training complete; Firdaus branding; export/eval', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
