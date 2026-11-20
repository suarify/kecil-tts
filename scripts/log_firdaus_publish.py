import subprocess

msg = """
## 2026-11-20 Firdaus v0.1 PUBLISHED (owner approved despite robotic quality)
- Owner: "ok can push firdaush as 0.1 first, it sound robotic though" -> published `Suarify/kecil-tts-firdaus-v0.1` from `runs/fenrir_gan_export/paradee_int8_webgpu.onnx` (best GAN checkpoint, UTMOS 1.8425/1.4653). 36 files, sha match, 0 float64, 8.47 MB. Internal training id stays `fenrir`; public voice name is Firdaus.
- Card is honest: early/rough release, owner found it robotic, UTMOS table (teacher 3.553, Yati v0.5 2.275, Nicole 2.180, Firdaus 1.843, Michael 1.813), F0 health dropout 33%/uverr 36%/span 2.35, stats correctly re-centered to male 122 Hz so not a setup bug, non-GAN checkpoint was worse (1.27).
- Space updated: Firdaus added as a 4th speaker with its own dropdown + model detection + speed map. Verified live: speakers Yati/Michael/Firdaus/Nicole, zero JS errors.
- Nicole pipeline remains prepared but paused (owner wanted Firdaus gated first).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_firdaus_repo.py', 'scripts/space_firdaus.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Publish Firdaus v0.1 + add to Space', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
