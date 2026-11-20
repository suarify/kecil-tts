import subprocess

msg = """
## 2026-11-20 Phase 5 rollout STARTED: Fenrir first (owner approved, 6.5h GPU budget)
- Owner: "train fenrir and nicole from yati now... yati is clearest, michael/nicole volume very soft... prioritize fenrir first... 6.5hr GPU left". Approved plan: full fenrir pipeline (~1.8h) -> export/eval/blind page -> owner gate BEFORE Nicole.
- Root cause of softness measured: Nicole output RMS 0.018-0.035 vs Yati v0.5 0.036-0.052; the VC output was quiet and the model learned that level. Fix: normalize every VC clip to RMS 0.07 (Yati teacher-row level) with peak limiting at 0.95 in the VC kernel.
- Step A (CPU, done): `scripts/gen_yati_v05_clips.py` generated all 1,281 Yati v0.5 student clips from `runs/p3_dec_gan_export/paradee_int8_webgpu.onnx` -> `runs/pathb_yati_v05_clips` (2.1 min).
- Dataset `suarifymalaysia/sm26-fenrir-vc-in` uploaded (bundled ONE .pt of 1,281 clips, ref_fenrir.wav, texts.tsv - avoids the 948/1281 wav-folder mount bug). Kernel `sm-26-fenrir-vc` RUNNING: ChatterboxVC to am_fenrir + loudness normalization -> `fenrirB.pt`.
- Kernel `sm-27-fenrir-train` prepared: rows (ms_rows with `--voice am_fenrir --f0-min 50 --f0-max 300`, male settings) -> text 10k UV from Yati v0.5 text -> GTA rows -> decoder 12k GTA from Yati v0.5 decoder -> short GAN 2k (D warmup 100, D lr 1e-4 to avoid the D-wins problem). Saves fenrir_text/dec/gan.pt.
- Inits: text from `runs/text_f0uv.pt`, decoder from `runs/kaggle_sm21gan/dec_gan.pt` (the Yati v0.5 weights).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/gen_yati_v05_clips.py', 'scripts/make_fenrir_vc_dataset.py',
         'scripts/make_fenrir_train_dataset.py', 'scripts/make_gta_cli.py', 'scripts/wait_sm26.py',
         'kaggle/sm26_fenrir_vc/sm26_fenrir_vc.py', 'kaggle/sm26_fenrir_vc/kernel-metadata.json',
         'kaggle/sm27_fenrir_train/sm27_fenrir_train.py', 'kaggle/sm27_fenrir_train/kernel-metadata.json']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Phase 5: Fenrir Path-B from Yati v0.5 - VC + train kernels', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
