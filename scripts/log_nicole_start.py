import subprocess

msg = """
## 2026-11-20 Nicole pipeline STARTED (owner approved after Firdaus gate)
- Firdaus v0.1 published and added to Space (4th speaker). Owner accepted robotic quality for v0.1.
- Nicole: dataset suarifymalaysia/sm28-nicole-vc-in uploaded (Yati v0.5 clips bundle + ref_nicole + texts), kernel sm-28-nicole-vc RUNNING (VC to af_nicole, loudness-normalized RMS 0.07).
- Prepared: sm-29-nicole-train kernel (female voice af_nicole, default f0 settings - fixed a bad find/replace that had left am_nicole + male f0 flags). Will build suarifymalaysia/sm29-nicole-train-in dataset from the VC output, then push. Same recipe: rows -> text 10k UV -> GTA -> decoder 12k -> GAN 2k from Yati v0.5 inits.
- Next: export/eval, blind page, owner gate before publishing.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/make_nicole_vc_dataset.py', 'scripts/make_nicole_vc_kernel.py',
         'scripts/make_nicole_train_dataset.py', 'scripts/make_nicole_train_kernel.py',
         'scripts/fix_nicole_kernel.py', 'scripts/check_nicole_kernel.py', 'scripts/check_nicole_kernel2.py',
         'scripts/wait_sm28.py', 'kaggle/sm29_nicole_train/sm29_nicole_train.py',
         'kaggle/sm29_nicole_train/kernel-metadata.json']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Nicole pipeline started: VC kernel running, train kernel prepared', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
