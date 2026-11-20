"""Create Nicole training kernel from the Fenrir one (female voice settings, af_nicole)."""
import os

p = 'kaggle/sm27_fenrir_train/sm27_fenrir_train.py'
s = open(p, encoding='utf-8').read()
s = s.replace('# sm27-fenrir-train', '# sm29-nicole-train')
s = s.replace('Fenrir', 'Nicole').replace('fenrir', 'nicole')
# female voice: use af_nicole, default f0-min 70 (remove the male overrides)
s = s.replace('--voice am_fenrir --f0-min 50 --f0-max 300 --letters', '--voice af_nicole --letters')
s = s.replace('(male voice: f0-min 50, am_fenrir)', '(female voice: af_nicole, default f0 settings)')
s = s.replace('sm27_report.json', 'sm29_report.json')
os.makedirs('kaggle/sm29_nicole_train', exist_ok=True)
open('kaggle/sm29_nicole_train/sm29_nicole_train.py', 'w', encoding='utf-8').write(s)
meta = '''{
 "id": "suarifymalaysia/sm-29-nicole-train",
 "title": "sm 29 nicole train",
 "code_file": "sm29_nicole_train.py",
 "language": "python",
 "kernel_type": "script",
 "is_private": true,
 "enable_gpu": true,
 "enable_tpu": false,
 "enable_internet": true,
 "keywords": ["gpu"],
 "dataset_sources": ["suarifymalaysia/sm29-nicole-train-in", "suarifymalaysia/sm-g2p"],
 "kernel_sources": [],
 "competition_sources": [],
 "model_sources": [],
 "machine_shape": "NvidiaTeslaT4"
}
'''
open('kaggle/sm29_nicole_train/kernel-metadata.json', 'w', encoding='utf-8').write(meta)
# verify replacements
assert 'am_fenrir' not in s and 'f0-min 50' not in s
assert 'nicoleB.pt' in s and 'af_nicole' in s
print('nicole training kernel prepared')
