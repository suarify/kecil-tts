"""sm20: same recipe as sm19 but with hard timeouts on every pip step and early progress markers."""
import json
import os

src = 'kaggle/sm19_nicole_final/sm19_nicole_final.py'
s = open(src, encoding='utf-8').read()
# wrap every pip call in `timeout 900` so a stuck prompt cannot hang for hours
s = s.replace("def sh(cmd, name=None, cwd=None):", "def sh(cmd, name=None, cwd=None):\n    if cmd.startswith('pip ') or ' -m pip install' in cmd:\n        cmd = 'timeout 900 ' + cmd")
# mark progress right after the input printout
s = s.replace("print('python', sys.version.split()[0], flush=True)",
              "print('python', sys.version.split()[0], flush=True)\nprint('SM20 START', flush=True)")
os.makedirs('kaggle/sm20_nicole_retry', exist_ok=True)
open('kaggle/sm20_nicole_retry/sm20_nicole_retry.py', 'w', encoding='utf-8').write(s)
meta = {'id': 'suarifymalaysia/sm-20-nicole-retry', 'title': 'sm 20 nicole retry',
        'code_file': 'sm20_nicole_retry.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'keywords': ['gpu'], 'dataset_sources': ['suarifymalaysia/sm12-in'], 'kernel_sources': [],
        'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}
json.dump(meta, open('kaggle/sm20_nicole_retry/kernel-metadata.json', 'w', encoding='utf-8'), indent=1)
print('sm20 written')
