"""Final environment fix for the data kernel: chatterbox first, then the official torch pair LAST."""
import os

src = 'kaggle/sm16_nicole_data2/sm16_nicole_data2.py'
s = open(src, encoding='utf-8').read()
start = s.index("sh('pip install -q chatterbox-tts', 'pip_cb')")
end = s.index('\nimport pandas as pd', start)
new = r"""sh('pip install -q chatterbox-tts soundfile', 'pip_cb')
sh('pip uninstall -y -q torchvision torchao torchtext || true', 'rm_extras')
sh('pip install -q "transformers==4.46.3"', 'pip_tr')
# CRITICAL ORDER: install torch+torchaudio LAST so no later pip step can restore Kaggle's broken pair.
sh(f'{sys.executable} -m pip install -q --force-reinstall torch==2.11.0 torchaudio==2.11.0 '
   '--index-url https://download.pytorch.org/whl/cu130', 'torch_pair')
sh(f'{sys.executable} -c "import torch, torchaudio; print(\'PAIR\', torch.__version__, torchaudio.__version__, torch.cuda.is_available())"', 'verify_pair')
sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from chatterbox.vc import ChatterboxVC; print(\'imports ok\')"', 'imports')
"""
s = s[:start] + new + s[end:]
os.makedirs('kaggle/sm19_nicole_final', exist_ok=True)
open('kaggle/sm19_nicole_final/sm19_nicole_final.py', 'w', encoding='utf-8').write(s)
import json
meta = {'id': 'suarifymalaysia/sm-19-nicole-final', 'title': 'sm 19 nicole final',
        'code_file': 'sm19_nicole_final.py', 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'keywords': ['gpu'], 'dataset_sources': ['suarifymalaysia/sm12-in'], 'kernel_sources': [],
        'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4'}
json.dump(meta, open('kaggle/sm19_nicole_final/kernel-metadata.json', 'w', encoding='utf-8'), indent=1)
print('sm19 written')
