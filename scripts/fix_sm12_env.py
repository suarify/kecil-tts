"""Replace the sm-12 setup block with the exact working setup from k15 (BYOD image)."""
p = 'kaggle/sm12_nicole_data/sm12_nicole_data.py'
s = open(p, encoding='utf-8').read()
start = s.index("sh('pip install -q chatterbox-tts soundfile', 'pip')")
end = s.index('\nimport pandas as pd', start)
new = """sh('pip install -q chatterbox-tts', 'pip_cb')          # exact setup from the working k15 kernel
sh('pip uninstall -y -q torchvision || true', 'pip_tv')
sh(f'{sys.executable} -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from chatterbox.vc import ChatterboxVC; print(\\'imports ok\\')"', 'imports')
"""
s = s[:start] + new + s[end:]
open(p, 'w', encoding='utf-8').write(s)
print('setup replaced; lines now:', len(s.splitlines()))

import json
m = 'kaggle/sm12_nicole_data/kernel-metadata.json'
d = json.load(open(m, encoding='utf-8'))
d['docker_image'] = 'gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01'
json.dump(d, open(m, 'w', encoding='utf-8'), indent=1)
print('docker_image set')
