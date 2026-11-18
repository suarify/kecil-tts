"""Same BYOD-image environment fix for the fenrir VC kernel."""
p = 'kaggle/sm14_fenrir_data/sm14_fenrir_data.py'
s = open(p, encoding='utf-8').read()
start = s.index("sh('pip install -q chatterbox-tts soundfile')")
end = s.index('\nREF = glob.glob', start)
new = """sh('pip install -q chatterbox-tts', 'pip_cb')
sh('pip uninstall -y -q torchvision || true', 'pip_tv')
sh(f'{sys.executable} -c "from chatterbox.vc import ChatterboxVC; print(\\'vc imports ok\\')"', 'imports')
"""
s = s[:start] + new + s[end:]
open(p, 'w', encoding='utf-8').write(s)
print('fenrir setup replaced; lines:', len(s.splitlines()))

import json
m = 'kaggle/sm14_fenrir_data/kernel-metadata.json'
d = json.load(open(m, encoding='utf-8'))
d['docker_image'] = 'gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01'
json.dump(d, open(m, 'w', encoding='utf-8'), indent=1)
print('docker_image set')
