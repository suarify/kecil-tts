"""sm-12 v10: install and preload libcudart.so.12 directly in the main process."""
p = 'kaggle/sm12_nicole_data/sm12_nicole_data.py'
s = open(p, encoding='utf-8').read()
anchor = "sh(f'{sys.executable} -c \"from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from chatterbox.vc import ChatterboxVC; print(\\'imports ok\\')\"', 'imports')"
assert anchor in s, 'anchor not found'
addition = anchor + """
# Kaggle's default image has torchaudio built for CUDA 12 but no libcudart.so.12 anywhere.
# Install the tiny runtime wheel and preload the library INTO THIS PROCESS before chatterbox imports it.
sh('pip install -q nvidia-cuda-runtime-cu12', 'pip_cudart')
import ctypes
cud = []
for pat in ('/usr/local/lib/python3.13/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12',
            '/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'):
    cud += glob.glob(pat)
print('cudart candidates:', cud, flush=True)
for q in cud:
    try:
        ctypes.CDLL(q, mode=ctypes.RTLD_GLOBAL)
        print('preloaded cudart from', q, flush=True)
        break
    except Exception as e:
        print('cudart preload failed:', type(e).__name__, str(e)[:100], flush=True)
"""
s = s.replace(anchor, addition)
open(p, 'w', encoding='utf-8').write(s)
print('sm12 v10 patched')
