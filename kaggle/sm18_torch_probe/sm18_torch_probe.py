import glob, os, subprocess, sys

print('PYVER', sys.version.split()[0], flush=True)
print('BEFORE', subprocess.run(['pip', 'show', 'torch', 'torchaudio'], capture_output=True, text=True).stdout[:800], flush=True)
for idx in ('https://download.pytorch.org/whl/cu130', 'https://download.pytorch.org/whl/cu128'):
    cmd = [sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', 'torch==2.11.0', 'torchaudio==2.11.0', '--index-url', idx]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print('INSTALL', idx, 'rc', p.returncode, (p.stderr or '')[-300:], flush=True)
    if p.returncode == 0:
        break
import torch, torchaudio
print('AFTER', torch.__version__, torchaudio.__version__, 'cuda', torch.cuda.is_available(), flush=True)
p = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'chatterbox-tts'], capture_output=True, text=True)
print('chatterbox install rc', p.returncode, flush=True)
p = subprocess.run([sys.executable, '-c', "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from chatterbox.vc import ChatterboxVC; print('CHATTERBOX IMPORTS OK')"], capture_output=True, text=True)
print('IMPORT rc', p.returncode, p.stdout[-500:], p.stderr[-800:], flush=True)
