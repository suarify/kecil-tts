"""Copy the front-end files into python/kokoro_malay/ so the package has no dependency on the repo layout."""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dst = os.path.join(ROOT, 'python', 'kokoro_malay')
for f in ('ms_norm.py', 'ms_g2p.py', 'lexicon_e.json', 'lexicon_en.json', 'lexicon_exceptions.json', 'vocab.json'):
    shutil.copy(os.path.join(ROOT, 'g2p', f), os.path.join(dst, f))
print('copied front end to', dst)
