"""Rename the Python package suaradee -> kecil_tts (distribution 'kecil-tts'), keep a suaradee shim.

- python/kecil_tts/  = the real package (renamed from python/suaradee/)
- python/suaradee/   = tiny shim re-exporting from kecil_tts (backwards compatible)
- pyproject.toml: name 'kecil-tts', packages ['kecil_tts','suaradee']
- js/web: keep a KecilTTS alias export next to Suaradee.
"""
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 1) rename the real package directory
src, dst = 'python/suaradee', 'python/kecil_tts'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))
# rewrite internal references inside the package
for f in os.listdir(dst):
    if f.endswith('.py'):
        p = os.path.join(dst, f)
        s = open(p, encoding='utf-8').read()
        s = s.replace('from .tts import SAMPLE_RATE, Suaradee', 'from .tts import SAMPLE_RATE, KecilTTS')
        s = s.replace("'Suaradee'", "'KecilTTS'")
        open(p, 'w', encoding='utf-8').write(s)

# rename the class in tts.py
p = os.path.join(dst, 'tts.py')
s = open(p, encoding='utf-8').read()
s = s.replace('class Suaradee', 'class KecilTTS')
s = s.replace('Suaradee', 'KecilTTS')
open(p, 'w', encoding='utf-8').write(s)

# 2) replace old dir with a backwards-compatible shim
shutil.rmtree(src)
os.makedirs(src, exist_ok=True)
open(f'{src}/__init__.py', 'w', encoding='utf-8').write(
    '"""Backwards-compatible shim: use `from kecil_tts import SAMPLE_RATE, KecilTTS` instead."""\n'
    'from kecil_tts import SAMPLE_RATE, KecilTTS\n\n'
    'Suaradee = KecilTTS  # old name still works\n'
    '__all__ = ["KecilTTS", "Suaradee", "SAMPLE_RATE"]\n'
)
open(f'{src}/tts.py', 'w', encoding='utf-8').write(
    'from kecil_tts.tts import *  # noqa: F401,F403\n'
    'from kecil_tts.tts import KecilTTS\n'
    'Suaradee = KecilTTS\n'
)
# copy data files (json) into the shim too so old direct imports still work
for f in os.listdir(dst):
    if f.endswith('.json'):
        shutil.copy(f'{dst}/{f}', f'{src}/{f}')
# ms_g2p / ms_norm modules: shim re-export
for mod in ('ms_g2p.py', 'ms_norm.py'):
    if os.path.exists(f'{dst}/{mod}'):
        open(f'{src}/{mod}', 'w', encoding='utf-8').write(f'from kecil_tts.{mod[:-3]} import *  # noqa\n')

# 3) pyproject
p = 'python/pyproject.toml'
s = open(p, encoding='utf-8').read()
s = s.replace('name = "suaradee"', 'name = "kecil-tts"')
s = s.replace('packages = ["suaradee"]', 'packages = ["kecil_tts", "suaradee"]')
s = s.replace('suaradee = ["*.json", "*.onnx"]', 'kecil_tts = ["*.json", "*.onnx"]\nsuaradee = ["*.json", "*.onnx"]')
open(p, 'w', encoding='utf-8').write(s)

# 4) JS: add KecilTTS alias next to Suaradee
for jp in ('web/suaradee.js',):
    s = open(jp, encoding='utf-8').read()
    if 'export const KecilTTS' not in s:
        s += '\n// Renamed export alias (Suaradee kept for backwards compatibility).\nexport const KecilTTS = Suaradee;\n'
        open(jp, 'w', encoding='utf-8').write(s)

print('local rename done')
print('kecil_tts files:', sorted(os.listdir(dst)))
print('suaradee shim files:', sorted(os.listdir(src)))
