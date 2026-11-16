"""Sync local sources with the Kecil-TTS rename (model ids, cards, publish scripts, space mirror)."""
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 1) cards
for f in os.listdir('hf'):
    if f.startswith('model_card_v') and f.endswith('.md'):
        p = os.path.join('hf', f)
        s = open(p, encoding='utf-8').read()
        s = s.replace('lseanlon/suaradee-michael-v', 'Suarify/kecil-tts-michael-v')
        s = s.replace('lseanlon/suaradee-v', 'Suarify/kecil-tts-yati-v')
        s = s.replace('Suarify/suaradee-michael-v', 'Suarify/kecil-tts-michael-v')
        s = s.replace('Suarify/suaradee-v', 'Suarify/kecil-tts-yati-v')
        voice = 'Michael' if 'michael' in s.lower()[:400] else 'Yati'
        s = re.sub(r'^# Suaradee v([0-9.]+)', rf'# Kecil-TTS {voice} v\1', s, flags=re.M)
        s = s.replace('Suaradee v', f'Kecil-TTS {voice} v').replace('Suaradee', 'Kecil-TTS')
        open(p, 'w', encoding='utf-8').write(s)
        print('card', f)

# 2) publish/build scripts: repo naming
p = 'scripts/publish_hf.py'
s = open(p, encoding='utf-8').read()
s = s.replace("build_wheel(ver, f'{A.user}/suaradee-v{ver}'", "build_wheel(ver, f'{A.user}/kecil-tts-yati-v{ver}'")
s = s.replace("repo = f'{A.user}/suaradee-v{ver}'", "repo = f'{A.user}/kecil-tts-yati-v{ver}'")
s = s.replace("commit_message=f'Suaradee v{ver}: 8.5 MB Malay TTS", "commit_message=f'Kecil-TTS Yati v{ver}: 8.5 MB Malay TTS")
open(p, 'w', encoding='utf-8').write(s)
print('publish_hf patched')

p = 'scripts/build_hf.py'
s = open(p, encoding='utf-8').read()
s = s.replace("REPO = f'{A.user}/suaradee-v{A.version}'", "REPO = f'{A.user}/kecil-tts-yati-v{A.version}'")
s = s.replace('lseanlon/suaradee-v0.3', 'lseanlon/kecil-tts-yati-v0.3')
s = s.replace('lseanlon/suaradee-v0.2', 'lseanlon/kecil-tts-yati-v0.2')
s = s.replace('lseanlon/suaradee-v0.1', 'lseanlon/kecil-tts-yati-v0.1')
open(p, 'w', encoding='utf-8').write(s)
print('build_hf patched')

# 3) python/js default repo ids
for p, subs in {
    'python/suaradee/tts.py': [("REPO = 'lseanlon/suaradee-v0.3'", "REPO = 'Suarify/kecil-tts-yati-v0.4'")],
    'web/suaradee.js': [("model = 'lseanlon/suaradee-v0.3'", "model = 'Suarify/kecil-tts-yati-v0.4'")],
    'hf/example.py': [('lseanlon/suaradee-v0.3', 'Suarify/kecil-tts-yati-v0.4')],
    'hf/example_node.mjs': [('lseanlon/suaradee-v0.3', 'Suarify/kecil-tts-yati-v0.4')],
    'hf/space_gradio/app.py': [('lseanlon/suaradee-v0.3', 'Suarify/kecil-tts-yati-v0.4')],
}.items():
    s = open(p, encoding='utf-8').read()
    for a, b in subs:
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)
    print('patched', p)

# 4) mirror the deployed space files into the repo
for f in ('index.html', 'kecil.js', 'suaradee.js'):
    shutil.copy(os.path.join('runs', 'space_live', f), os.path.join('hf', 'space_web', f))
print('space mirrored')
