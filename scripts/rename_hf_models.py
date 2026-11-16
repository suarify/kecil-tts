"""Rename the Suarify model repos to the Kecil-TTS scheme and fix every reference in their cards.

  suaradee-v0.X          -> kecil-tts-yati-v0.X     (Yati, female)
  suaradee-michael-v0.X  -> kecil-tts-michael-v0.X
Card text: links + title + prose, lseanlon -> Suarify. The Space is NOT renamed (its URL is public
and referenced elsewhere); only the model ids change.
"""
import configparser
import os
import re
import tempfile

from huggingface_hub import HfApi, hf_hub_download

cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])

RENAMES = [(f'Suarify/suaradee-v0.{v}', f'Suarify/kecil-tts-yati-v0.{v}') for v in (1, 2, 3, 4)]
RENAMES += [(f'Suarify/suaradee-michael-v0.{v}', f'Suarify/kecil-tts-michael-v0.{v}') for v in (1, 2)]

for old, new in RENAMES:
    try:
        api.move_repo(from_id=old, to_id=new, repo_type='model')
        print('moved', old, '->', new)
    except Exception as e:
        print('MOVE FAILED', old, str(e)[:160])


def fix_card(repo, is_michael):
    p = hf_hub_download(repo, 'README.md', token=cp['Suarify Malaysia']['hf_token'])
    s = open(p, encoding='utf-8').read()
    s = s.replace('lseanlon/', 'Suarify/')
    s = s.replace('suaradee-michael-v', 'kecil-tts-michael-v')
    s = s.replace('suaradee-v', 'kecil-tts-yati-v')
    voice = 'Michael' if is_michael else 'Yati'
    s = re.sub(r'^# Suaradee v([0-9.]+)', rf'# Kecil-TTS {voice} v\1', s, flags=re.M)
    s = s.replace('Suaradee v', f'Kecil-TTS {voice} v')
    s = s.replace('Suaradee', 'Kecil-TTS')
    api.upload_file(path_or_fileobj=s.encode('utf-8'), path_in_repo='README.md', repo_id=repo,
                    commit_message='Rename to Kecil-TTS and fix links')
    print('card updated', repo)


for old, new in RENAMES:
    fix_card(new, 'michael' in new)

print('\nfinal model list:')
for m in api.list_models(author='Suarify', limit=50):
    if 'kecil' in m.id or 'suaradee' in m.id:
        print(' ', m.id)
