"""Fix stale default repo in js/suaradee.js for all model repos."""
import configparser
import os
import re

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

REPOS = ['Suarify/kecil-tts-yati-v0.5.0', 'Suarify/kecil-tts-yati-v0.4.1',
         'Suarify/kecil-tts-michael-v0.3.1', 'Suarify/kecil-tts-nicole-v0.1',
         'Suarify/kecil-tts-firdaus-v0.1']
cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
for r in REPOS:
    fs = list_repo_files(r)
    if 'js/suaradee.js' not in fs:
        print('skip', r, '(no js/suaradee.js)'); continue
    p = hf_hub_download(r, 'js/suaradee.js')
    s = open(p, encoding='utf-8').read()
    new = re.sub(r'Suarify/kecil-tts-[a-z0-9.]+', r, s)
    if new == s:
        print('no change', r); continue
    tmp = f'runs/_js_{r.split("/")[-1]}.js'
    open(tmp, 'w', encoding='utf-8').write(new)
    api.upload_file(path_or_fileobj=tmp, path_in_repo='js/suaradee.js', repo_id=r, repo_type='model',
                    commit_message='Fix JS default repo to this model')
    print('fixed js', r)
print('DONE')
