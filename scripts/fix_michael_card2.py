"""Correct the Michael card: no wheel exists, so remove the pip wheel line; from_pretrained still works."""
import configparser
import os

from huggingface_hub import HfApi, hf_hub_download

cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
repo = 'Suarify/kecil-tts-michael-v0.3.1'
p = hf_hub_download(repo, 'README.md')
lines = open(p, encoding='utf-8').read().splitlines()
out = []
for l in lines:
    if 'resolve/main/dist/' in l and '.whl' in l:
        continue  # no wheel published for this repo
    out.append(l)
s = '\n'.join(out) + '\n'
tmp = 'runs/_michael_card2.md'
open(tmp, 'w', encoding='utf-8').write(s)
api.upload_file(path_or_fileobj=tmp, path_in_repo='README.md', repo_id=repo, repo_type='model',
                commit_message='Remove non-existent wheel install line from quick-start')
print('removed stale wheel line')
