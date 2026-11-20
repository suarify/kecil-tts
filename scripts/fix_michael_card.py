"""Fix the stale KecilTTS quick-start in the Michael card."""
import configparser
import os

from huggingface_hub import HfApi, hf_hub_download

cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
repo = 'Suarify/kecil-tts-michael-v0.3.1'
p = hf_hub_download(repo, 'README.md')
s = open(p, encoding='utf-8').read()
s = s.replace('dist/kecil_tts_8m-0.3.0-py3-none-any.whl', 'dist/suaradee-0.3.0-py3-none-any.whl')
s = s.replace('from kecil_tts import SAMPLE_RATE, KecilTTS', 'from suaradee import SAMPLE_RATE, Suaradee')
s = s.replace('KecilTTS.from_pretrained', 'Suaradee.from_pretrained')
tmp = 'runs/_michael_card.md'
open(tmp, 'w', encoding='utf-8').write(s)
api.upload_file(path_or_fileobj=tmp, path_in_repo='README.md', repo_id=repo, repo_type='model',
                commit_message='Fix quick-start: suaradee module + correct wheel')
print('fixed michael card')
