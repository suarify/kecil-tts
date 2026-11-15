"""Re-upload only the web Space (after a copy tweak) without rebuilding model repos."""
import configparser
import os
import shutil

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])

src = os.path.join(ROOT, 'hf', 'space_web')
tmp = os.path.join(ROOT, 'hf_suarify', 'space_web')
shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(src, tmp)
for d, _, fs in os.walk(tmp):
    for f in fs:
        if f.endswith(('.html', '.js', '.mjs', '.md', '.json')):
            p = os.path.join(d, f)
            s = open(p, encoding='utf-8').read().replace('lseanlon/', 'Suarify/')
            open(p, 'w', encoding='utf-8').write(s)
api.upload_folder(folder_path=tmp, repo_id='Suarify/suaradee-web', repo_type='space',
                  commit_message='Space copy: v0.3 is the accuracy release')
print('space updated')
