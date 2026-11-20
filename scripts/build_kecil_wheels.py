"""Rebuild 'kecil-tts' wheels for all 5 model repos (from the renamed local python/ package).

For each repo: set REPO default to that repo, set version per repo, build wheel,
upload to dist/ as kecil-tts-<ver>-py3-none-any.whl. Keep old suaradee wheel in place (no deletion).
"""
import configparser
import os
import re
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, list_repo_files

cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
REPOS = {
    'Suarify/kecil-tts-yati-v0.5.0': '0.5.0',
    'Suarify/kecil-tts-yati-v0.4.1': '0.4.1',
    'Suarify/kecil-tts-michael-v0.3.1': '0.3.1',
    'Suarify/kecil-tts-nicole-v0.1': '0.1.0',
    'Suarify/kecil-tts-firdaus-v0.1': '0.1.0',
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

for repo, ver in REPOS.items():
    tmp = f'runs/kecilwheel_{repo.split("/")[-1]}'
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree('python', f'{tmp}/python',
                    ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
    # point default REPO at this repo
    p = f'{tmp}/python/kecil_tts/tts.py'
    s = open(p, encoding='utf-8').read()
    s = re.sub(r"REPO\s*=\s*['\"][^'\"]+['\"]", f"REPO = '{repo}'", s)
    open(p, 'w', encoding='utf-8').write(s)
    # version
    p = f'{tmp}/python/pyproject.toml'
    s = open(p, encoding='utf-8').read()
    s = re.sub(r'version = "[^"]+"', f'version = "{ver}"', s)
    open(p, 'w', encoding='utf-8').write(s)

    outdir = f'{tmp}/dist'
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run([sys.executable, '-m', 'pip', 'wheel', f'{tmp}/python', '--no-deps', '-w', outdir, '-q'],
                       env=dict(os.environ, PYTHONUTF8='1'), capture_output=True, text=True)
    if r.returncode:
        print('BUILD FAIL', repo, r.stderr[-500:]); continue
    whl = [f for f in os.listdir(outdir) if f.endswith('.whl')][0]
    api.upload_file(path_or_fileobj=f'{outdir}/{whl}', path_in_repo=f'dist/{whl}', repo_id=repo, repo_type='model',
                    commit_message=f'Add renamed kecil-tts {ver} wheel (import kecil_tts / KecilTTS)')
    print('uploaded', repo, whl)
print('DONE')
