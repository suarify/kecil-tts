"""Publish Suaradee v0.1 and v0.2 plus the web Space under another Hugging Face account.

    python scripts/publish_hf.py --user Suarify --profile "Suarify Malaysia"

The token is read from the local Hugging Face stored-token profile (never printed). Everything that names
`lseanlon/` (cards, examples, the web demo, the Python package's default repo) is retargeted to --user,
and the wheel is rebuilt so `Suaradee.from_pretrained()` defaults to the new account's repo.
"""
import argparse
import configparser
import os
import re
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--user', required=True)
ap.add_argument('--profile', required=True)
ap.add_argument('--dry-run', action='store_true')
ap.add_argument('--only', default=None, help='comma list of versions, e.g. 0.4')
ap.add_argument('--skip-space', action='store_true', help='publish model repos only (the Space may be owned elsewhere)')
A = ap.parse_args()
OLD = 'lseanlon'

cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
token = cp[A.profile]['hf_token']
api = HfApi(token=token)
who = api.whoami()
assert who['name'] == A.user, f"token belongs to {who['name']}, not {A.user}"
print('authenticated as', who['name'], '(token not shown)')

VERSIONS = [
    ('0.1', 'runs/k09/ms_int8_gpu.onnx', 'hf/model_card_v0.1.md', 'runs/k09'),
    ('0.2', 'runs/k12/ms_A_int8_gpu.onnx', 'hf/model_card_v0.2.md', 'runs/k12'),
    ('0.3', 'runs/k14_final/ms_clean_webgpu_int8.onnx', 'hf/model_card_v0.3.md', 'runs/k14_final'),
    ('0.4', 'runs/k16_final/ms_c4_int8.onnx', 'hf/model_card_v0.4.md', 'runs/k16_final'),
    ('0.4.1', 'runs/k16_final/fixh_export/paradee_int8_webgpu.onnx', 'hf/model_card_v0.4.1.md', 'runs/k16_final/fixh_export'),
]
if A.only:
    keep = {v.strip() for v in A.only.split(',')}
    VERSIONS = [v for v in VERSIONS if v[0] in keep]
TEXT = ('.md', '.py', '.js', '.mjs', '.html', '.toml', '.txt', '.json')
OUT = os.path.join(ROOT, 'hf_' + A.user.lower())
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)


def retarget(folder):
    n = 0
    for d, _, fs in os.walk(folder):
        for f in fs:
            if f.endswith(TEXT) and not f.endswith('.onnx'):
                p = os.path.join(d, f)
                try:
                    s = open(p, encoding='utf-8').read()
                except UnicodeDecodeError:
                    continue
                if OLD + '/' in s:
                    open(p, 'w', encoding='utf-8').write(s.replace(OLD + '/', A.user + '/'))
                    n += 1
    return n


def build_wheel(version, repo, dst):
    tmp = tempfile.mkdtemp()
    shutil.copytree(os.path.join(ROOT, 'python'), os.path.join(tmp, 'python'), ignore=shutil.ignore_patterns('build', 'dist', '*.egg-info', '__pycache__', 'tests'))
    p = os.path.join(tmp, 'python', 'suaradee', 'tts.py')
    s = open(p, encoding='utf-8').read()
    s = re.sub(r"REPO = '[^']+'", f"REPO = '{repo}'", s)
    open(p, 'w', encoding='utf-8').write(s)
    p = os.path.join(tmp, 'python', 'pyproject.toml')
    s = open(p, encoding='utf-8').read()
    wver = version if version.count('.') == 2 else version + '.0'
    open(p, 'w', encoding='utf-8').write(re.sub(r'version = "[^"]+"', f'version = "{wver}"', s))
    os.makedirs(dst, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', os.path.join(tmp, 'python'), '--no-deps', '-w', dst, '-q'], check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)


for ver, onnx, card, samples in VERSIONS:
    env = dict(os.environ, PYTHONUTF8='1')
    subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'build_hf.py'), '--onnx', os.path.join(ROOT, onnx), '--version', ver,
                    '--card', os.path.join(ROOT, card), '--samples-from', samples], check=True, env=env, capture_output=True)
    dst = os.path.join(OUT, f'model_v{ver}')
    shutil.copytree(os.path.join(ROOT, 'hf', 'model'), dst)
    shutil.rmtree(os.path.join(dst, 'dist'), ignore_errors=True)
    build_wheel(ver, f'{A.user}/kecil-tts-yati-v{ver}', os.path.join(dst, 'dist'))
    print(f'v{ver}: retargeted', retarget(dst), 'text files;', os.listdir(os.path.join(dst, 'dist')))
if not A.skip_space:
    shutil.copytree(os.path.join(ROOT, 'hf', 'space_web'), os.path.join(OUT, 'space_web'))
    print('web space: retargeted', retarget(os.path.join(OUT, 'space_web')), 'files')

for d, _, fs in os.walk(OUT):
    for f in fs:
        if f.endswith(TEXT):
            assert OLD not in open(os.path.join(d, f), encoding='utf-8', errors='ignore').read(), f'{OLD} still in {f}'
print('no mention of', OLD, 'left in the upload folders')
if A.dry_run:
    sys.exit(0)

for ver, *_ in VERSIONS:
    repo = f'{A.user}/kecil-tts-yati-v{ver}'
    api.create_repo(repo, repo_type='model', exist_ok=True)
    print(api.upload_folder(folder_path=os.path.join(OUT, f'model_v{ver}'), repo_id=repo, repo_type='model',
                            commit_message=f'Kecil-TTS Yati v{ver}: 8.5 MB Malay TTS (ONNX int8), Python and transformers.js'))
sp = f'{A.user}/suaradee-web'
if not A.skip_space:
    api.create_repo(sp, repo_type='space', space_sdk='static', exist_ok=True)
    print(api.upload_folder(folder_path=os.path.join(OUT, 'space_web'), repo_id=sp, repo_type='space',
                            commit_message='Suaradee web demo (WebGPU/WASM)'))
else:
    print('space skipped (--skip-space): model repos only')
