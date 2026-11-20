"""Update all 5 model cards + examples to the renamed kecil-tts package (with suaradee shim noted)."""
import configparser
import os
import re

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
REPOS = {
    'Suarify/kecil-tts-yati-v0.5.0': '0.5.0',
    'Suarify/kecil-tts-yati-v0.4.1': '0.4.1',
    'Suarify/kecil-tts-michael-v0.3.1': '0.3.1',
    'Suarify/kecil-tts-nicole-v0.1': '0.1.0',
    'Suarify/kecil-tts-firdaus-v0.1': '0.1.0',
}

PY_EX = '''"""Minimal Python example for {repo} (8.5 MB Malay TTS)."""
# pip install onnxruntime numpy huggingface_hub soundfile
# pip install https://huggingface.co/{repo}/resolve/main/dist/kecil_tts-{ver}-py3-none-any.whl
import soundfile as sf
from kecil_tts import SAMPLE_RATE, KecilTTS

tts = KecilTTS.from_pretrained("{repo}")
audio = tts("Selamat pagi, boleh saya bantu encik hari ini?")
sf.write("hello.wav", audio, SAMPLE_RATE)
print(tts.phonemize("Jumlah bayaran anda ialah RM1,250.50."))
'''
STREAM_EX = '''"""Streaming Python example for {repo}."""
# pip install onnxruntime numpy huggingface_hub soundfile
# pip install https://huggingface.co/{repo}/resolve/main/dist/kecil_tts-{ver}-py3-none-any.whl
import soundfile as sf
import numpy as np
from kecil_tts import SAMPLE_RATE, KecilTTS

tts = KecilTTS.from_pretrained("{repo}")
parts = []
for chunk in tts.stream("Selamat pagi, boleh saya bantu encik hari ini?"):
    parts.append(chunk)
sf.write("hello.wav", np.concatenate(parts), SAMPLE_RATE)
'''

for repo, ver in REPOS.items():
    # ---- card ----
    p = hf_hub_download(repo, 'README.md')
    s = open(p, encoding='utf-8').read()
    s = re.sub(r'dist/suaradee-[0-9.]+-py3-none-any\.whl', f'dist/kecil_tts-{ver}-py3-none-any.whl', s)
    s = s.replace('from suaradee import SAMPLE_RATE, Suaradee', 'from kecil_tts import SAMPLE_RATE, KecilTTS')
    s = s.replace('Suaradee.from_pretrained', 'KecilTTS.from_pretrained')
    tmp = f'runs/_cardk_{repo.split("/")[-1]}.md'
    open(tmp, 'w', encoding='utf-8').write(s)
    api.upload_file(path_or_fileobj=tmp, path_in_repo='README.md', repo_id=repo, repo_type='model',
                    commit_message='Card: use kecil-tts package (KecilTTS)')
    print('card', repo)

    # ---- examples ----
    fs = list_repo_files(repo)
    for path, tmpl in (('examples/python_example.py', PY_EX), ('examples/stream_python.py', STREAM_EX)):
        if path not in fs:
            continue
        t = tmpl.format(repo=repo, ver=ver)
        tmp = f'runs/_exk_{repo.split("/")[-1]}_{path.split("/")[-1]}'
        open(tmp, 'w', encoding='utf-8').write(t)
        api.upload_file(path_or_fileobj=tmp, path_in_repo=path, repo_id=repo, repo_type='model',
                        commit_message='Example: use kecil_tts / KecilTTS')
        print('  example', path)
print('DONE')
