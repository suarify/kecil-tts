"""Fix the stale examples in all published model repos: use the real 'suaradee' module + correct wheel/repo."""
import os

from huggingface_hub import HfApi, list_repo_files

REPOS = {
    'Suarify/kecil-tts-yati-v0.5.0': ('suaradee-0.5.0', 'Suarify/kecil-tts-yati-v0.5.0'),
    'Suarify/kecil-tts-yati-v0.4.1': ('suaradee-0.4.1', 'Suarify/kecil-tts-yati-v0.4.1'),
    'Suarify/kecil-tts-michael-v0.3.1': (None, 'Suarify/kecil-tts-michael-v0.3.1'),
    'Suarify/kecil-tts-nicole-v0.1': ('suaradee-0.1.0', 'Suarify/kecil-tts-nicole-v0.1'),
    'Suarify/kecil-tts-firdaus-v0.1': ('suaradee-0.1.0', 'Suarify/kecil-tts-firdaus-v0.1'),
}
import configparser
cp = configparser.ConfigParser(); cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
from huggingface_hub import hf_hub_download

PY_EX = '''"""Minimal Python example for {repo} (8.5 MB Malay TTS)."""
# pip install onnxruntime numpy huggingface_hub soundfile
# pip install https://huggingface.co/{repo}/resolve/main/dist/{wheel}-py3-none-any.whl
import soundfile as sf
from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained("{repo}")
audio = tts("Selamat pagi, boleh saya bantu encik hari ini?")
sf.write("hello.wav", audio, SAMPLE_RATE)
print(tts.phonemize("Jumlah bayaran anda ialah RM1,250.50."))
'''
STREAM_EX = '''"""Streaming Python example for {repo}."""
# pip install onnxruntime numpy huggingface_hub soundfile
# pip install https://huggingface.co/{repo}/resolve/main/dist/{wheel}-py3-none-any.whl
import soundfile as sf
from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained("{repo}")
parts = []
for chunk in tts.stream("Selamat pagi, boleh saya bantu encik hari ini?"):
    parts.append(chunk)
audio = __import__("numpy").concatenate(parts)
sf.write("hello.wav", audio, SAMPLE_RATE)
'''
NODE_EX = '''// Node example for {repo}. Install: npm i @huggingface/transformers
import {{ Suaradee }} from '../../js/suaradee.js';
const tts = await Suaradee.from_pretrained("{repo}");
const audio = await tts.speak("Selamat pagi, boleh saya bantu encik hari ini?");
console.log("samples:", audio.audio.length, "sr:", audio.sampleRate);
'''

for repo, (wheel, _) in REPOS.items():
    if wheel is None:
        files = [f for f in list_repo_files(repo) if f.startswith('examples/')]
        if not files:
            print('skip', repo, 'no examples / no wheel known'); continue
        # find wheel name from repo files
        ws = [f.split('/')[-1][:-4] for f in list_repo_files(repo) if f.startswith('dist/')]
        wheel = ws[0] if ws else 'suaradee-0.1.0'
    targets = {
        'examples/python_example.py': PY_EX,
        'examples/stream_python.py': STREAM_EX,
        'examples/node_example.mjs': NODE_EX,
        'examples/stream_node.mjs': NODE_EX,
    }
    for path, tmpl in targets.items():
        try:
            hf_hub_download(repo, path)   # only replace if exists
        except Exception:
            continue
        txt = tmpl.format(repo=repo, wheel=wheel)
        tmp = f'runs/_ex_{repo.split("/")[-1]}_{path.split("/")[-1]}'
        open(tmp, 'w', encoding='utf-8').write(txt)
        api.upload_file(path_or_fileobj=tmp, path_in_repo=path, repo_id=repo, repo_type='model',
                        commit_message='Fix example to use suaradee module + correct wheel/repo')
        print('fixed', repo, path)
print('DONE')
