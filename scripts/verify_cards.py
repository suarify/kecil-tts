from huggingface_hub import hf_hub_download
import re
REPOS = ['Suarify/kecil-tts-yati-v0.5.0', 'Suarify/kecil-tts-yati-v0.4.1',
         'Suarify/kecil-tts-michael-v0.3.1', 'Suarify/kecil-tts-nicole-v0.1',
         'Suarify/kecil-tts-firdaus-v0.1']
for r in REPOS:
    p = hf_hub_download(r, 'README.md', force_download=True)
    s = open(p, encoding='utf-8').read()
    from_suaradee = 'from suaradee import SAMPLE_RATE, Suaradee' in s
    stale = ('from kecil_tts import' in s) or ('KecilTTS' in s)
    wheel_ok = bool(re.search(r'dist/(suaradee|kecil_tts_8m)-[0-9.]+-py3-none-any\.whl', s))
    print(r.split('/')[-1], '| suaradee import:', from_suaradee, '| stale KecilTTS:', stale, '| wheel ref ok:', wheel_ok)
