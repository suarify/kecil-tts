"""Final verification of the suaradee -> kecil-tts rename across all repos."""
from huggingface_hub import hf_hub_download, list_repo_files
import zipfile, re
REPOS = {
    'Suarify/kecil-tts-yati-v0.5.0': '0.5.0',
    'Suarify/kecil-tts-yati-v0.4.1': '0.4.1',
    'Suarify/kecil-tts-michael-v0.3.1': '0.3.1',
    'Suarify/kecil-tts-nicole-v0.1': '0.1.0',
    'Suarify/kecil-tts-firdaus-v0.1': '0.1.0',
}
for r, ver in REPOS.items():
    fs = list_repo_files(r)
    whl = f'dist/kecil_tts-{ver}-py3-none-any.whl'
    has_new = whl in fs
    # verify wheel internals
    top = ''
    if has_new:
        p = hf_hub_download(r, whl)
        z = zipfile.ZipFile(p)
        top = sorted(set(n.split('/')[0] for n in z.namelist() if not n.startswith('kecil_tts-')))[:3]
    # card
    p = hf_hub_download(r, 'README.md', force_download=True)
    s = open(p, encoding='utf-8').read()
    card_ok = 'from kecil_tts import SAMPLE_RATE, KecilTTS' in s and 'KecilTTS.from_pretrained' in s
    stale = 'from suaradee import' in s or 'Suaradee.from_pretrained' in s
    print(r.split('/')[-1], '| new wheel', has_new, top, '| card new API', card_ok, '| stale', stale)
