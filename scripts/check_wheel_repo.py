from huggingface_hub import hf_hub_download
import zipfile, re
repos = [('Suarify/kecil-tts-nicole-v0.1','suaradee-0.1.0'),
         ('Suarify/kecil-tts-yati-v0.5.0','suaradee-0.5.0'),
         ('Suarify/kecil-tts-firdaus-v0.1','suaradee-0.1.0'),
         ('Suarify/kecil-tts-michael-v0.3.1','suaradee-0.3.0')]
for r, w in repos:
    try:
        p = hf_hub_download(r, f'dist/{w}-py3-none-any.whl')
        z = zipfile.ZipFile(p)
        t = z.read('suaradee/tts.py').decode('utf-8', errors='replace')
        print(r.split('/')[-1], '| REPO:', re.findall(r"REPO\s*=\s*['\"]([^'\"]+)", t)[:1])
    except Exception as e:
        print(r.split('/')[-1], 'ERR', type(e).__name__, str(e)[:70])
