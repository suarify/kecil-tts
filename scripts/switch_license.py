"""Switch all 5 published model repos from Apache-2.0 to CC BY 4.0 (card + LICENSE file)."""
import configparser
import os
import re

import requests
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])

REPOS = ['Suarify/kecil-tts-yati-v0.5.0', 'Suarify/kecil-tts-yati-v0.4.1',
         'Suarify/kecil-tts-michael-v0.3.1', 'Suarify/kecil-tts-nicole-v0.1',
         'Suarify/kecil-tts-firdaus-v0.1']
# official CC BY 4.0 legal code
legal = requests.get('https://creativecommons.org/licenses/by/4.0/legalcode.txt', timeout=30).text
assert 'Attribution 4.0 International' in legal and len(legal) > 10000
LICENSE_TEXT = ("Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
                "=============================================================\n\n"
                "You are free to share and adapt this work, even for commercial purposes, "
                "provided you give appropriate credit, provide a link to the license, and "
                "indicate if changes were made.\n\n"
                "Full legal code:\nhttps://creativecommons.org/licenses/by/4.0/legalcode\n\n"
                "-------------------------------------------------------------------------\n\n" + legal)

for repo in REPOS:
    card_path = hf_hub_download(repo, 'README.md')
    s = open(card_path, encoding='utf-8').read()
    s = s.replace('license: apache-2.0', 'license: cc-by-4.0')
    s = s.replace('license: apache-2.0', 'license: cc-by-4.0')
    s = re.sub(r'license: apache-2\.0', 'license: cc-by-4.0', s)
    s = s.replace('license: apache-2.0', 'license: cc-by-4.0')
    # card body: swap the license sentence. Handle a few phrasings.
    s = s.replace('Code and weights: Apache-2.0.', 'Code and weights: CC BY 4.0 (Creative Commons Attribution 4.0 International).')
    s = s.replace('Code and weights: Apache-2.0', 'Code and weights: CC BY 4.0')
    s = s.replace('**Apache-2.0**', '**CC BY 4.0**')
    s = s.replace('Apache-2.0', 'CC BY 4.0')
    s = s.replace('apache-2.0', 'cc-by-4.0')
    open(card_path, 'w', encoding='utf-8').write(s)
    api.upload_file(path_or_fileobj=card_path, path_in_repo='README.md', repo_id=repo, repo_type='model',
                    commit_message='Switch license to CC BY 4.0')

    # LICENSE file: replace (or add) with CC BY 4.0 text
    tmp = f'runs/_ccby_license_{repo.split("/")[-1]}.txt'
    open(tmp, 'w', encoding='utf-8').write(LICENSE_TEXT)
    api.upload_file(path_or_fileobj=tmp, path_in_repo='LICENSE', repo_id=repo, repo_type='model',
                    commit_message='Add CC BY 4.0 license text')
    print('updated', repo)
print('DONE')
