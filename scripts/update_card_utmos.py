import configparser
import os

from huggingface_hub import HfApi, hf_hub_download

cp = configparser.ConfigParser()
cp.read(os.path.expanduser('~/.cache/huggingface/stored_tokens'))
api = HfApi(token=cp['Suarify Malaysia']['hf_token'])
repo = 'Suarify/kecil-tts-yati-v0.5.0'
p = hf_hub_download(repo, 'README.md', repo_type='model')
s = open(p, encoding='utf-8').read()

old = """| Model | UTMOS mean | UTMOS p10 |
|---|---|---|
| Chatterbox teacher clips (ceiling) | 3.553 | 3.162 |
| **Yati v0.5 (this model)** | **2.275** | **1.793** |
| Yati v0.4.1 | 2.158 | 1.738 |
| Nicole v0.1 | — (not scored) | — |
| Michael v0.3.1 | 1.813 | 1.445 |
"""

new = """| Model | UTMOS mean | UTMOS p10 |
|---|---|---|
| Chatterbox teacher clips (ceiling) | 3.553 | 3.162 |
| **Yati v0.5 (this model)** | **2.275** | **1.793** |
| Yati v0.4.1 | 2.158 | 1.738 |
| Michael v0.3.1 | 1.813 | 1.445 |

Full development leaderboard (80-sentence eval set, UTMOS22-strong mean / 10th percentile):

| Candidate | UTMOS mean | UTMOS p10 | Released? |
|---|---|---|---|
| Chatterbox teacher (ceiling) | 3.553 | 3.162 | — |
| **P3 → v0.5 (adversarial decoder)** | **2.275** | **1.793** | ✅ this model |
| P1 (log-F0 + voiced/unvoiced text head) | 2.180 | 1.765 | ❌ no audible gain |
| Yati v0.4.1 (baseline) | 2.158 | 1.738 | ✅ previous release |
| P2 (GTA decoder fine-tune) | 2.010 | 1.540 | ❌ worse than baseline |

The teacher ceiling is ~1.28 UTMOS above this model. "Better" here is a small, measured +0.12 over v0.4.1; in the owner's blind loudness-matched test the difference was not clearly perceptible. A Phase-4 data-expansion attempt was evaluated on 50 fresh sentences: general (2.2%) and conversational (1.9%) teacher clips met the 2.5% CER gate, but numbers (43.8%) and code-switch (10.8%) did not, so the noisy data was not used.
"""
assert old in s, 'benchmark table not found'
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
api.upload_file(path_or_fileobj=p, path_in_repo='README.md', repo_id=repo, repo_type='model',
                commit_message='Add full UTMOS leaderboard and Phase 4 CER result to model card')
print('card updated:', repo)
