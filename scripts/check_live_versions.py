from huggingface_hub import hf_hub_download
p = hf_hub_download('Suarify/suaradee-web', 'index.html', repo_type='space', force_download=True)
s = open(p, encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if 'versions:' in l:
        print(i, l[:280])
