from huggingface_hub import hf_hub_download
p = hf_hub_download('Suarify/suaradee-web', 'index.html', repo_type='space', force_download=True)
s = open(p, encoding='utf-8').read()
print('v0.5 dropdown', 'Suarify/kecil-tts-yati-v0.5.0' in s)
print('nicole speaker', "nicole: { name: 'Nicole'" in s)
print('waveform', 'id="wave"' in s)
print('stats panel', 'id="stRt"' in s)
print('auto-speak chips', "$('go').disabled) $('go').click()" in s)
