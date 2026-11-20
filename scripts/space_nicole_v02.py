"""Space: add Nicole v0.2 to her dropdown, update card link + note, fix version order."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
assert 'Suarify/kecil-tts-nicole-v0.2' not in s
old = """  nicole: { name: 'Nicole', sub: 'female voice', card: 'https://huggingface.co/Suarify/kecil-tts-nicole-v0.1',
    note: 'Female voice trained from Yati clips. v0.1 is an early release; a cleaner version is in progress.',
    versions: [['Suarify/kecil-tts-nicole-v0.1', 'v0.1']] }"""
new = """  nicole: { name: 'Nicole', sub: 'female voice', card: 'https://huggingface.co/Suarify/kecil-tts-nicole-v0.2',
    note: 'Female voice rebuilt from Yati v0.5 with fixed loudness and +25% volume (v0.2). v0.1 was an early, soft release.',
    versions: [['Suarify/kecil-tts-nicole-v0.2', 'v0.2 (latest)'], ['Suarify/kecil-tts-nicole-v0.1', 'v0.1']] }"""
assert old in s
s = s.replace(old, new, 1)
# speed map entry for v0.2
s = s.replace("  'Suarify/kecil-tts-nicole-v0.1': {rt: 19, label: 'nicole v0.1', base: 21, baseLabel: 'Yati v0.4.1'},",
              "  'Suarify/kecil-tts-nicole-v0.2': {rt: 19, label: 'Nicole v0.2', base: 21, baseLabel: 'Yati v0.4.1'},\n  'Suarify/kecil-tts-nicole-v0.1': {rt: 19, label: 'nicole v0.1', base: 21, baseLabel: 'Yati v0.4.1'},", 1)
# footer voice link
s = s.replace('Suarify/kecil-tts-nicole-v0.1"', 'Suarify/kecil-tts-nicole-v0.2"')
open(P, 'w', encoding='utf-8').write(s)
print('Space updated: Nicole v0.2')
