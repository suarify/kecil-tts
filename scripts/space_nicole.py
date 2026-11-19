"""Add nicole v0.1 to the Space's voice dropdown."""
p = 'runs/local_v041/index.html'
s = open(p, encoding='utf-8').read()
old = "versions: [['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1 (latest)']"
new = "versions: [['Suarify/kecil-tts-nicole-v0.1', 'nicole v0.1 (new)'], ['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1 (latest)']"
assert old in s
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('nicole added to space dropdown')
