"""Add Firdaus v0.1 as a fourth speaker in the Space (third male voice)."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
assert 'Suarify/kecil-tts-firdaus-v0.1' not in s
old = "  nicole: { name: 'Nicole', sub: 'female voice', card: 'https://huggingface.co/Suarify/kecil-tts-nicole-v0.1',"
new = """  firdaus: { name: 'Firdaus', sub: 'male voice · early release', card: 'https://huggingface.co/Suarify/kecil-tts-firdaus-v0.1',
    note: 'Male voice built from Yati v0.5. v0.1 sounds robotic - an early release, published for feedback.',
    versions: [['Suarify/kecil-tts-firdaus-v0.1', 'v0.1']] },
  nicole: { name: 'Nicole', sub: 'female voice', card: 'https://huggingface.co/Suarify/kecil-tts-nicole-v0.1',"""
assert old in s
s = s.replace(old, new, 1)
# speaker auto-detect from model id
s = s.replace("(qs.get('model') || '').includes('nicole') ? 'nicole' : 'yati'",
              "(qs.get('model') || '').includes('nicole') ? 'nicole' : (qs.get('model') || '').includes('firdaus') ? 'firdaus' : 'yati'", 1)
# speed map entry
s = s.replace("  'Suarify/kecil-tts-nicole-v0.1': {rt: 19, label: 'nicole v0.1', base: 21, baseLabel: 'Yati v0.4.1'},",
              "  'Suarify/kecil-tts-nicole-v0.1': {rt: 19, label: 'nicole v0.1', base: 21, baseLabel: 'Yati v0.4.1'},\n  'Suarify/kecil-tts-firdaus-v0.1': {rt: 19, label: 'Firdaus v0.1', base: 19, baseLabel: 'Michael v0.3.1'},", 1)
open(P, 'w', encoding='utf-8').write(s)
print('Firdaus added to Space dropdown')
