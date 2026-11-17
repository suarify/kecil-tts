"""Point the Space at Michael v0.3.1 and refresh the benchmark column label."""
p = 'runs/local_v041/index.html'
s = open(p, encoding='utf-8').read()

old = ("versions: [['Suarify/kecil-tts-michael-v0.3.0', 'v0.3.0 (latest, retrained)'], "
       "['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2'], ['Suarify/kecil-tts-michael-v0.2.1', 'v0.2.1'], "
       "['Suarify/kecil-tts-michael-v0.2', 'v0.2'], ['Suarify/kecil-tts-michael-v0.1', 'v0.1']]")
new = ("versions: [['Suarify/kecil-tts-michael-v0.3.1', 'v0.3.1 (latest)'], "
       "['Suarify/kecil-tts-michael-v0.3.0', 'v0.3.0'], ['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2'], "
       "['Suarify/kecil-tts-michael-v0.2', 'v0.2'], ['Suarify/kecil-tts-michael-v0.1', 'v0.1']]")
assert old in s
s = s.replace(old, new)

s = s.replace("card: 'https://huggingface.co/Suarify/kecil-tts-michael-v0.3.0'",
              "card: 'https://huggingface.co/Suarify/kecil-tts-michael-v0.3.1'")
s = s.replace('Michael v0.3.0</a>', 'Michael v0.3.1</a>')
s = s.replace('https://huggingface.co/Suarify/kecil-tts-michael-v0.3.0" target="_blank" rel="noopener">Michael v0.3.1',
              'https://huggingface.co/Suarify/kecil-tts-michael-v0.3.1" target="_blank" rel="noopener">Michael v0.3.1')
s = s.replace('<th>Michael v0.3.0 (male)</th>', '<th>Michael v0.3.1 (male)</th>')
s = s.replace('note: \'First male voice. v0.2.2 has brighter, less watery sound and steadier timing than v0.2 (same voice, better audio export).\'',
              'note: \'Male voice. v0.3.1 fixes the slow timing of v0.3.0 and uses the training-faithful synthesis (same weights, native speed).\'')
open(p, 'w', encoding='utf-8').write(s)
print('space updated:', 'michael-v0.3.1' in s)
