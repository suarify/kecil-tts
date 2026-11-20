import os, re, soundfile as sf
s = open('runs/listen/nicole_v05_gate.html', encoding='utf-8').read()
files = re.findall(r'data-src="([^"]+)"', s)
bad = [f for f in files if not os.path.exists(os.path.join('runs/listen', f))]
print('SRC', len(files), '| bad', bad[:3])
print('labels', 'Nicole v0.5 (new)' in s, 'Nicole v0.1 (old)' in s, 'Yati v0.5' in s)
