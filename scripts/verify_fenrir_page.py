import os, re, soundfile as sf
s = open('runs/listen/firdaus_gate.html', encoding='utf-8').read()
files = re.findall(r'data-src="([^"]+)"', s)
bad = [f for f in files if not os.path.exists(os.path.join('runs/listen', f))]
print('SRC', len(files), '| bad', bad[:4])
print('labels', "'Firdaus'" in s, "'Yati v0.5'" in s, "'Nicole'" in s)
