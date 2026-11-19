import os, re, soundfile as sf
s = open('runs/listen/sota_p3.html', encoding='utf-8').read()
files = re.findall(r'data-src="([^"]+)"', s)
print('SRC count', len(files), '| sample', files[:3])
bad = []
for f in files:
    p = os.path.join('runs/listen', f)
    if not os.path.exists(p):
        bad.append(f)
        continue
    y, sr = sf.read(p)
    if len(y) == 0 or sr != 24000:
        bad.append(f)
print('bad', bad[:5])
print('label P3 present', "'P3'" in s)
