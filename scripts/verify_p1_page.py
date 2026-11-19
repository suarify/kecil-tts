import glob, os, soundfile as sf
fs = sorted(glob.glob('runs/listen/sota_p1/*.wav'))
print('wav files:', len(fs))
bad = []
for f in fs:
    try:
        y, sr = sf.read(f)
        if len(y) == 0 or sr != 24000:
            bad.append((f, len(y), sr))
    except Exception as e:
        bad.append((f, str(e)[:60]))
print('bad:', bad[:5])
print('total MB:', round(sum(os.path.getsize(f) for f in fs)/1e6, 2))
