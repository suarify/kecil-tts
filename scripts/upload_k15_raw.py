"""Package the rescued kmb-15 raw wavs as dataset seanlon/kmb-15-raw (safe filenames, no '#')."""
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'runs', 'k15_rescue', 'raw2')
DST = os.path.join(ROOT, 'kaggle', 'ds_k15raw', 'raw2')
os.makedirs(DST, exist_ok=True)
for f in os.listdir(DST):
    os.remove(os.path.join(DST, f))
n = 0
for f in sorted(os.listdir(SRC)):
    if not f.endswith('.wav'):
        continue
    p = os.path.join(SRC, f)
    if os.path.getsize(p) < 10000:
        print('skip tiny', f, os.path.getsize(p))
        continue
    shutil.copy(p, os.path.join(DST, f.replace('#', '_')))
    n += 1
mb = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)) / 1e6
print(f'packaged {n} wavs, {mb:.1f} MB -> {DST}')
created = os.path.exists(os.path.join(ROOT, 'runs', '.k15raw_created'))
cmd = ['kaggle', 'datasets', 'version' if created else 'create']
if created:
    cmd += ['-m', 'more rescued raw clips']
cmd += ['-p', os.path.join(ROOT, 'kaggle', 'ds_k15raw'), '--dir-mode', 'zip']
print(' '.join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
print((r.stdout + r.stderr)[-600:])
if r.returncode == 0:
    open(os.path.join(ROOT, 'runs', '.k15raw_created'), 'w').write('1')
