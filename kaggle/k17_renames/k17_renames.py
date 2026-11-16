# Rescue kernel (CPU only): kmb-15-cbgen2's output has 559 raw Chatterbox wavs named
# '<id>#<seed>.wav'. Kaggle's output download truncates URLs at '#', so they arrive as 0-byte
# files. Kernel sources mount the previous kernel's output on disk, where '#' is just a character:
# copy them to safe names ('<id>_<seed>.wav') and re-emit, so they can be downloaded and used.
import glob
import os
import shutil

OUT = '/kaggle/working'
os.makedirs(f'{OUT}/raw2', exist_ok=True)
src = glob.glob('/kaggle/input/kmb-15-cbgen2/raw2/*.wav')
print('mounted wavs:', len(src), flush=True)

n = bad = 0
for p in src:
    if os.path.getsize(p) == 0:
        bad += 1
        continue
    shutil.copy(p, f'{OUT}/raw2/' + os.path.basename(p).replace('#', '_'))
    n += 1
for f in ('gen_plan.csv', 'scores_all.csv', 'gen_report.json'):
    q = f'/kaggle/input/kmb-15-cbgen2/{f}'
    if os.path.exists(q):
        shutil.copy(q, f'{OUT}/{f}')
print('copied', n, 'wavs; empty on disk:', bad, flush=True)
print('sample:', sorted(os.listdir(f'{OUT}/raw2'))[:3], flush=True)
print('total MB:', round(sum(os.path.getsize(os.path.join(OUT, 'raw2', f)) for f in os.listdir(f'{OUT}/raw2')) / 1e6, 1),
      flush=True)
