import os
import subprocess
tracked = subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.splitlines()
big = []
for f in tracked:
    try:
        sz = os.path.getsize(f)
        if sz > 100 * 1024 * 1024:
            big.append((f, round(sz / 1024 / 1024, 1)))
    except OSError:
        pass
for f, mb in sorted(big, key=lambda x: -x[1]):
    print(mb, 'MB', f)
print('count', len(big))
