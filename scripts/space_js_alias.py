"""Space JS: add KecilTTS alias to kecil.js, keep Suaradee. index.html keeps working (uses Suaradee)."""
import os

for p in ('runs/local_v041/kecil.js', 'runs/local_v041/suaradee.js'):
    if not os.path.exists(p):
        continue
    s = open(p, encoding='utf-8').read()
    if 'export const KecilTTS' not in s and 'export class KecilTTS' not in s:
        s = s.rstrip() + '\n\n// Renamed alias (Suaradee kept for backwards compatibility).\nexport const KecilTTS = Suaradee;\n'
        open(p, 'w', encoding='utf-8').write(s)
        print('alias added to', p)
    else:
        print('alias exists', p)
