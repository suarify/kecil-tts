s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if any(k in l for k in ('infobtn', 'stTime', 'stFirst', 'stRt', 'firstT', 'Time taken', 'const t0', 'const s =', 'speedStrip(sec')):
        print(i, l[:230])
