s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
keys = ['canvas', 'waveform', 'speedstrip', 'spd1', 'spd2', 'spdbar', 'SPEAKERS', 'versions:',
        'note:', 'chips', 'id="info', 'firstT', 'speedStrip(', '<section', 'hbench']
for i, l in enumerate(s, 1):
    if any(k in l for k in keys):
        print(i, l[:260])
