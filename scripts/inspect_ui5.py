s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
keys = ['<header', 'class="brand', 'h1', 'class="topbar', 'id="lang', 'themeBtn', 'class="headrow',
        'id="stats"', 'id="wave"', 'id="edge"', 'Runs anywhere', 'Keywords', 'class="tags',
        'Need custom software', 'Contact Suarify', 'class="card cta', '<footer', '</main>',
        'id="info"', 'function speedStrip', 'drawWave(', 'id="fxd"']
for i, l in enumerate(s, 1):
    if any(k in l for k in keys):
        print(i, l[:230])
