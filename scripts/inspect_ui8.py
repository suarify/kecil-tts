s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if any(k in l for k in ('class="card bench"', 'data-i18n="hbench"', 'class="micro"', 'id="info"', 'id="ph"', 'nbench', '.micro{')):
        print(i, l[:200])
