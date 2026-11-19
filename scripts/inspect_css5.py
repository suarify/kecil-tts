s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if any(k in l for k in ('headrow{', 'topbar{', '.tg{', '@media', '.edge{', '.tags{', '.cta', 'footer{', '.card{')):
        print(i, l[:260])
