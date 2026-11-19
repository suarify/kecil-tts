s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i in range(238, 246): print(i+1, repr(s[i][:200]))
