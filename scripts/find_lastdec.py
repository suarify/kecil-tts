s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if 'lastAudio' in l and ('let ' in l or 'var ' in l or 'let ' in l):
        print(i, l[:180])
    if l.strip().startswith('let tts') or l.strip().startswith('let lastAudio'):
        print('DECL', i, l[:180])
print('--- top of module ---')
for i in range(224, 232): print(i+1, s[i][:160])
