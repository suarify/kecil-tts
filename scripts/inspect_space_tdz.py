s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i in range(268, 282): print(i+1, s[i][:200])
print('---')
for i in range(310, 322): print(i+1, s[i][:200])
print('--- tts decl')
for i, l in enumerate(s, 1):
    if 'let tts' in l or 'let lastAudio' in l or 'window.addEventListener' in l:
        print(i, l[:200])
