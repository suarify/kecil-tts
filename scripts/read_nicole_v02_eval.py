s = open('runs/nicole_v02_eval.log', encoding='utf-8', errors='replace').read().replace('\x00', '')
i = s.find('"tag": "nicole_v02"')
print(s[i:i+420] if i >= 0 else s[-500:])
