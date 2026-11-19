s = open('runs/gta_smoke.log', encoding='utf-8', errors='replace').read()
s = s.replace('\x00', '')
lines = [l.strip() for l in s.splitlines() if l.strip()]
for l in lines[-35:]:
    print(l[:240])
