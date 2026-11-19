s = open('runs/p1_eval.log', encoding='utf-8', errors='replace').read()
s = ''.join(c for c in s if c != '\x00')
lines = [l.strip() for l in s.splitlines() if l.strip()]
for l in lines[-40:]:
    print(l[:300])
