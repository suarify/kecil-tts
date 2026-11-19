"""Check REVIEW.md Phase 2 Gate 2: UTMOS >= P1 + 0.1 (owner A/B is the alternative path)."""
import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else 'p2'
m = json.load(open(f'runs/sota/{tag}/metrics.json', encoding='utf-8'))
p1 = json.load(open('runs/sota/p1/metrics.json', encoding='utf-8'))

print(f'=== REVIEW Phase 2 Gate 2: {tag} vs P1 ===')
print(f"UTMOS mean: {m.get('utmos_mean')} vs P1 {p1.get('utmos_mean')}  (need >= {round(p1.get('utmos_mean', 0) + 0.1, 4)})")
print(f"UTMOS p10:  {m.get('utmos_p10')} vs P1 {p1.get('utmos_p10')}")
ok = m.get('utmos_mean', 0) >= p1.get('utmos_mean', 0) + 0.1
print('AUTOMATIC PATH:', 'PASS' if ok else 'FAIL')
print('OWNER PATH: build the blind P2-vs-P1 page and pass if P2 picked >= 8/12')
