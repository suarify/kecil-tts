"""Check REVIEW.md Phase 1 automatic Gate 1 against runs/sota/<tag>/metrics.json."""
import json
import sys

tag = sys.argv[1] if len(sys.argv) > 1 else 'p1'
m = json.load(open(f'runs/sota/{tag}/metrics.json', encoding='utf-8'))
base = json.load(open('runs/sota/yati_v041_held60/metrics.json', encoding='utf-8'))

checks = []


def chk(name, ok, detail):
    checks.append((name, bool(ok), detail))


chk('dropout <= 3%', (m.get('dropout') or 9) <= 0.03, f"{m.get('dropout')} (base {base.get('dropout')})")
chk('V/UV error <= 7%', (m.get('uv_err') or 9) <= 0.07, f"{m.get('uv_err')} (base {base.get('uv_err')})")
chk('jumps>25Hz <= 3%', (m.get('jumps_gt25') or 9) <= 0.03, f"{m.get('jumps_gt25')} (base {base.get('jumps_gt25')})")
sr = m.get('span_ratio')
chk('span ratio 0.7-1.5', sr is not None and 0.7 <= sr <= 1.5, f"{sr} (base {base.get('span_ratio')})")
chk('UTMOS >= v0.4.1', m.get('utmos_mean', 0) >= base.get('utmos_mean', 0),
    f"{m.get('utmos_mean')} (base {base.get('utmos_mean')})")
chk('UTMOS p10 >= v0.4.1', m.get('utmos_p10', 0) >= base.get('utmos_p10', 0),
    f"{m.get('utmos_p10')} (base {base.get('utmos_p10')})")
# CER is "when run"; not available locally (no Whisper model) -> informational only

print(f'=== REVIEW Phase 1 automatic Gate 1: {tag} ===')
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name:24s} {detail}")
allok = all(ok for _, ok, _ in checks)
print('OVERALL:', 'PASS' if allok else 'FAIL', '(automatic part; owner A/B still required)')
