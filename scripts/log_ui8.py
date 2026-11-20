import subprocess

msg = """
## 2026-11-20 Space UI v8 (owner request while Fenrir trains)
- (1) The "first audio... / Phonemes" micro line hidden (elements kept with `hidden` attr so JS refs stay valid - removing them broke load() with a null textContent error, fixed and re-verified).
- (2) Benchmarks card is now a `<details id="benchd"><summary>` collapsible, collapsed by default (no `open` attr).
- Verified live: infoVisible false, phVisible false, benchOpen false, model loads, 3 speakers, zero JS errors.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space8.py', 'scripts/verify_space8.py', 'scripts/verify_space8b.py', 'scripts/inspect_ui8.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Space UI v8: hidden status/phonemes + collapsible benchmarks', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
