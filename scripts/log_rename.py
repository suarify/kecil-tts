import subprocess

msg = """
## 2026-11-20 Python package renamed suaradee -> kecil-tts (owner approved, full rename + shim)
- Owner asked to change the name suaradee to kecil-tts. Chose full rename with backwards-compatible shim.
- Local: `python/suaradee/` -> `python/kecil_tts/` (real package), class `Suaradee` -> `KecilTTS`; `python/suaradee/` is now a tiny shim re-exporting `KecilTTS`/`SAMPLE_RATE` and aliasing `Suaradee = KecilTTS` so old code still works. pyproject distribution now `kecil-tts` with both packages. Verified imports locally.
- Rebuilt wheels for all 5 repos: `kecil_tts-<ver>-py3-none-any.whl` (contains both `kecil_tts` and `suaradee` shim), uploaded to dist/ (old suaradee wheels left in place).
- Updated all 5 cards + python examples to `from kecil_tts import SAMPLE_RATE, KecilTTS` / `KecilTTS.from_pretrained` + new wheel URLs. Verified: all cards new API, no stale Suaradee/Suaradee.from_pretrained, new wheel present with both modules.
- Space JS: added `export const KecilTTS = Suaradee` to kecil.js + suaradee.js, re-uploaded. Verified live: both exports present and identical, 4 speakers, zero JS errors.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/rename_package.py', 'scripts/build_kecil_wheels.py', 'scripts/update_cards_rename.py',
         'scripts/space_js_alias.py', 'scripts/verify_rename.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Rename package suaradee -> kecil-tts with backwards-compatible shim', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
