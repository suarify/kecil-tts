import subprocess

msg = """
## 2026-11-20 Model card quick-start fix (owner question: "is `from suaradee import` legit?")
- Verified: the published wheels DO install the `suaradee` module (`suaradee/__init__.py` -> `Suaradee`, `SAMPLE_RATE`) and their default REPO points to the correct model. So the README quick-start `from suaradee import SAMPLE_RATE, Suaradee` / `Suaradee.from_pretrained(...)` IS legit and correct.
- The STALE parts were the `examples/*.py` files, which imported a non-existent `kecil_tts`/`KecilTTS` and referenced an old wheel URL (michael-v0.2). Fixed all example files across the 5 repos to use the real `suaradee` module, correct wheel filenames and correct repo ids.
- Fixed stale `js/suaradee.js` default repo (`Suarify/kecil-tts-yati` -> each repo's full id) for the 3 repos that have that file.
- Michael's card had a stale quick-start (kecil_tts/KecilTTS + a wheel URL that doesn't exist - his repo has no dist/ wheel). Fixed import to `suaradee`/`Suaradee` and removed the non-existent wheel install line (from_pretrained still works).
- Verified all 5 cards now: suaradee import present, no stale KecilTTS, wheel refs correct.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/fix_examples.py', 'scripts/fix_js_repo.py', 'scripts/fix_michael_card.py',
         'scripts/fix_michael_card2.py', 'scripts/verify_cards.py', 'scripts/check_wheel_repo.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Fix stale examples/JS/cards to use suaradee module', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
