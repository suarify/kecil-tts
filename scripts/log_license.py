import subprocess

msg = """
## 2026-11-20 License switched to CC BY 4.0 on all 5 published model repos (owner approved)
- Owner: change licensing from open source (Apache-2.0) to Creative Commons Attribution 4.0 (CC BY 4.0).
- Updated all 5 repos (yati v0.5, v0.4.1, michael v0.3.1, nicole v0.1, firdaus v0.1): README license tag `apache-2.0` -> `cc-by-4.0`, all body references Apache-2.0 -> CC BY 4.0, and replaced/added the LICENSE file with the official CC BY 4.0 legal code (fetched from creativecommons.org).
- Verified remotely: all 5 show license ['cc-by-4.0'], LICENSE present, no 'apache' left in cards.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/switch_license.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Switch all 5 model repos to CC BY 4.0', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
