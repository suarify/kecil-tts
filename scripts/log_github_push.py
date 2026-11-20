import subprocess

msg = """
## 2026-11-20 Source pushed to GitHub (suarify/kecil-tts)
- Owner asked "source commit n push". Initially blocked: cached GitHub credential was dragonjump (403), and HTTPS push of large .pt files rejected (>100MB each).
- Used the owner-provided SSH profile `github-suarify` (id_ed25519_suarify) - authenticated as suarify.
- Large generated data files (data/teacher/shard_0000.pt 328MB, data/teacher_gta/shard_0000.pt 749MB, kaggle/ds_k15clips/cb_clips2.pt 189MB, kaggle/ds_rows_v02/shard_0000.pt 179MB) were untracked and .gitignore extended (*.pt, *.bin). Used git-filter-repo to strip them from all 165 commits' history (verified 0 blobs >100MB).
- Remote `main` already had an older clean v0.4-era source repo (8 commits, 33 files) - owner approved force-push. Force-pushed local main -> remote main (dfd6d9a -> 813ce79). Verified `git ls-remote origin main` matches local HEAD.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
subprocess.run(['git', 'add', 'STATUS.md'], check=True)
r = subprocess.run(['git', 'commit', '-m', 'Log GitHub push', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
r2 = subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True, text=True)
print('pushed rc', r2.returncode, r2.stderr[-200:] if r2.returncode else 'ok')
