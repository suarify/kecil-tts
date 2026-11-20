import subprocess

msg = """
## 2026-11-20 Nicole VC done + training submitted; alt-account finding
- VC kernel sm-28-nicole-vc COMPLETE: 1,281 clips, 0 failures, 33 min, RMS 0.07 (loudness fixed). Dataset sm29-nicole-train-in uploaded (20 files). Training kernel sm-29-nicole-train QUEUED (female af_nicole, default f0, text 10k UV -> GTA -> decoder 12k -> GAN 2k from Yati v0.5 inits).
- Alt Kaggle token investigation: the owner-provided token `KGAT_be0b...` authenticates but belongs to the SAME account (suarifymalaysia) - a throwaway kernel pushed with it returned /code/suarifymalaysia/...; deleted the probe kernel. Token saved at ~/.kaggle/kaggle-alt.json as a spare credential. Quota MCP tool: 17h used of 30h, ~13h left, resets 2026-11-21T00:00Z - enough for Nicole + Michael.
- Owner's 3 points answered: (1) new Nicole will sound like af_nicole and be at Yati loudness (RMS 0.07) not soft; (2) alt account is the same account, no switch possible/needed; (3) Michael watery/phasey -> plan after Nicole: pYIN fmin=50 + Phase 1-3 recipe from his v0.3.1 weights.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/wait_sm29.py', 'scripts/setup_kaggle_alt.py', 'scripts/verify_kaggle_alt.py',
         'scripts/verify_kaggle_alt2.py', 'scripts/identify_alt.py', 'scripts/test_alt_access.py',
         'scripts/probe_alt_username.py', 'scripts/del_probe.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Nicole training submitted; alt token = same account; quota ok', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
