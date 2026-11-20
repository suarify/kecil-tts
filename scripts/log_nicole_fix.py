import subprocess

msg = """
## 2026-11-20 Nicole training re-submitted (filename fix) + Michael plan noted
- sm-29-nicole-train first run ERROR'd at setup: looked for `nicoleB.pt` but the dataset file is `nicoleB_v05.pt` (a find/replace artifact). Fixed the glob, re-pushed, QUEUED again. No real GPU spent.
- Michael watery/phasey improvement planned AFTER the Nicole gate: REVIEW section 5 says pYIN fmin=50 + Phase 1-3 recipe from his v0.3.1 weights. Available assets: local `runs/michael_v02_rebuilt.pt`, `sm13_in/dec_init_male.pt` + `textA_init_male.pt` / `textA_c4_last.pt`, Kaggle datasets `sm-michael-clips`, `sm-delta-train-in`. Michael needs his own data rebuilt with male F0 settings (bigger task than Nicole), will scope with owner after Nicole gate.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/get_sm29_log.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Nicole train re-submitted (filename fix); Michael plan noted', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
