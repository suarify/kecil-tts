import subprocess

msg = """
## 2026-11-20 Space bug fixed + UTMOS results posted on model card
- Owner reported the HF Space was broken ("model not loaded"). Diagnosed via browser console: two `ReferenceError: Cannot access 'tts'/'lastAudio' before initialization` in `index.html` from the UI polish (speedStrip() called by drawSpeakers() before `let tts`, and a resize listener before `let lastAudio`). This aborted the whole module script before load() ran.
- Fix: declared `let tts, lastAudio;` at the top of the module and removed the later declarations. JS syntax checked (`node --check`), Space re-uploaded, verified LIVE in the browser: "Model ready on webgpu in 15.8 s", Speak button active, 3 speakers rendered. Remaining console entries are benign ONNX Runtime warnings only.
- Owner asked to post the UTMOS result on the model: updated `Suarify/kecil-tts-yati-v0.5.0` README.md with the full development leaderboard (teacher 3.553, v0.5/P3 2.275, P1 2.180, v0.4.1 2.158, P2 2.010) plus the Phase-4 CER finding (general 2.2% / conversational 1.9% pass; numbers 43.8% / codeswitch 10.8% fail). Verified remotely.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/check_space_logs.py', 'scripts/inspect_space_tdz.py',
         'scripts/find_lastdec.py', 'scripts/find_modstart.py', 'scripts/update_card_utmos.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Fix Space JS TDZ error + post UTMOS results on model card', '--quiet'])
print('committed')
