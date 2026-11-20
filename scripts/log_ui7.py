import subprocess

msg = """
## 2026-11-20 Space UI v7 (owner request while Fenrir trains)
- Effects lab is now a modal popup (`.modal-wrap#fxModal`), opened only by a new ⚙ gear icon beside the speed/stream row; closes via ×, backdrop click, or Escape.
- Speak + Replay buttons moved to their own row BELOW the speed/stream row.
- The status sentence ("first audio after ... on webgpu ...") and Phonemes line moved ABOVE the textarea (right under the waveform) and shrunk to 10px micro text.
- Verified live: gear opens/closes the modal, info font-size 10px, info above textarea, model loads normally, zero JS errors.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space7.py', 'scripts/verify_space7.py', 'scripts/inspect_ui7.py']
subprocess.run(['git', 'add'] + files, check=True)
r = subprocess.run(['git', 'commit', '-m', 'Space UI v7: effects-lab gear modal, button rows, micro status', '--quiet'], capture_output=True, text=True)
print('committed rc', r.returncode)
