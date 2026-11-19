import subprocess

msg = """
## 2026-11-19 Phase 1 verdict + Phase 2 approved
- Owner listened to the blind page (`runs/listen/sota_p1.html`) and said "both almost the same" -> P1 is NOT published; v0.4.1 remains the released voice. Consistent with the flat objective numbers (UTMOS +0.03, voicing unchanged).
- Owner explicitly approved Phase 2 GTA with Kaggle GPU: fine-tune the Yati v0.4.1 decoder on its own text-model-predicted F0/N (the second root cause in REVIEW.md: the decoder only ever saw measured inputs). 3k steps, bs16, seg64, lr1e-4.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
subprocess.run(['git', 'add', 'STATUS.md'], check=True)
subprocess.run(['git', 'commit', '-m', 'Phase 1 verdict: P1 not published; Phase 2 GTA approved', '--quiet'])
print('committed')
