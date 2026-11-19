import subprocess

msg = """
## 2026-11-19 REVIEW.md Phase 0 DONE: eval harness + baselines
- Adopted REVIEW.md v8 (supersedes HANDOFF 7 items 2-3: no fenrir/further nicole until Phase 3).
- Fixed 80-sentence eval set eval/sota_eval.tsv (40 general + 20 code-switch + 20 numbers, stable ids from data/text/test_200.tsv).
- scripts/sota_eval.py: renders with ORT CPU -> runs/sota/<tag>/<id>.wav, pYIN F0 health (fmin=70, same as ms_rows), UTMOS22-strong at 16 kHz, predicted-vs-measured F0 on 60 held rows, leaderboard.tsv.
- Baselines: TEACHER (15 Chatterbox-converted clips) UTMOS 3.5534/3.1623 (mean/p10); Yati v0.4.1 2.1447/1.7684, F0 dropout 16.4%, V/UV err 25.4%, jumps>25Hz 9.2%, span ratio 2.867; Michael v0.3.1 1.8132/1.4449.
- Confirms REVIEW's root-cause: the text side's F0 prediction, not the 8M head (bigger head gave zero gain). Next: Phase 1 log-F0 + uv head (CPU first, gate 1 automatic + owner A/B).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
subprocess.run(['git', 'add', '-A'])
subprocess.run(['git', 'commit', '-m', 'REVIEW Phase 0: SOTA eval harness + baselines (UTMOS, F0 health)', '--quiet'])
print('committed')
