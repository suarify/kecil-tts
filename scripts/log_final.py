import subprocess

msg = """
## 2026-11-20 FINAL DECISION: stop here, keep Yati v0.5 as the live voice
- Owner considered a balanced GAN continuation (~4-5h) and a small-data retrain (~6h) within the remaining ~7 GPU-hours, and chose: STOP. No further GPU spend.
- Final state: `Suarify/kecil-tts-yati-v0.5.0` published (42 files, sha match, 0 float64, 8.47 MB, UTMOS 2.275/1.793); Space `Suarify/suaradee-web` updated with the polished UI (Yati v0.5, Nicole as 3rd speaker, auto-speak chips, stats + waveform, sleeker fonts). Nicole v0.1 and Michael v0.3.1 remain published.
- REVIEW phases 1-4 all exhausted: P1 (log-F0+UV) and P2 (GTA decoder) failed gates and were judged "almost the same" by the owner; P3 (GAN, after a NaN bug fix) gave +0.12 UTMOS and was published as v0.5 despite the owner noting the difference was not clearly significant; P4 (more data) failed the CER gate (general/conversational clean at ~2%, numbers/codeswitch noisy) and was not pursued given the owner's 7h budget.
- Route A (fine-tune Kokoro-82M + distil, 10-20 GPU-h) remains available ONLY with a fresh owner decision (REVIEW section 4).
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
subprocess.run(['git', 'add', 'STATUS.md'], check=True)
subprocess.run(['git', 'commit', '-m', 'Final decision: stop, keep Yati v0.5 live', '--quiet'])
print('committed')
