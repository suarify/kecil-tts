import subprocess

msg = """
## 2026-11-19 REVIEW Phase 1 owner A/B page (owner chose to skip the automatic gate + retry)
- Owner explicitly chose "Skip to owner A/B" over the REVIEW-prescribed retry and over stopping Phase 1. The automatic Gate 1 remains FAILED; no publish decision yet.
- Built `runs/listen/sota_p1.html` via `scripts/make_sota_p1_page.py --baseline runs/sota/yati_v041 --p1 runs/sota/p1 --n 12`: 12 sentences (6 general / 3 codeswitch / 3 numbers), A/B shuffled per row, loudness-matched to equal RMS, "pick cleaner" + "watery" ticks, labels revealed only after pressing Reveal, P1>=8/12 tally. 24 wavs verified (24 kHz, non-empty).
- Page opened in the Review pane for the owner. Await owner verdict before ANY publish or Space upload.
- Space UI v2 staged locally (NOT uploaded): `scripts/polish_space2.py` adds Nicole as a real third speaker (version dropdown underneath), auto-speaking sample chips (already in v1), a large render-stats panel (realtime factor / time taken / first audio / audio length / device + comparison bar) and a live waveform canvas with a moving playhead; JS syntax checked with `node --check`.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space2.py', 'scripts/verify_space2.py', 'scripts/extract_js.py',
         'scripts/diag_p1_uv.py', 'scripts/check_sm21_log.py', 'scripts/check_p1_wavs.py',
         'scripts/verify_p1_page.py', 'scripts/make_p1_repo.py', 'scripts/log_p1_gate1.py',
         'scripts/read_eval_log.py', 'scripts/inspect_space2.py', 'scripts/check_diagalign.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Phase 1 owner A/B page + Space UI v2 (Nicole speaker, stats, waveform)', '--quiet'])
print('committed')
