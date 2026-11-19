import subprocess

msg = """
## 2026-11-20 Space UI v4: smaller info icon + streaming timing clarified
- Owner: info icon 30% smaller and floating top-right after the Kecil-TTS word -> .infobtn 30->21px, font 17->12px, translateY(-4px), margin 8px (verified live 21px).
- Owner: "why time taken >2s even in streaming" -> two causes fixed: (a) the timer t0 was started before `await ensureAudio()` (AudioContext resume + FX chain build) and phonemize(), so setup time was counted as synthesis; moved t0 to just before generation. (b) Streaming only speeds FIRST audio, not total synthesis; the stats panel now swaps labels in streaming mode: "Time taken" -> "Time to audio" (shows firstT) and "First audio" -> "Total gen" (shows total), with an explanatory note in the info line.
- Verified live after generation in streaming mode: "Time to audio 554 ms", "Total gen 554 ms", render speed 5.6x, no JS errors. Longer sentences will show first-audio < total gen.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space4.py', 'scripts/inspect_time.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Space UI v4: smaller info icon, honest streaming timing', '--quiet'])
print('committed')
