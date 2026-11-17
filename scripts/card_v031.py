"""Write the v0.3.1 card from the v0.3.0 one."""
src = 'runs/michael_v030_repo/README.md'
dst = 'runs/michael_v031_repo/README.md'
s = open(src, encoding='utf-8').read()
s = s.replace('Kecil-TTS Michael v0.3.0', 'Kecil-TTS Michael v0.3.1').replace('kecil-tts-michael-v0.3.0', 'kecil-tts-michael-v0.3.1')
old_intro = '**v0.3.0 is the first version exported from the true training checkpoints**'
new_intro = ('**v0.3.1 keeps the v0.3.0 weights and removes the timing clamp that made the voice too slow.** '
             'v0.3.0 was the first version exported from the true training checkpoints')
assert old_intro in s
s = s.replace(old_intro, new_intro)

changelog = """## What changed in v0.3.1

1. **Timing clamp removed.** v0.3.0 forced every phoneme to at least 75 ms, which made the speech about 22% slower
   than the model intends (measured on the same sentence: 5.85 s vs 4.55 s). v0.3.1 uses the model's own durations.
2. **Faithful synthesis.** No gap-masking, burst gating, gain smoothing or air shelf: the vocoder output is exactly
   what training produced. In listening tests this sounded cleaner than the processed variant.

Weights, file size (8.46 MB) and recognition scores are unchanged from v0.3.0.

## What changed in v0.3.0"""
s = s.replace('## What changed in v0.3.0', changelog)
open(dst, 'w', encoding='utf-8').write(s)
print('card written')
