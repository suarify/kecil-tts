s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    'nicole is 3rd speaker': "nicole: { name: 'Nicole'" in s,
    'nicole removed from yati versions': "'Suarify/kecil-tts-nicole-v0.1', 'nicole v0.1 (new)'" not in s,
    'nicole model detection': "includes('nicole') ? 'nicole'" in s,
    '3-col speaker grid': 'repeat(auto-fit,minmax(180px,1fr))' in s,
    'chips auto-speak': "if (!$('go').disabled) $('go').click()" in s,
    'stats panel': 'id="stats"' in s and 'id="stRt"' in s and 'id="stTime"' in s and 'id="stDev"' in s,
    'waveform canvas': 'id="wave"' in s,
    'drawWave fn': 'function drawWave(' in s,
    'playhead fn': 'function playhead(' in s,
    'stats on generate': 'speedStrip(sec / s, s, firstT, tts.device, sec);' in s,
    'waveform on generate': 'drawWave(lastAudio, 0); startPlayhead();' in s,
    'old strip gone': 'class="speedstrip"' not in s,
    'new fonts': 'Plus+Jakarta+Sans' in s and 'Outfit' in s,
}
for k, v in checks.items():
    print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
