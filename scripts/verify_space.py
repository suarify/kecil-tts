s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    'speed strip html': 'id="speedstrip"' in s,
    'speedStrip fn': 'function speedStrip(' in s,
    'chip speaks': "if (!$('go').disabled) $('go').click()" in s,
    'speed on generate': 'speedStrip(sec / s);' in s,
    'p1 dropdown': 'Suarify/kecil-tts-yati-v0.5.0' in s,
    'new fonts': 'Plus+Jakarta+Sans' in s and 'Outfit' in s,
    'old fonts gone': 'Sora,Inter' not in s,
}
for k, v in checks.items():
    print(('PASS' if v else 'FAIL'), k)
