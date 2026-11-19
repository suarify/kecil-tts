s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    '1 stats/wave hidden by default': '#stats,#wave{display:none}' in s,
    '1 revealed after generate': "$('stats').classList.add('on')" in s and "$('wave').classList.add('on')" in s,
    '2 info icon beside header': 'id="infoBtn"' in s and 'class="infobtn"' in s,
    '2 edge -> modal wrap': 'id="edgeModal"' in s and 'class="modal-wrap"' in s,
    '2 modal close + esc': 'id="edgeClose"' in s and "e.key === 'Escape'" in s,
    '2 edge section inside modal': 'modal-card" id="edge"' in s,
    '3 topbar right on mobile': '@media(max-width:560px)' in s and '.topbar{margin-left:auto}' in s,
    '3 no headrow wrap': '.headrow{flex-wrap:wrap}' not in s,
    '4 keywords hidden card': 'class="card seo-hidden"' in s and '.seo-hidden{height:.1px' in s,
    '4 keyword sentence': 'Kecil-TTS-8M is a kokoro tts,' in s and 'small kokoro tts.' in s,
    '5 cta section removed': 'class="card cta"' not in s,
    '5 footer cta-row': 'class="cta-row"' in s and 'Contact Suarify &rarr;' in s,
    '5 contact not button': 'class="btn"' not in s,
}
for k, v in checks.items():
    print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
