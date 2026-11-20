s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    'gear icon beside stream': 'id="fxBtn"' in s and 'class="gear"' in s,
    'fx is modal': 'id="fxModal"' in s and 'id="fxCard"' in s and 'class="modal-wrap" id="fxModal"' in s,
    'fx old details gone': '<details id="fxd">' not in s,
    'fx close handlers': "$('fxClose').onclick" in s and "$('fxBtn').onclick" in s,
    'buttons own row': s.find('id="go"') > s.find('id="st"'),
    'info micro above textarea': s.find('id="info"') < s.find('<textarea') and 'class="micro" id="info"' in s,
    'phonemes micro': 'class="micro">Phonemes' in s,
    'esc closes both': "$('fxModal').hidden = true" in s and "$('edgeModal').hidden = true" in s,
}
for k, v in checks.items(): print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
