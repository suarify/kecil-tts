s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    'info element kept but hidden': 'id="info" hidden' in s,
    'ph element kept but hidden': 'Phonemes: <code id="ph">' in s and '<p class="micro" hidden>Phonemes' in s,
    'bench is details collapsed': '<details id="benchd">' in s and '<details id="benchd" open' not in s,
}
for k, v in checks.items(): print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
