s = open('runs/local_v041/index.html', encoding='utf-8').read()
checks = {
    'info line hidden': 'class="micro" id="info"' not in s,
    'phonemes line hidden': 'Phonemes: <code id="ph">' not in s,
    'bench is details': '<details id="benchd">' in s and '<summary data-i18n="hbench">Benchmarks</summary>' in s,
    'bench collapsed (no open attr)': '<details id="benchd" open' not in s,
    'bench details closed': '</details></section>' in s,
}
for k, v in checks.items(): print(('PASS' if v else 'FAIL'), k)
print('ALL', all(checks.values()))
