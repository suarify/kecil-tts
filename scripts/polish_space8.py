"""Space UI v8: hide info/phonemes micro lines; benchmarks collapsible, collapsed by default."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# 1. hide the first-audio / phonemes lines entirely (keep elements so JS refs stay valid)
old = '''    <p class="micro" id="info"></p>
    <p class="micro">Phonemes: <code id="ph"></code></p>
'''
new = '''    <p class="micro" id="info" hidden></p>
    <p class="micro" hidden>Phonemes: <code id="ph"></code></p>
'''
assert old in s, 'micro lines not found'
s = s.replace(old, new, 1)

# 2. benchmarks -> details/summary, collapsed by default
old = '<section class="card bench"><h2 data-i18n="hbench">Benchmarks</h2>\n    <table>'
new = '<section class="card bench"><details id="benchd"><summary data-i18n="hbench">Benchmarks</summary>\n    <table>'
assert old in s, 'bench header not found'
s = s.replace(old, new, 1)
old = '    <p class="na" data-i18n="nbench">CER measured 2026-11-17'
# close the details right before the bench section's </section>
marker = 'No MOS or side-by-side human rating exists for either voice at this date.</p>\n  </section>'
assert marker in s, 'bench close marker not found'
s = s.replace(marker, 'No MOS or side-by-side human rating exists for either voice at this date.</p>\n  </details></section>', 1)

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('UI v8 written')
