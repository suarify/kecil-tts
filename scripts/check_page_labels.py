s = open('scripts/make_sota_p1_page.py', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if 'LABELP1' in l or 'Phase 1' in l or 'SOTA P1' in l or 'need &ge;8' in l or 'Pass =' in l or 'open(A.out' in l:
        print(i, l[:180])
