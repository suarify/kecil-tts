s = open('scripts/f0n_diag2.py', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if 'aln' in l or 'alignment' in l or 'pred_dur' in l or 'F0p' in l or 'F0Ntrain' in l:
        print(i, l[:180])
