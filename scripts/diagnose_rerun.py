import os, re, torch
s = open('runs/kaggle_sm21gan/sm-21-gan.log', encoding='utf-8', errors='replace').read().replace('\x00', '')
lines = [l.strip() for l in s.splitlines() if l.strip()]
for l in lines:
    if 'NONFINITE' in l or 'diverged' in l or 'after:' in l or 'before:' in l or re.search(r'step 50 loss', l):
        d = re.search(r'"data":"(.*?)"\}', l)
        print((d.group(1) if d else l)[:240].replace('\\n',''))
print('--- checkpoint diff vs v0.4.1 ---')
base = torch.load('runs/k16_final/dec_c4_last.pt', map_location='cpu', weights_only=False)['model']
for f in sorted(os.listdir('runs/kaggle_sm21gan')):
    if f.startswith('dec_gan') and f.endswith('.pt'):
        m = torch.load(f'runs/kaggle_sm21gan/{f}', map_location='cpu', weights_only=False)['model']
        ks = [k for k in m if k in base and m[k].shape == base[k].shape]
        diffs = [(m[k].float()-base[k].float()).abs().mean().item() for k in ks]
        print(f, 'mean diff', round(sum(diffs)/len(diffs), 7), 'max', round(max(diffs), 7))
