import torch
base = torch.load('runs/k16_final/dec_c4_last.pt', map_location='cpu', weights_only=False)['model']
for f in ('dec_gan.pt', 'dec_gan_2000.pt', 'dec_gan_8000.pt'):
    m = torch.load(f'runs/kaggle_sm21gan/{f}', map_location='cpu', weights_only=False)['model']
    keys = [k for k in m if k in base and m[k].shape == base[k].shape]
    diffs = [(m[k].float() - base[k].float()).abs().mean().item() for k in keys]
    print(f, 'mean abs diff', round(sum(diffs) / max(1, len(diffs)), 6), '| max', round(max(diffs), 6))
