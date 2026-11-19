import torch
base = torch.load('runs/k16_final/dec_c4_last.pt', map_location='cpu', weights_only=False)['model']
m = torch.load('runs/gan_smoke2/last.pt', map_location='cpu', weights_only=False)['model']
keys = [k for k in m if k in base and m[k].shape == base[k].shape]
d = [(m[k].float() - base[k].float()).abs().mean().item() for k in keys]
print('mean abs diff', round(sum(d)/len(d), 7), 'max', round(max(d), 7))
