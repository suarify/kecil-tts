import torch
for f in ('runs/text_f0uv.pt', 'runs/kaggle_sm27/fenrir_text.pt'):
    d = torch.load(f, map_location='cpu', weights_only=False)['model']
    print(f, 'lf0_mu', round(float(d['lf0_mu']), 3), 'lf0_sd', round(float(d['lf0_sd']), 3),
          '-> Hz', round(__import__('math').exp(float(d['lf0_mu'])), 1))
