"""Compare the raw (pre-clamp) durations from the old and the delta-trained text side."""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

from ms_g2p import G2P  # noqa: E402
from ms_norm import normalize  # noqa: E402
from student import TextStudent  # noqa: E402

TEXTS = [
    ('iii', 'Ini terletak di bawah Jenis III, atau kematian sel sitoplasma.'),
    ('atau2', 'Kita boleh pergi sekarang atau tunggu sehingga petang.'),
    ('mixed', 'Buku ini untuk murid tahun lima di sekolah kebangsaan.'),
]
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')

for label, ck in (('old (12k/7.5k)', 'runs/kaggle_sm06/textA_C4_last.pt'),
                  ('delta (+4.5k)', 'runs/kaggle_sm07/textA_delta_last.pt')):
    ts = TextStudent('s+mlp').eval()
    ts.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    print(f'\n{label}')
    with torch.no_grad():
        for key, text in TEXTS:
            ph = g2p(normalize(text))
            ids = torch.tensor([g2p.tokens(ph)], dtype=torch.long)
            t, p, s = ts, ts.predictor, ts.style
            d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
            x, T = d_en, ids.shape[1]
            sx = s[:, :, None].expand(-1, -1, T)
            x = torch.cat([x, sx], 1)
            for b in p.text_encoder.lstms:
                if isinstance(b, nn.LSTM):
                    x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
                else:
                    x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
            d = x.transpose(-1, -2)
            dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]
            n = torch.clamp(torch.round(dur), min=1).long()
            ones = float((n == 1).float().mean())
            long_ = int((n >= 10).sum())
            print(f'  {key:7s} frames {int(n.sum()):4d} | 1-frame {ones:.0%} | >=10-frame {long_} | max {int(n.max())}',
                  f'| atau span {n[ph.find("ataw"):ph.find("ataw") + 4].tolist() if ph.find("ataw") >= 0 else "-"}')
