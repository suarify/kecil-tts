"""Compare baseline vs P1 predicted F0/UV on the same held rows with TEACHER-FORCED alignment."""
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'g2p'); sys.path.insert(0, 'paradee_ms'); sys.path.insert(0, 'python')
from ms_g2p import G2P
from ms_norm import normalize
from student import alignment
from student_uv import TextStudentUV

g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
rows = torch.load('data/teacher/shard_0000.pt', map_location='cpu', weights_only=False)[:60]


def predict(model, r, uv):
    ids = torch.tensor([g2p.tokens(g2p(normalize(r['text'])))], dtype=torch.long)
    t, p, s = model, model.predictor, model.style
    d_en = t.bert_encoder(t.bert(ids, attention_mask=torch.ones_like(ids))).transpose(-1, -2)
    x, T = d_en, ids.shape[1]
    sx = s[:, :, None].expand(-1, -1, T)
    x = torch.cat([x, sx], 1)
    for b in p.text_encoder.lstms:
        x = (b(x.transpose(-1, -2))[0].transpose(-1, -2) if isinstance(b, nn.LSTM)
             else torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1))
    d = x.transpose(-1, -2)
    aln = alignment(r['pred_dur'])[None]   # TEACHER FORCED (same as training)
    en = d.transpose(-1, -2) @ aln
    if uv:
        lf0n, uvlog, N = t.prosody(en, s)
        f0 = t.f0_hz(lf0n, uvlog)[0].numpy()
        mask = (torch.sigmoid(uvlog) > 0.5)[0].numpy()
    else:
        f0, N = p.F0Ntrain(en, s)
        f0 = f0[0].numpy()
        mask = f0 > 50
    return f0, mask


for tag, path, uv in (('base', 'runs/k16_final/textA_c4_last.pt', False),
                      ('p1', 'runs/kaggle_sm21/text_f0uv.pt', True)):
    m = TextStudentUV('s+mlp').eval()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=False)
    drop, uverr, voicedfrac = [], [], []
    with torch.no_grad():
        for r in rows:
            f0, mask = predict(m, r, uv)
            m0 = r['F0'].numpy()
            L = min(len(f0), len(m0))
            a, b = f0[:L], m0[:L]
            vm = b > 50
            if vm.sum() < 10:
                continue
            drop.append(float((a[vm] < 50).mean()))
            uverr.append(float((vm != (a > 50)).mean()))
            voicedfrac.append(float(mask.mean()))
    print(tag, 'dropout', round(float(np.mean(drop)), 4), 'uverr', round(float(np.mean(uverr)), 4),
          'pred voiced frac', round(float(np.mean(voicedfrac)), 3))
