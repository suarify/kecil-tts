"""Compare torch text-side intermediates against Michael's ONNX intermediates to find the bad mapping."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '.stubs'))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
sys.path.insert(0, os.path.join(ROOT, 'g2p'))
os.chdir(ROOT)

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from ms_g2p import G2P
from ms_norm import normalize

text = 'Selamat pagi, ini ujian suara Michael versi baru.'
g2p = G2P(stress=False, r='r', diph='glide', final_a='a')
ids = torch.tensor([g2p.tokens(g2p(normalize(text)))], dtype=torch.long)

# ONNX with intermediates exposed
m = onnx.load('runs/michael_v02.onnx')
want = [('/Round_output_0', onnx.TensorProto.FLOAT)]
have = {o.name for o in m.graph.value_info} | {o for n in m.graph.node for o in n.output}
extra = [w for w, t in want if w in have]
m.graph.output.extend([onnx.helper.make_tensor_value_info(w, t, None) for w, t in want if w in have])
onnx.save(m, 'runs/michael_probe2.onnx')
sess = ort.InferenceSession('runs/michael_probe2.onnx', providers=['CPUExecutionProvider'])
outs = sess.run(None, {'input_ids': ids.numpy(), 'speed': np.ones(1, np.float32)})
for nm, o in zip(['waveform'] + extra, outs):
    print('onnx', nm, np.asarray(o).shape, np.asarray(o).ravel()[:12])
dur_onnx = outs[1][0]
print('onnx dur sum:', dur_onnx.sum(), 'frames ->', dur_onnx.sum() * 600, 'samples')

# torch side (rebuild state from the exported pt)
from student import TextStudent
from vhead import VDecoder, FHOP, SR  # noqa: F401
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

ck = torch.load('runs/michael_v02_rebuilt.pt', map_location='cpu')
ts = TextStudent('s+mlp').eval()
vd = VDecoder(head='v2').eval()


def strip(mod):
    for x in mod.modules():
        try:
            remove_weight_norm(x)
        except Exception:
            pass
        if is_parametrized(x, 'weight'):
            try:
                remove_parametrizations(x, 'weight', leave_parametrized=True)
            except Exception:
                pass


strip(ts); strip(vd)
ts.load_state_dict({k[len('ts.'):]: v for k, v in ck['ts'].items()}, strict=False)
vd.load_state_dict({k[len('vd.'):]: v for k, v in ck['vd'].items()}, strict=False)

with torch.no_grad():
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
    dur_logit = p.duration_proj(p.lstm(d)[0])
    dur = torch.sigmoid(dur_logit).sum(-1)[0]
    print('torch dur sum:', float(dur.sum()), 'tokens:', T)
    print('torch per-token dur:', dur[:12].tolist())
