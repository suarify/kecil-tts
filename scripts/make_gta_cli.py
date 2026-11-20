"""Build GTA decoder inputs from a trained TextStudentUV checkpoint (Kaggle CLI version)."""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--rows', required=True)
ap.add_argument('--text', required=True)
ap.add_argument('--out', required=True)
A = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student import alignment  # noqa: E402
from student_uv import TextStudentUV  # noqa: E402

rows = torch.load(A.rows, map_location='cpu', weights_only=False)
m = TextStudentUV('s+mlp').eval()
m.load_state_dict(torch.load(A.text, map_location='cpu', weights_only=False)['model'], strict=True)
print(f'GTA: {len(rows)} rows from {A.rows} via {A.text}', flush=True)

with torch.no_grad():
    for i, r in enumerate(rows):
        Fr = len(r['F0']) // 2
        ids = r['ids'][None]
        aln = alignment(r['pred_dur'], Fr)[None]
        _, lf0n, uv, N, asr_tok = m(ids, torch.tensor([ids.shape[1]]), aln)
        r['gta_F0'] = m.f0_hz(lf0n, uv)[0].cpu()
        r['gta_N'] = N[0].cpu()
        r['gta_asr'] = (asr_tok @ aln)[0].cpu()
        if (i + 1) % 300 == 0:
            print(f'  {i + 1}/{len(rows)}', flush=True)

os.makedirs(os.path.dirname(A.out), exist_ok=True)
torch.save(rows, A.out)
print('saved', A.out, '| MB', round(os.path.getsize(A.out) / 1e6, 1), flush=True)
