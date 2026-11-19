"""Build teacher-forced (GTA) decoder inputs from the Phase-1 text model.

For every row: run TextStudentUV with MEASURED durations and store gta_F0 (Hz via f0_hz),
gta_N and gta_asr = asr_tok @ aln. Audio stays aligned because durations are measured.
Output: data/teacher_gta/shard_0000.pt (same rows + three new fields).
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'paradee_ms'))
os.chdir(ROOT)

from student import alignment  # noqa: E402
from student_uv import TextStudentUV  # noqa: E402

SRC = 'data/teacher/shard_0000.pt'
DST = 'data/teacher_gta/shard_0000.pt'
CKPT = 'runs/kaggle_sm21/text_f0uv.pt'
rows = torch.load(SRC, map_location='cpu', weights_only=False)
m = TextStudentUV('s+mlp').eval()
m.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False)['model'], strict=True)
print(f'GTA: {len(rows)} rows from {SRC} via {CKPT}', flush=True)

with torch.no_grad():
    for i, r in enumerate(rows):
        Fr = len(r['F0']) // 2
        ids = r['ids'][None]
        aln = alignment(r['pred_dur'], Fr)[None]
        dur, lf0n, uv, N, asr_tok = m(ids, torch.tensor([ids.shape[1]]), aln)
        F0 = m.f0_hz(lf0n, uv)[0]
        r['gta_F0'] = F0.cpu()
        r['gta_N'] = N[0].cpu()
        r['gta_asr'] = (asr_tok @ aln)[0].cpu()
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(rows)}', flush=True)

os.makedirs(os.path.dirname(DST), exist_ok=True)
torch.save(rows, DST)
print('saved', DST, '| size MB', round(os.path.getsize(DST) / 1e6, 1))
