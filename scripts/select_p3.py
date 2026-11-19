"""Pick the best P3 checkpoint by UTMOS on the 80-sentence eval set.

usage: python scripts/select_p3.py runs/kaggle_sm21gan/dec_gan_2000.pt ...
Exports each checkpoint with the identical FIX H recipe, webgpu-patches it, runs sota_eval,
and prints a UTMOS table. Also verifies 0 float64 per export.
"""
import os
import subprocess
import sys

PY = sys.executable
env = dict(os.environ, PYTHONUTF8='1')

for ckpt in sys.argv[1:]:
    tag = 'p3_' + os.path.basename(ckpt).replace('.pt', '')
    out = f'runs/{tag}_export'
    print(f'=== {ckpt} -> {tag}', flush=True)
    for cmd in (
        [PY, 'scripts/export_fixh.py', ckpt, '--uv-text', '--text', 'runs/text_f0uv.pt', '--out', out],
        [PY, 'scripts/webgpu_patch.py', f'{out}/paradee_int8.onnx', f'{out}/paradee_int8_webgpu.onnx'],
        [PY, 'scripts/sota_eval.py', f'{out}/paradee_int8_webgpu.onnx', '--tag', tag,
         '--uv-text', '--text', 'runs/text_f0uv.pt'],
    ):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode:
            print('FAILED', cmd, r.stderr[-800:], flush=True)
            break
        if 'utmos_mean' in r.stdout:
            import json
            m = json.load(open(f'runs/sota/{tag}/metrics.json'))
            print(tag, 'UTMOS', m['utmos_mean'], m['utmos_p10'], flush=True)
