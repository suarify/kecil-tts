"""Export + eval a Fenrir checkpoint with the identical Yati v0.5 FIX H recipe.

usage: python scripts/export_fenrir.py runs/kaggle_sm27/fenrir_gan.pt fenrir_gan
"""
import os
import subprocess
import sys

PY = sys.executable
ckpt = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else 'fenrir'
text = 'runs/kaggle_sm27/fenrir_text.pt'
out = f'runs/{tag}_export'
env = dict(os.environ, PYTHONUTF8='1', PYTHONPATH='python')

for cmd in (
    [PY, 'scripts/export_fixh.py', ckpt, '--uv-text', '--text', text, '--out', out],
    [PY, 'scripts/webgpu_patch.py', f'{out}/paradee_int8.onnx', f'{out}/paradee_int8_webgpu.onnx'],
    [PY, 'scripts/sota_eval.py', f'{out}/paradee_int8_webgpu.onnx', '--tag', tag, '--uv-text', '--text', text],
):
    print('===', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print('FAILED', r.stderr[-1500:], flush=True)
        break
    if 'utmos_mean' in r.stdout:
        import json
        m = json.load(open(f'runs/sota/{tag}/metrics.json'))
        print('RESULT', tag, 'UTMOS', m['utmos_mean'], m['utmos_p10'], flush=True)
