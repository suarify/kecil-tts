"""Export + eval a Nicole checkpoint with the identical FIX H recipe."""
import os
import subprocess
import sys

PY = sys.executable
ckpt = sys.argv[1]
tag = sys.argv[2]
text = 'runs/kaggle_sm29/nicole_text.pt'
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
        print('FAILED', r.stderr[-1200:], flush=True)
        break
    if 'utmos_mean' in r.stdout:
        import json
        m = json.load(open(f'runs/sota/{tag}/metrics.json'))
        print('RESULT', tag, 'UTMOS', m['utmos_mean'], m['utmos_p10'], flush=True)
