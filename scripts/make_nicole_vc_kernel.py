"""Create the Nicole VC kernel from the Fenrir one (ref + output names swapped)."""
import os

p = 'kaggle/sm26_fenrir_vc/sm26_fenrir_vc.py'
s = open(p, encoding='utf-8').read()
s = s.replace('# sm26-fenrir-vc', '# sm28-nicole-vc')
s = s.replace('to am_fenrir', 'to af_nicole')
s = s.replace('ref_fenrir.wav', 'ref_nicole.wav')
s = s.replace('fenrirB.pt', 'nicoleB_v05.pt')
s = s.replace('fenrirB_partial.pt', 'nicoleB_v05_partial.pt')
s = s.replace('sm26_report.json', 'sm28_report.json')
os.makedirs('kaggle/sm28_nicole_vc', exist_ok=True)
open('kaggle/sm28_nicole_vc/sm28_nicole_vc.py', 'w', encoding='utf-8').write(s)

meta = '''{
 "id": "suarifymalaysia/sm-28-nicole-vc",
 "title": "sm 28 nicole vc",
 "code_file": "sm28_nicole_vc.py",
 "language": "python",
 "kernel_type": "script",
 "is_private": true,
 "enable_gpu": true,
 "enable_tpu": false,
 "enable_internet": true,
 "keywords": ["gpu"],
 "dataset_sources": ["suarifymalaysia/sm28-nicole-vc-in"],
 "kernel_sources": [],
 "competition_sources": [],
 "model_sources": [],
 "machine_shape": "NvidiaTeslaT4"
}
'''
open('kaggle/sm28_nicole_vc/kernel-metadata.json', 'w', encoding='utf-8').write(meta)
print('nicole VC kernel prepared')
