p = 'kaggle/sm29_nicole_train/sm29_nicole_train.py'
s = open(p, encoding='utf-8').read()
s = s.replace('# rows (male f0-min 50) ->', '# rows (female, default f0) ->')
s = s.replace('# 1. rows (male voice: f0-min 50, am_nicole)', '# 1. rows (female voice: af_nicole, default f0 settings)')
s = s.replace('--voice am_nicole --f0-min 50 --f0-max 300 --letters', '--voice af_nicole --letters')
assert 'am_nicole' not in s and 'f0-min 50' not in s and 'af_nicole' in s
open(p, 'w', encoding='utf-8').write(s)
print('nicole kernel fixed: female voice, default f0')
