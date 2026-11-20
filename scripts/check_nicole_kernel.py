s = open('kaggle/sm29_nicole_train/sm29_nicole_train.py', encoding='utf-8').read()
for k in ('am_fenrir', 'f0-min 50', 'nicoleB.pt', 'af_nicole', 'nicole_text.pt', 'f0-min'):
    print(k, k in s)
for l in s.splitlines():
    if 'ms_rows.py' in l and '--voice' in l: print('ROW LINE:', l[:220])
