s = open('kaggle/sm29_nicole_train/sm29_nicole_train.py', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if '--voice' in l or 'ms_rows.py' in l or 'f0-min' in l or 'male' in l.lower() or 'female' in l.lower():
        print(i, l[:230])
