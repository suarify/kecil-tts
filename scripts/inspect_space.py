s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
keys = ['<body', 'chips', 'textarea', 'button.go', 'speak', 'versions:', 'dataset-src',
        'sample', 'class="info', 'id="voice', 'id="speed', 'onclick', 'function ', '<script']
for i, l in enumerate(s, 1):
    if any(k in l for k in keys):
        print(i, l[:240])
