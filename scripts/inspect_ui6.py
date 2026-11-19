s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
keys = ['af_nicole', '3 · Runs', 'edgeh:', 'Understandable and the most accurate', 'note:', '<footer', '.cta-row', '.by{', 'header{', '</footer>', 'modal-card', '@keyframes']
for i, l in enumerate(s, 1):
    if any(k in l for k in keys):
        print(i, l[:260])
