import re
s = open('runs/kaggle_sm21gan/sm-21-gan.log', encoding='utf-8', errors='replace').read().replace('\x00', '')
seen = []
for d in re.findall(r'"data":"(.*?)"\}', s):
    if re.match(r'step \d+ loss', d):
        seen.append(d.replace('\\n', ''))
for d in seen:
    if 'nan' in d.lower():
        print('FIRST NAN:', d[:240])
        break
print('--- first 16 step lines ---')
for d in seen[:16]:
    print(d[:200])
