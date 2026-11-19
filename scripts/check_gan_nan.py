import re
s = open('runs/kaggle_sm21gan/sm-21-gan.log', encoding='utf-8', errors='replace').read().replace('\x00', '')
for line in s.splitlines():
    if re.search(r'step \d+ loss', line):
        d = re.search(r'"data":"(step \d+ loss.*?)"\}', line)
        if d:
            t = d.group(1)
            if 'nan' in t.lower():
                print('NAN FIRST:', t[:220].replace('\\n', ''))
                break
    if 'joint' in line.lower() or 'WARM' in line:
        pass
# print 40 lines around the transition
idx = s.find('step 500 loss')
print('--- around step 500 ---')
for d in re.findall(r'"data":"(.*?)"\}', s):
    if 'step' in d and 'loss' in d and ('5' in d[:10] or '6' in d[:10]):
        print(d[:220].replace('\\n', ''))
