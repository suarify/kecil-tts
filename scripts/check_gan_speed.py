import re
s = open('runs/kaggle_sm21gan/sm-21-gan.log', encoding='utf-8', errors='replace').read().replace('\x00', '')
for line in s.splitlines():
    if re.search(r'step (50|100|1000|2000|4000|8000) loss', line):
        d = re.search(r'"data":"(.*?)"\}', line)
        if d:
            print(d.group(1)[:240].replace('\\n', ''))
