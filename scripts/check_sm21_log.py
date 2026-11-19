import json
import re

s = open('runs/kaggle_sm21/sm-21-f0uv-text-head.log', encoding='utf-8', errors='replace').read()
for m in re.finditer(r'"data":"(.*?)"\}', s):
    d = m.group(1)
    if 'before:' in d or 'EVAL 500' in d or 'EVAL 1000' in d or 'UV text side' in d or 'rows' in d.lower():
        print(d[:320].replace('\\n', ' '))
