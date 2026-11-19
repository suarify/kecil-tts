import json
import re

s = open('runs/sm21_kernel_logs.txt', encoding='utf-8', errors='replace').read()
for line in s.splitlines():
    if 'input:' in line or 'python 3' in line or 'EMPTY INPUT' in line:
        print(line[:400].replace('\\n', ' '))
