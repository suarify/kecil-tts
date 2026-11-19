import re
s = open('runs/listen/sota_p3.html', encoding='utf-8').read()
code = re.findall(r'<script>(.*?)</script>', s, re.S)[0]
open('runs/_gate3_check.js', 'w', encoding='utf-8').write(code)
print('extracted', len(code))
