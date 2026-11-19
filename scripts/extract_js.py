import re

s = open('runs/local_v041/index.html', encoding='utf-8').read()
mods = re.findall(r'<script type="module">(.*?)</script>', s, re.S)
plain = re.findall(r'<script>(.*?)</script>', s, re.S)
for i, code in enumerate(mods):
    open(f'runs/_check_mod{i}.mjs', 'w', encoding='utf-8').write(code)
for i, code in enumerate(plain):
    open(f'runs/_check_plain{i}.js', 'w', encoding='utf-8').write(code)
print('modules', len(mods), 'plain', len(plain))
