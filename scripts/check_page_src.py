import re
s = open('runs/listen/sota_p2.html', encoding='utf-8').read()
srcs = re.findall(r'data-src="([^"]+)"', s)
print('SRC sample:', srcs[:3])
print('total', len(srcs))
print('all relative:', all(not x.startswith(('runs/', '/', 'C:')) for x in srcs))
