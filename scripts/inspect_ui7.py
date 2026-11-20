s = open('runs/local_v041/index.html', encoding='utf-8').read().splitlines()
for i, l in enumerate(s, 1):
    if any(k in l for k in ('id="fxd"', 'id="fx"', 'data-i18n="fxh"', 'id="go"', 'id="replay"',
                           'id="sp"', 'id="st"', 'id="info"', 'id="ph"', 'id="wave"',
                           '<textarea', 'class="chips', 'id="pre"', 'id="fxreset"', '</details>',
                           'modal-wrap', 'id="edgeModal"', 'infoBtn')):
        print(i, l[:220])
