"""Space UI tuning v6: Nicole label, modal title, animated SVGs, footer gradient separator."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# 1. Nicole: drop af_nicole from sub + note
s = s.replace("sub: 'female voice · af_nicole'", "sub: 'female voice'", 1)
s = s.replace("note: 'Female voice trained on Yati clips converted once to the af_nicole reference. v0.1 is an early release with a slight watery texture; a cleaner version is in progress.'",
              "note: 'Female voice trained from Yati clips. v0.1 is an early release; a cleaner version is in progress.'", 1)
assert 'af_nicole' not in s, 'af_nicole still present'

# 2. modal title: remove the "3 ·" prefix (EN + BM + i18n)
s = s.replace('<h2 data-i18n="edgeh">3 · Runs anywhere: edge, IoT, kiosk, vending machine</h2>',
              '<h2 data-i18n="edgeh">Runs anywhere: edge, IoT, kiosk, vending machine</h2>', 1)
s = s.replace("edgeh: '3 · Runs anywhere: edge, IoT, kiosk, vending machine'",
              "edgeh: 'Runs anywhere: edge, IoT, kiosk, vending machine'", 1)
s = s.replace("edgeh: '3 · Berjalan di mana-mana: edge, IoT, kiosk, mesin layan diri'",
              "edgeh: 'Berjalan di mana-mana: edge, IoT, kiosk, mesin layan diri'", 1)
assert '3 · Runs' not in s and '3 · Berjalan' not in s

# 3. animate the modal SVGs gently
CSS = '''/* v6: gentle SVG motion + footer gradient separator */
@keyframes floaty{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-6px) rotate(.6deg)}}
.edge figure:first-child svg{animation:floaty 3.2s ease-in-out infinite}
.edge figure:last-child svg{animation:floaty 3.8s ease-in-out infinite .5s}
@media(prefers-reduced-motion:reduce){.edge figure svg{animation:none}}
footer{position:relative;background:linear-gradient(180deg,var(--tint),transparent);padding:22px 16px 40px;border-radius:0 0 16px 16px}
footer::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--teal),var(--navy2),var(--teal))}
footer::after{content:"";position:absolute;bottom:0;left:0;right:0;height:4px;background:linear-gradient(135deg,var(--navy),var(--teal))}
'''
# replace the existing footer rule (it currently sets padding etc.)
old_footer_css = 'footer{max-width:860px;margin:0 auto;padding:0 16px 40px;color:var(--mut);font-size:13px}'
assert old_footer_css in s, 'footer css not found'
s = s.replace(old_footer_css, CSS + 'footer{max-width:860px;margin:0 auto;color:var(--mut);font-size:13px}', 1)

# 4. Yati note removal (item 4 in owner list) - already targeted? do it explicitly
s = s.replace("    note: 'Understandable and the most accurate so far. v0.4.1 has cleaner, less watery sound than v0.4 (same voice, better audio export).',\n", "", 1)
assert 'Understandable and the most accurate' not in s

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('UI v6 written')
