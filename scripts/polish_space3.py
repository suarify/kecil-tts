"""Space UI tuning v3 (all 5 owner requests). Idempotent-ish; asserts guard each edit."""
import re

P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# ---------- CSS additions ----------
CSS = '''
/* --- v3 tuning --- */
#stats,#wave{display:none}
#stats.on{display:grid}
#wave.on{display:block}
.infobtn{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.3);color:#fff;border-radius:50%;width:30px;height:30px;font:700 17px Outfit,sans-serif;cursor:pointer;margin-left:10px;vertical-align:middle;line-height:1}
.infobtn:hover{background:rgba(255,255,255,.26)}
.modal-wrap{position:fixed;inset:0;background:rgba(4,14,24,.72);z-index:100;display:flex;align-items:center;justify-content:center;padding:20px}
.modal-wrap[hidden]{display:none}
.modal-card{max-width:720px;width:100%;max-height:86vh;overflow:auto;position:relative;margin:0}
.modal-close{position:absolute;top:8px;right:12px;background:transparent;border:0;color:var(--mut);font:700 26px Outfit,sans-serif;cursor:pointer;line-height:1}
.modal-close:hover{color:var(--teal-d)}
.seo-hidden{height:.1px!important;opacity:.001!important;overflow:hidden!important;padding:0!important;margin:0!important;border:0!important;box-shadow:none!important}
.cta-row{margin-top:10px;padding-top:12px;border-top:1px solid var(--line);font-size:12.5px;line-height:1.5}
.cta-row a{color:var(--teal-d)}
'''
assert '.modal-wrap{' not in s
s = s.replace('</style>', CSS + '</style>', 1)

# ---------- 1 + 3: info icon beside header, keep topbar right on mobile ----------
old_h1 = '<h1>Kecil-TTS <span style="color:#7fd6d6">8M</span></h1><div class="by"'
new_h1 = '<h1>Kecil-TTS <span style="color:#7fd6d6">8M</span></h1><button class="infobtn" id="infoBtn" title="About / Info">&#9432;</button><div class="by"'
assert old_h1 in s
s = s.replace(old_h1, new_h1, 1)

old_media = '@media(max-width:560px){.spk{grid-template-columns:1fr}.badge{width:52px;height:52px}.badge img{width:34px}.edge{grid-template-columns:1fr}.headrow{flex-wrap:wrap}}'
new_media = '@media(max-width:560px){.spk{grid-template-columns:1fr}.badge{width:46px;height:46px}.badge img{width:30px}.edge{grid-template-columns:1fr}.headrow{flex-wrap:nowrap}.topbar{margin-left:auto}.tg{padding:6px 9px;font-size:11px}h1{font-size:22px}}'
assert old_media in s
s = s.replace(old_media, new_media, 1)

# ---------- 2: edge card -> modal ----------
old_edge_start = '  <section class="card" id="edge"><h2 data-i18n="edgeh">3 · Runs anywhere: edge, IoT, kiosk, vending machine</h2>'
new_edge_start = '  <div class="modal-wrap" id="edgeModal" hidden><section class="card modal-card" id="edge"><button class="modal-close" id="edgeClose" aria-label="Close">&#215;</button><h2 data-i18n="edgeh">3 · Runs anywhere: edge, IoT, kiosk, vending machine</h2>'
assert old_edge_start in s
s = s.replace(old_edge_start, new_edge_start, 1)

old_edge_end = '    </ul>\n  </section>\n  <section class="card"><h2 data-i18n="kwh">Keywords</h2>'
new_edge_end = '    </ul>\n  </section></div>\n  <section class="card seo-hidden"><h2 data-i18n="kwh">Keywords</h2>'
assert old_edge_end in s
s = s.replace(old_edge_end, new_edge_end, 1)

# ---------- 4: keywords -> hidden sentence ----------
m = re.search(r'    <div class="tags">.*?    </div>\n  </section>', s, re.S)
assert m, 'tags block not found'
keywords = re.findall(r'<span>(.*?)</span>', m.group(0))
sentence = 'Kecil-TTS-8M is a ' + ', '.join(keywords) + '.'
s = s[:m.start()] + '    <p>' + sentence + '</p>\n  </section>' + s[m.end():]
print('keywords in hidden sentence:', len(keywords))

# ---------- 5: CTA -> footer row, contact as sentence ----------
cta = '''  <section class="card cta"><h2 data-i18n="ctah">Need custom software or an enterprise voice-AI system?</h2>
    <p data-i18n="ctap">Custom voices, on-premise deployment, call-centre agents and integrations for Malaysian businesses. Tell us what you need.</p>
    <a class="btn" href="https://suarify.my/contact-form" target="_blank" rel="noopener">Contact Suarify &rarr; suarify.my/contact-form</a></section>
'''
assert cta in s
s = s.replace(cta, '', 1)

old_footer = '<footer>Kecil-TTS-8M is open source'
new_footer = '''<footer>
<div class="cta-row"><b data-i18n="ctah">Need custom software or an enterprise voice-AI system?</b> <span data-i18n="ctap">Custom voices, on-premise deployment, call-centre agents and integrations for Malaysian businesses. Tell us what you need.</span> Contact Suarify &rarr; <a href="https://suarify.my/contact-form" target="_blank" rel="noopener">suarify.my/contact-form</a></div>
Kecil-TTS-8M is open source'''
assert old_footer in s
s = s.replace(old_footer, new_footer, 1)

# ---------- JS: modal handlers + reveal stats/wave ----------
old_js = 'load();\n</script>'
new_js = '''// v3: info modal
$('infoBtn').onclick = () => { $('edgeModal').hidden = false; };
$('edgeClose').onclick = () => { $('edgeModal').hidden = true; };
$('edgeModal').onclick = (e) => { if (e.target === $('edgeModal')) $('edgeModal').hidden = true; };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('edgeModal').hidden = true; });
load();
</script>'''
assert old_js in s
s = s.replace(old_js, new_js, 1)

old_reveal = '  speedStrip(sec / s, s, firstT, tts.device, sec);\n'
new_reveal = '  speedStrip(sec / s, s, firstT, tts.device, sec);\n  $(\'stats\').classList.add(\'on\'); $(\'wave\').classList.add(\'on\');\n'
assert old_reveal in s
s = s.replace(old_reveal, new_reveal, 1)

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('UI v3 written')
