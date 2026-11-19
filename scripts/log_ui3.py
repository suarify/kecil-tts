import subprocess

msg = """
## 2026-11-20 Space UI tuning v3 (5 owner requests) - uploaded + verified live
- (1) Stats panel + waveform hidden by default (`#stats,#wave{display:none}`), revealed only after speech is generated (`.on` class added in the Speak handler).
- (2) The '3 · Runs anywhere: edge, IoT, kiosk...' card is now a modal popup (`.modal-wrap#edgeModal`), opened by a new ⓘ info icon beside the Kecil-TTS header, closed via × / backdrop click / Escape.
- (3) On mobile the BM/light toggles stay in the top-right row (`.headrow{flex-wrap:nowrap}`, `.topbar{margin-left:auto}`, smaller tg/badge/h1 instead of wrapping).
- (4) Keywords card is visually hidden (`.seo-hidden`: height .1px, opacity .001) with all 37 keywords written as one sentence inside for SEO.
- (5) The 'Need custom software...' CTA section card was removed; its content now sits in a footer row (`.cta-row`) and 'Contact Suarify → suarify.my/contact-form' is a plain sentence with an inline link, not a button.
- Verified live in browser: model loads ("Model ready on webgpu in 5.5 s"), Speak active, 3 speakers, stats/wave hidden, modal opens/closes, keywords opacity .001, CTA row present, no CTA button, zero JS errors.
"""
with open('STATUS.md', 'a', encoding='utf-8') as f:
    f.write(msg)
files = ['STATUS.md', 'scripts/polish_space3.py', 'scripts/verify_space3.py', 'scripts/inspect_ui5.py',
         'scripts/inspect_css5.py']
subprocess.run(['git', 'add'] + files, check=True)
subprocess.run(['git', 'commit', '-m', 'Space UI tuning v3: hidden stats, info modal, mobile toggles, SEO keywords, footer CTA', '--quiet'])
print('committed')
