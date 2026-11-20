"""Space UI v7: effects-lab modal (gear icon), buttons on own row, tiny info line above textarea."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# ---- CSS ----
CSS = '''
/* v7: gear icon + tiny status line */
.gear{background:transparent;border:1px solid var(--line);color:var(--mut);border-radius:8px;width:30px;height:30px;font-size:15px;line-height:1;cursor:pointer;display:inline-grid;place-items:center;vertical-align:middle}
.gear:hover{border-color:var(--teal);color:var(--teal-d)}
.micro{font-size:10px!important;line-height:1.35!important;color:var(--mut)!important;margin:4px 0 2px!important;word-break:break-word}
.micro code{font-size:9.5px!important}
'''
assert '.gear{' not in s
s = s.replace('</style>', CSS + '</style>', 1)

# ---- A. move info + phonemes to a tiny line right under the waveform (above textarea) ----
old = '''    <canvas id="wave" class="wave" width="760" height="92" aria-label="Waveform of the generated speech"></canvas>
    <textarea id="t" rows="3">Selamat pagi, boleh saya bantu encik hari ini?</textarea>
    <div class="chips row" id="chips"></div>
    <div class="row">
      <button id="go" class="go" disabled>Loading model…</button>
      <button id="replay" class="sec" disabled>Replay</button>
      <label class="m">speed <input id="sp" type="range" min="0.7" max="1.3" step="0.05" value="1"></label>
      <label class="m"><input id="st" type="checkbox" checked> stream</label>
    </div>
    <p class="info" id="info"></p>
    <p class="info">Phonemes: <code id="ph"></code></p>'''
new = '''    <canvas id="wave" class="wave" width="760" height="92" aria-label="Waveform of the generated speech"></canvas>
    <p class="micro" id="info"></p>
    <p class="micro">Phonemes: <code id="ph"></code></p>
    <textarea id="t" rows="3">Selamat pagi, boleh saya bantu encik hari ini?</textarea>
    <div class="chips row" id="chips"></div>
    <div class="row">
      <label class="m">speed <input id="sp" type="range" min="0.7" max="1.3" step="0.05" value="1"></label>
      <label class="m"><input id="st" type="checkbox" checked> stream</label>
      <button class="gear" id="fxBtn" title="Effects lab">⚙</button>
    </div>
    <div class="row">
      <button id="go" class="go" disabled>Loading model…</button>
      <button id="replay" class="sec" disabled>Replay</button>
    </div>'''
assert old in s, 'text block not found'
s = s.replace(old, new, 1)

# ---- B. effects lab section -> modal ----
old = '''  <section class="card"><details id="fxd"><summary data-i18n="fxh">Effects lab (optional)</summary>
    <p class="info">Shape the voice after it is made: tone, a delayed second layer (doubling / chorus), an echo tail and a small room. Everything runs in your browser, live while it plays. Pick a preset or move the sliders (double-click a slider to reset it).</p>
    <div class="row"><select id="pre"></select><button class="sec" id="fxreset">Reset</button><button class="sec" id="dlraw" disabled>Download wav</button><button class="sec" id="dlfx" disabled>Download wav with effects</button></div>
    <div class="fx" id="fx"></div>
  </details></section>'''
new = '''  <div class="modal-wrap" id="fxModal" hidden><section class="card modal-card" id="fxCard"><button class="modal-close" id="fxClose" aria-label="Close">&#215;</button><h2 data-i18n="fxh">Effects lab (optional)</h2>
    <p class="info">Shape the voice after it is made: tone, a delayed second layer (doubling / chorus), an echo tail and a small room. Everything runs in your browser, live while it plays. Pick a preset or move the sliders (double-click a slider to reset it).</p>
    <div class="row"><select id="pre"></select><button class="sec" id="fxreset">Reset</button><button class="sec" id="dlraw" disabled>Download wav</button><button class="sec" id="dlfx" disabled>Download wav with effects</button></div>
    <div class="fx" id="fx"></div>
  </section></div>'''
assert old in s, 'effects section not found'
s = s.replace(old, new, 1)

# ---- C. JS handlers for fx modal ----
old = "$('infoBtn').onclick = () => { $('edgeModal').hidden = false; };"
new = """$('infoBtn').onclick = () => { $('edgeModal').hidden = false; };
$('fxBtn').onclick = () => { $('fxModal').hidden = false; };
$('fxClose').onclick = () => { $('fxModal').hidden = true; };
$('fxModal').onclick = (e) => { if (e.target === $('fxModal')) $('fxModal').hidden = true; };"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace("document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('edgeModal').hidden = true; });",
              "document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('edgeModal').hidden = true; $('fxModal').hidden = true; } });", 1)
assert "$('fxBtn').onclick" in s

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('UI v7 written')
