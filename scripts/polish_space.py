"""Polish the HF Space UI: sleeker fonts, speed-highlight strip, chips that speak on click.

usage: python scripts/polish_space.py [--p1-repo REPO] [--p1-label LABEL]
Edits runs/local_v041/index.html idempotently (safe to re-run).
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = 'runs/local_v041/index.html'
ap = argparse.ArgumentParser()
ap.add_argument('--p1-repo', default='Suarify/kecil-tts-yati-v0.5.0')
ap.add_argument('--p1-label', default='v0.5 (SOTA P1)')
A = ap.parse_args()

s = open(P, encoding='utf-8').read()
orig = s

# ---------- 1. sleeker type: Plus Jakarta Sans body + Outfit display ----------
s = s.replace('family=Inter:wght@400;500;600&family=Sora:wght@600;700;800',
              'family=Outfit:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700')
s = s.replace('Sora,Inter,sans-serif', "Outfit,'Plus Jakarta Sans',sans-serif")
s = s.replace('Inter,system-ui,sans-serif', "'Plus Jakarta Sans',system-ui,sans-serif")
s = s.replace('Inter,sans-serif', "'Plus Jakarta Sans',sans-serif")

# ---------- 2. speed strip CSS ----------
CSS = '''/* --- live speed highlight --- */
.speedstrip{display:flex;align-items:center;gap:9px;background:linear-gradient(95deg,var(--tint),transparent);border:1px solid var(--line);border-radius:12px;padding:9px 12px;margin:0 0 12px;font-size:12.5px;color:var(--mut);letter-spacing:.005em}
.speedstrip .bolt{font-size:15px;line-height:1}
.speedstrip b{color:var(--teal-d);font-variant-numeric:tabular-nums;font-weight:700}
.speedstrip .bar{flex:1;height:6px;min-width:36px;background:var(--line);border-radius:99px;overflow:hidden}
.speedstrip .bar i{display:block;height:100%;width:50%;background:linear-gradient(90deg,var(--teal),#5fd6d6);border-radius:99px;transition:width .45s ease}
.speedstrip .cmp{color:var(--teal-d);font-weight:700;white-space:nowrap}
html[data-theme=dark] .speedstrip{background:linear-gradient(95deg,var(--tint),transparent)}
'''
if '.speedstrip{' not in s:
    s = s.replace('footer{max-width:860px', CSS + 'footer{max-width:860px', 1)

# ---------- 3. speed strip HTML: top row above the textarea ----------
HTML = '''    <div class="speedstrip" id="speedstrip"><span class="bolt">⚡</span><span id="spd1">—</span><span class="bar"><i id="spdbar"></i></span><span class="cmp" id="spd2"></span></div>
'''
if 'id="speedstrip"' not in s:
    needle = '  <section class="card"><h2 data-i18n="h2s">2 · Type Malay text</h2>\n'
    assert needle in s, 'text card header not found'
    s = s.replace(needle, needle + HTML, 1)

# ---------- 4. SPEED benchmarks + strip JS ----------
SPEED_JS = '''// ---- speed benchmarks (1 CPU thread, 24 kHz audio) used by the live highlight ----
const SPEED = {
  'Suarify/kecil-tts-yati-v0.4.1': {rt: 21, label: 'v0.4.1', base: 20, baseLabel: 'v0.4'},
  'Suarify/kecil-tts-yati-v0.4': {rt: 20, label: 'v0.4', base: 19, baseLabel: 'v0.3'},
  'Suarify/kecil-tts-michael-v0.3.1': {rt: 19, label: 'v0.3.1', base: 18, baseLabel: 'v0.3.0'},
  'Suarify/kecil-tts-michael-v0.3.0': {rt: 18, label: 'v0.3.0', base: 17, baseLabel: 'v0.2.2'},
  'Suarify/kecil-tts-nicole-v0.1': {rt: 19, label: 'nicole v0.1', base: 21, baseLabel: 'Yati v0.4.1'},
  '__P1__': {rt: 21, label: '__LABEL__', base: 21, baseLabel: 'v0.4.1'}
};
function speedStrip(measured){
  const info = SPEED[$('ver').value] || {rt: 21, label: 'this voice', base: 21, baseLabel: 'v0.4.1'};
  const rt = measured == null ? info.rt : measured;
  $('spd1').innerHTML = (measured == null ? '⚡ ' : '⚡ This run ') + '<b>' + rt.toFixed(1) + '× realtime</b> · ' + info.label;
  $('spdbar').style.width = Math.min(100, Math.round(rt / 40 * 100)) + '%';
  const pct = Math.round((rt / info.base - 1) * 100);
  $('spd2').textContent = (pct >= 0 ? '+' : '') + pct + '% vs ' + info.baseLabel + ' (' + info.base + '×)';
}
'''
SPEED_JS = SPEED_JS.replace('__P1__', A.p1_repo).replace('__LABEL__', A.p1_label)
if 'function speedStrip(' not in s:
    needle = 'const qs = new URLSearchParams(location.search);\n'
    assert needle in s, 'qs line not found'
    s = s.replace(needle, needle + SPEED_JS, 1)

# call on first draw + on every generation
s = s.replace("  $('card').href = s.card; $('note').textContent = s.note;\n",
              "  $('card').href = s.card; $('note').textContent = s.note;\n  if (typeof speedStrip === 'function') speedStrip();\n")
s = s.replace("  $('info').textContent = `${$('st').checked ? `first audio after ${firstT.toFixed(2)} s; ` : ''}${sec.toFixed(2)} s of audio in ${s.toFixed(2)} s on ${tts.device} (${(sec / s).toFixed(1)}x real time).`;\n",
              "  $('info').textContent = `${$('st').checked ? `first audio after ${firstT.toFixed(2)} s; ` : ''}${sec.toFixed(2)} s of audio in ${s.toFixed(2)} s on ${tts.device} (${(sec / s).toFixed(1)}x real time).`;\n  speedStrip(sec / s);\n")

# ---------- 5. chips fill the textarea AND speak ----------
old_chip = "b.onclick = () => { $('t').value = e; }; $('chips').appendChild(b); }"
new_chip = "b.onclick = () => { $('t').value = e; if (!$('go').disabled) $('go').click(); }; $('chips').appendChild(b); }"
assert old_chip in s, 'chip onclick not found'
s = s.replace(old_chip, new_chip, 1)

# ---------- 6. add P1 to the Yati version dropdown (idempotent) ----------
if A.p1_repo not in s:
    s = s.replace("versions: [['Suarify/kecil-tts-nicole-v0.1', 'nicole v0.1 (new)'], ['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1 (latest)']",
                  "versions: [['" + A.p1_repo + "', '" + A.p1_label + "'], ['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1'], ['Suarify/kecil-tts-nicole-v0.1', 'nicole v0.1']")

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('polished', P, '| P1:', A.p1_repo, A.p1_label)
