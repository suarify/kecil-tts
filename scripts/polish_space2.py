"""HF Space UI upgrade v2: Nicole as a real speaker, big stats panel + live waveform, chips auto-speak.

Run after scripts/polish_space.py. Idempotent.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# ---------- 1. speaker grid fits 3 speakers ----------
s = s.replace('.spk{display:grid;grid-template-columns:1fr 1fr;gap:10px}',
              '.spk{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}')

# ---------- 2. Nicole becomes a third speaker; remove her from Yati's version list ----------
s = s.replace("versions: [['Suarify/kecil-tts-nicole-v0.1', 'nicole v0.1 (new)'], ['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1 (latest)']",
              "versions: [['Suarify/kecil-tts-yati-v0.5.0', 'v0.5 (SOTA P1)'], ['Suarify/kecil-tts-yati-v0.4.1', 'v0.4.1']")
old_michael_end = "    versions: [['Suarify/kecil-tts-michael-v0.3.1', 'v0.3.1 (latest)'], ['Suarify/kecil-tts-michael-v0.3.0', 'v0.3.0'], ['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2'], ['Suarify/kecil-tts-michael-v0.2', 'v0.2'], ['Suarify/kecil-tts-michael-v0.1', 'v0.1']] }\n};"
new_michael_end = """    versions: [['Suarify/kecil-tts-michael-v0.3.1', 'v0.3.1 (latest)'], ['Suarify/kecil-tts-michael-v0.3.0', 'v0.3.0'], ['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2'], ['Suarify/kecil-tts-michael-v0.2', 'v0.2'], ['Suarify/kecil-tts-michael-v0.1', 'v0.1']] },
  nicole: { name: 'Nicole', sub: 'female voice · af_nicole', card: 'https://huggingface.co/Suarify/kecil-tts-nicole-v0.1',
    note: 'Female voice trained on Yati clips converted once to the af_nicole reference. v0.1 is an early release with a slight watery texture; a cleaner version is in progress.',
    versions: [['Suarify/kecil-tts-nicole-v0.1', 'v0.1']] }
};"""
assert old_michael_end in s, 'michael block not found'
s = s.replace(old_michael_end, new_michael_end, 1)

old_detect = "let speaker = SPEAKERS[qs.get('speaker')] ? qs.get('speaker') : (qs.get('model') || '').includes('michael') ? 'michael' : (qs.get('speaker') === 'suaradee' ? 'yati' : 'yati');"
new_detect = "let speaker = SPEAKERS[qs.get('speaker')] ? qs.get('speaker') : (qs.get('model') || '').includes('michael') ? 'michael' : (qs.get('model') || '').includes('nicole') ? 'nicole' : 'yati';"
assert old_detect in s, 'speaker detection not found'
s = s.replace(old_detect, new_detect, 1)

# ---------- 3a. replace the small speed strip CSS with the big stats panel ----------
i = s.find('.speedstrip{')
j = s.find('html[data-theme=dark] .speedstrip{background:linear-gradient(95deg,var(--tint),transparent)}')
assert i >= 0 and j >= 0 and j > i
j += len('html[data-theme=dark] .speedstrip{background:linear-gradient(95deg,var(--tint),transparent)}')
NEW_CSS = '''/* --- live render stats + waveform --- */
.stats{display:grid;grid-template-columns:1.35fr repeat(4,1fr);gap:10px 14px;background:linear-gradient(100deg,var(--tint),transparent);border:1px solid var(--line);border-radius:14px;padding:13px 15px 11px;margin:0 0 10px}
.stat{display:flex;flex-direction:column;gap:3px;min-width:0}
.stat .lbl{font:700 9.5px Outfit,'Plus Jakarta Sans',sans-serif;letter-spacing:.13em;color:var(--mut);text-transform:uppercase;white-space:nowrap}
.stat .big{font:800 27px/1.05 Outfit,'Plus Jakarta Sans',sans-serif;color:var(--teal-d);font-variant-numeric:tabular-nums;letter-spacing:-.035em}
.stat .big.sm{font-size:17px;color:var(--navy);letter-spacing:-.01em}
.stat .sub{font-size:11px;color:var(--teal-d);font-weight:700;white-space:nowrap}
.stats .bar{grid-column:1/-1;height:5px;background:var(--line);border-radius:99px;overflow:hidden;margin-top:2px}
.stats .bar i{display:block;height:100%;width:50%;background:linear-gradient(90deg,var(--teal),#5fd6d6);border-radius:99px;transition:width .45s ease}
.wave{width:100%;height:92px;display:block;border:1px solid var(--line);border-radius:12px;background:var(--bg);margin:0 0 12px}
html[data-theme=dark] .stat .big.sm{color:#dcebf4}
@media(max-width:620px){.stats{grid-template-columns:1fr 1fr}.stat .big{font-size:24px}}'''
s = s[:i] + NEW_CSS + s[j:]

# ---------- 3b. replace the strip HTML with the stats panel + canvas ----------
old_html = '    <div class="speedstrip" id="speedstrip"><span class="bolt">⚡</span><span id="spd1">—</span><span class="bar"><i id="spdbar"></i></span><span class="cmp" id="spd2"></span></div>\n'
new_html = '''    <div class="stats" id="stats">
      <div class="stat"><span class="lbl">⚡ Render speed</span><span class="big" id="stRt">—</span><span class="sub" id="stCmp"></span></div>
      <div class="stat"><span class="lbl">Time taken</span><span class="big sm" id="stTime">—</span></div>
      <div class="stat"><span class="lbl">First audio</span><span class="big sm" id="stFirst">—</span></div>
      <div class="stat"><span class="lbl">Audio length</span><span class="big sm" id="stLen">—</span></div>
      <div class="stat"><span class="lbl">Device</span><span class="big sm" id="stDev">—</span></div>
      <div class="bar"><i id="spdbar"></i></div>
    </div>
    <canvas id="wave" class="wave" width="760" height="92" aria-label="Waveform of the generated speech"></canvas>
'''
assert old_html in s, 'old speedstrip HTML not found'
s = s.replace(old_html, new_html, 1)

# ---------- 3c. replace speedStrip() JS and add waveform drawing ----------
i = s.find('function speedStrip(measured){')
j = s.find('\nlet speaker = ', i)
assert i >= 0 and j > i
NEW_JS = '''function speedStrip(measured, render, first, dev, audiolen){
  const info = SPEED[$('ver').value] || {rt: 21, label: 'this voice', base: 21, baseLabel: 'v0.4.1'};
  const rt = measured == null ? info.rt : measured;
  const pct = Math.round((rt / info.base - 1) * 100);
  $('stRt').textContent = rt.toFixed(1) + '×';
  $('stCmp').textContent = (measured == null ? info.label + ' · ' : '') + (pct >= 0 ? '+' : '') + pct + '% vs ' + info.baseLabel;
  $('stTime').textContent = render == null ? '—' : (render < 1 ? (render * 1000).toFixed(0) + ' ms' : render.toFixed(2) + ' s');
  $('stFirst').textContent = first == null ? '—' : (first < 1 ? (first * 1000).toFixed(0) + ' ms' : first.toFixed(2) + ' s');
  $('stLen').textContent = audiolen == null ? '—' : audiolen.toFixed(1) + ' s';
  $('stDev').textContent = dev || (typeof tts !== 'undefined' && tts ? tts.device : 'auto');
  $('spdbar').style.width = Math.min(100, Math.round(rt / 40 * 100)) + '%';
}
// --- waveform with a moving playhead ---
let waveRAF = null, waveStart = 0;
function drawWave(audio, progress){
  const c = $('wave'); if (!c) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = c.width = Math.max(1, Math.floor(c.clientWidth * dpr));
  const H = c.height = Math.floor(92 * dpr);
  const g = c.getContext('2d'), mid = H / 2;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = 'rgba(120,150,170,.28)'; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, mid); g.lineTo(W, mid); g.stroke();
  if (!audio || !audio.length) return;
  const col = getComputedStyle(document.documentElement).getPropertyValue('--teal').trim() || '#1f9a9a';
  g.strokeStyle = col; g.lineWidth = Math.max(1, dpr * 0.8);
  const step = Math.max(1, Math.floor(audio.length / W));
  g.beginPath();
  for (let px = 0; px < W; px++){
    let mn = 1, mx = -1;
    for (let k = px * step; k < (px + 1) * step && k < audio.length; k++){ const v = audio[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mn > mx){ mn = 0; mx = 0; }
    g.moveTo(px + 0.5, mid - mx * mid * 0.92); g.lineTo(px + 0.5, mid - mn * mid * 0.92);
  }
  g.stroke();
  if (progress != null && progress >= 0){
    const x = Math.max(0, Math.min(W, progress * W));
    g.strokeStyle = '#ff7a59'; g.lineWidth = Math.max(1.5, dpr * 1.2);
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
  }
}
function playhead(){
  if (!lastAudio || !AC){ waveRAF = null; return; }
  const el = AC.currentTime - waveStart, dur = lastAudio.length / 24000;
  if (el < -0.2 || el > dur + 0.2){ drawWave(lastAudio, null); waveRAF = null; return; }
  drawWave(lastAudio, Math.max(0, el / dur));
  waveRAF = requestAnimationFrame(playhead);
}
function startPlayhead(){ if (waveRAF) cancelAnimationFrame(waveRAF); waveRAF = requestAnimationFrame(playhead); }
window.addEventListener('resize', () => drawWave(lastAudio, null));
'''
s = s[:i] + NEW_JS + s[j:]

# ---------- 3d. hook generation + replay into the stats/waveform ----------
old_call = "  speedStrip(sec / s);\n"
new_call = """  speedStrip(sec / s, s, firstT, tts.device, sec);
  waveStart = AC.currentTime + 0.06; drawWave(lastAudio, 0); startPlayhead();
"""
assert old_call in s, 'speedStrip call not found'
s = s.replace(old_call, new_call, 1)

old_replay = "$('replay').onclick = async () => { if (!lastAudio) return; await ensureAudio(); playChunk(lastAudio, AC.currentTime + 0.03); };"
new_replay = "$('replay').onclick = async () => { if (!lastAudio) return; await ensureAudio(); waveStart = AC.currentTime + 0.03; playChunk(lastAudio, waveStart); startPlayhead(); };"
assert old_replay in s, 'replay handler not found'
s = s.replace(old_replay, new_replay, 1)

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('upgraded', P)
