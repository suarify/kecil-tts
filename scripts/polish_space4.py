"""Space UI tuning v4: smaller floating info icon + honest streaming timing."""
P = 'runs/local_v041/index.html'
s = open(P, encoding='utf-8').read()
orig = s

# ---- info icon: 30% smaller (30->21px), float up-right right after the word ----
old = '.infobtn{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.3);color:#fff;border-radius:50%;width:30px;height:30px;font:700 17px Outfit,sans-serif;cursor:pointer;margin-left:10px;vertical-align:middle;line-height:1}'
new = '.infobtn{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.3);color:#fff;border-radius:50%;width:21px;height:21px;font:700 12px Outfit,sans-serif;cursor:pointer;margin-left:8px;vertical-align:top;line-height:1;padding:0;transform:translateY(-4px)}'
assert old in s, 'infobtn css not found'
s = s.replace(old, new, 1)

# ---- label ids so stats can swap meaning in streaming mode ----
old = '<div class="stat"><span class="lbl">Time taken</span><span class="big sm" id="stTime">—</span></div>\n      <div class="stat"><span class="lbl">First audio</span><span class="big sm" id="stFirst">—</span></div>'
new = '<div class="stat"><span class="lbl" id="lblTime">Time taken</span><span class="big sm" id="stTime">—</span></div>\n      <div class="stat"><span class="lbl" id="lblFirst">First audio</span><span class="big sm" id="stFirst">—</span></div>'
assert old in s, 'stat labels not found'
s = s.replace(old, new, 1)

# ---- speedStrip: streaming-aware display ----
old = """  $('stTime').textContent = render == null ? '—' : (render < 1 ? (render * 1000).toFixed(0) + ' ms' : render.toFixed(2) + ' s');
  $('stFirst').textContent = first == null ? '—' : (first < 1 ? (first * 1000).toFixed(0) + ' ms' : first.toFixed(2) + ' s');"""
new = """  const fmt = (v) => v == null ? '—' : (v < 1 ? (v * 1000).toFixed(0) + ' ms' : v.toFixed(2) + ' s');
  if (streaming) {
    $('lblTime').textContent = 'Time to audio'; $('stTime').textContent = fmt(first);
    $('lblFirst').textContent = 'Total gen';    $('stFirst').textContent = fmt(render);
  } else {
    $('lblTime').textContent = 'Time taken';    $('stTime').textContent = fmt(render);
    $('lblFirst').textContent = 'First audio';  $('stFirst').textContent = fmt(first);
  }"""
assert old in s, 'speedStrip stat lines not found'
s = s.replace(old, new, 1)

s = s.replace('function speedStrip(measured, render, first, dev, audiolen){',
              'function speedStrip(measured, render, first, dev, audiolen, streaming){', 1)
assert 'function speedStrip(measured, render, first, dev, audiolen, streaming){' in s

# ---- move t0 to measure only generation, and pass streaming flag ----
old = "$('go').onclick = async () => {\n  $('go').disabled = true; const t0 = performance.now();\n  await ensureAudio();\n  $('ph').textContent = tts.phonemize($('t').value).join(' | ');\n"
new = """$('go').onclick = async () => {
  $('go').disabled = true;
  await ensureAudio();
  $('ph').textContent = tts.phonemize($('t').value).join(' | ');
  const t0 = performance.now();   // time ONLY synthesis, not audio-context/FX setup
"""
assert old in s, 'onclick header not found'
s = s.replace(old, new, 1)

old = "  speedStrip(sec / s, s, firstT, tts.device, sec);"
new = "  speedStrip(sec / s, s, firstT, tts.device, sec, $('st').checked);"
assert old in s
s = s.replace(old, new, 1)

# clarify the info sentence for streaming
old = "  $('info').textContent = `${$('st').checked ? `first audio after ${firstT.toFixed(2)} s; ` : ''}${sec.toFixed(2)} s of audio in ${s.toFixed(2)} s on ${tts.device} (${(sec / s).toFixed(1)}x real time).`;"
new = "  $('info').textContent = `${$('st').checked ? `first audio after ${firstT.toFixed(2)} s (streaming starts playback sooner; total synthesis is unchanged); ` : ''}${sec.toFixed(2)} s of audio in ${s.toFixed(2)} s on ${tts.device} (${(sec / s).toFixed(1)}x real time).`;"
assert old in s, 'info line not found'
s = s.replace(old, new, 1)

assert s != orig
open(P, 'w', encoding='utf-8').write(s)
print('UI v4 written')
