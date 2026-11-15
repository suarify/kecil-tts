"""Build runs/listen/v03.html from runs/listen/v03/items.json (made by scripts/tonal_knob.py)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'runs/listen/v03/items.json'), encoding='utf-8'))
names = {'1.0': 'A: v0.2 as shipped', '0.5': 'B: half the noise', '0.25': 'C: a quarter of the noise'}
cols = [dict(key=str(k), name=names.get(str(k), f'noise x{k}')) for k in d['keep']]
html = '''<!doctype html><html><head><meta charset="utf-8"><title>Suaradee v0.3 check</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#0b6e4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#5fd3a8}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:980px}
h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:6px 0}.row{border-top:1px solid var(--line);padding:12px 0}
.t{font-weight:600}.g{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px}.g div{flex:1;min-width:230px}audio{width:100%;height:32px}
label{font-size:13px;color:var(--mut);display:block;margin-bottom:2px;cursor:pointer}input{accent-color:var(--acc)}
#sum{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);font-weight:600}
textarea{width:100%;height:54px;background:var(--bg);color:var(--fg);border:1px solid var(--line);font:12px monospace}
</style></head><body><h1>Suaradee v0.3 check: less noise in the voice</h1>
<p>Same model, same words, same timing. Only the mix inside voiced sound changes: A is v0.2 as shipped, B keeps half of the noise, C keeps a quarter. The removed noise energy is given to the harmonics, so loudness and brightness stay the same.</p>
<p>For each sentence tick the one you prefer (or "none is OK"). Listen for: is the damp / wet sound gone, and does C start to sound buzzy or robotic?</p>
<div id="sum"></div><div id="l"></div><p>Result to paste back:</p><textarea id="res" readonly></textarea>
<script>
const D=__DATA__,C=__COLS__,K='v03picks';let P={};try{P=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const l=document.getElementById('l');
D.forEach((it,i)=>{const r=document.createElement('div');r.className='row';
 r.innerHTML='<div class="t">'+(i+1)+'. '+it.text+'</div><div class="g">'+C.map(c=>'<div><label><input type="radio" name="p'+i+'" value="'+c.key+'"> '+c.name+'</label><audio controls preload="none" src="'+it.files[c.key]+'"></audio></div>').join('')+'</div><label><input type="radio" name="p'+i+'" value="none"> none is OK</label>';
 l.appendChild(r);
 r.querySelectorAll('input').forEach(x=>{if(P[it.id]===x.value)x.checked=true;x.onchange=()=>{P[it.id]=x.value;try{localStorage.setItem(K,JSON.stringify(P))}catch(e){}show()}})});
function show(){const n={};Object.values(P).forEach(v=>n[v]=(n[v]||0)+1);
 const s=C.map(c=>c.name.split(':')[0]+' '+(n[c.key]||0)).join(' | ')+' | none '+(n.none||0)+' | rated '+Object.keys(P).length+'/'+D.length;
 document.getElementById('sum').textContent=s;document.getElementById('res').value=s+'\\n'+JSON.stringify(P)}
show();
</script></body></html>'''
html = html.replace('__DATA__', json.dumps(d['items'], ensure_ascii=False)).replace('__COLS__', json.dumps(cols))
out = os.path.join(ROOT, 'runs/listen/v03.html')
open(out, 'w', encoding='utf-8').write(html)
print('wrote', out, len(d['items']), 'sentences')
