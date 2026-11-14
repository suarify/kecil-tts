"""Build runs/listen/pilot.html from runs/k03/pilot_scores.csv.

Serve with:  python -m http.server 8765 --directory <repo>/runs   then open /listen/pilot.html
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
d = pd.read_csv(os.path.join(RUNS, 'k03', 'pilot_scores.csv'), dtype={'id': str}).rename(columns={'Unnamed: 0': 'id'})
if 'id' not in d.columns:
    d = d.rename(columns={d.columns[0]: 'id'})
d = d.sort_values(['category', 'id'], ascending=[True, True])
rows = []
for r in d.itertuples():
    rows.append(dict(id=r.id, cat=r.category, text=r.text, teacher=r.teacher,
                     raw='../k03/' + r.rel_raw if isinstance(r.rel_raw, str) and r.rel_raw else '',
                     t2='../k03/' + r.rel, t2v='../k03/' + r.rel_v,
                     cer=round(float(r.cer), 3), vcer=round(float(r.v_cer), 3),
                     sim=round(float(r.sim), 2), vsim=round(float(r.v_sim), 2),
                     trimmed=float(r.trimmed), hyp=str(r.hyp), retried=bool(getattr(r, 'retried', False) == True)))  # noqa: E712

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Malay Bella pilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6f5e;--bad:#b4442c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#6fbfa6;--bad:#e07a5f}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1150px}
h1{font-size:22px;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 14px}
#bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center;font-variant-numeric:tabular-nums}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
.row{border-top:1px solid var(--line);padding:12px 0}
.t{font-weight:600}.tt{font-size:13px;color:var(--mut)}.cat{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.g{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:6px}
.c{border:1px solid var(--line);border-radius:8px;padding:8px}.c.pick{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
audio{width:100%;height:32px}.s{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}.bad{color:var(--bad)}
.flags{margin-top:6px;font-size:13px;display:flex;gap:14px;flex-wrap:wrap}
@media (max-width:760px){.g{grid-template-columns:1fr}}
</style></head><body>
<h1>Malay Bella pilot: 150 sentences</h1>
<p class="m">Raw = Chatterbox output untouched. A = trimmed. B = trimmed then voice-converted toward real Bella. Pick A or B (closer to Bella, still natural Malay), and tick a flag if something is wrong. Click "Copy results" when done and paste them to Claude.</p>
<div id="bar"><span id="tally"></span><button id="copy">Copy results</button><label><input type="checkbox" id="only"> only clips I flagged</label></div>
<div id="list"></div>
<script>
const ROWS=__ROWS__;
let st={};try{st=JSON.parse(localStorage.getItem("pilot")||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem("pilot",JSON.stringify(st))}catch(e){}};
const FLAGS=[["pron","mispronounced"],["cut","cut too early"],["tail","junk at end"],["voice","not Bella"]];
function tally(){let a=0,b=0,f={};for(const v of Object.values(st)){if(v.pick=="A")a++;if(v.pick=="B")b++;for(const k of v.flags||[])f[k]=(f[k]||0)+1}
 document.getElementById("tally").textContent=`Picked ${a+b}/${ROWS.length} · A trimmed ${a} · B voice-converted ${b} · flags `+FLAGS.map(([k,l])=>`${l} ${f[k]||0}`).join(", ")}
function render(){const L=document.getElementById("list");L.innerHTML="";const only=document.getElementById("only").checked;
 for(const r of ROWS){const s=st[r.id]||(st[r.id]={flags:[]});if(only&&!s.flags.length)continue;
  const d=document.createElement("div");d.className="row";
  d.innerHTML=`<div class="cat">${r.cat} · ${r.id}${r.retried?" · retried":""}${r.trimmed>0.3?` · trimmed ${r.trimmed.toFixed(1)}s`:""}</div>
  <div class="t"></div><div class="tt"></div><div class="g"></div><div class="flags"></div>`;
  d.querySelector(".t").textContent=r.text;
  if(r.teacher!==r.text)d.querySelector(".tt").textContent="teacher read: "+r.teacher;
  const g=d.querySelector(".g");
  const cell=(lab,src,info,key)=>{const e=document.createElement("div");e.className="c"+(key&&s.pick==key?" pick":"");
   e.innerHTML=`<div>${key?`<label><input type="radio" name="${r.id}" ${s.pick==key?"checked":""}> <b>${lab}</b></label>`:`<b>${lab}</b>`}</div><audio controls preload="none" src="${src}"></audio><div class="s">${info}</div>`;
   if(key)e.querySelector("input").onchange=()=>{s.pick=key;save();render();tally()};g.appendChild(e)};
  cell("Raw",r.raw,"untrimmed",null);
  cell("A trimmed",r.t2,`CER <span class="${r.cer>0.1?"bad":""}">${(r.cer*100).toFixed(1)}%</span> · sim ${r.sim}`,"A");
  cell("B voice-converted",r.t2v,`CER <span class="${r.vcer>0.1?"bad":""}">${(r.vcer*100).toFixed(1)}%</span> · sim ${r.vsim}`,"B");
  const fl=d.querySelector(".flags");
  for(const [k,l] of FLAGS){const x=document.createElement("label");x.innerHTML=`<input type="checkbox" ${s.flags.includes(k)?"checked":""}> ${l}`;
   x.querySelector("input").onchange=(ev)=>{s.flags=ev.target.checked?[...s.flags,k]:s.flags.filter(z=>z!=k);save();tally()};fl.appendChild(x)}
  L.appendChild(d)}}
document.getElementById("only").onchange=render;
document.getElementById("copy").onclick=()=>{const out=Object.entries(st).filter(([k,v])=>v.pick||v.flags.length).map(([k,v])=>`${k}\t${v.pick||"-"}\t${v.flags.join(",")}`).join("\n");
 navigator.clipboard.writeText(document.getElementById("tally").textContent+"\n"+out).then(()=>alert("Copied"))};
render();tally();
</script></body></html>'''
os.makedirs(os.path.join(RUNS, 'listen'), exist_ok=True)
with open(os.path.join(RUNS, 'listen', 'pilot.html'), 'w', encoding='utf-8') as f:
    f.write(HTML.replace('__ROWS__', json.dumps(rows, ensure_ascii=False)))
print(len(rows), 'rows ->', os.path.join(RUNS, 'listen', 'pilot.html'))
