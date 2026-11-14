"""Build runs/listen/transplant.html from runs/k05 (prosody-transplant proof).

Per sentence: the Chatterbox clip is shown LABELLED as the reference (not pickable, so it cannot
absorb every vote as it did at Gate A). Blind and shuffled below it: F1 (plain Kokoro teacher),
TPd (+ Chatterbox word timing), TPf (+ timing and pitch).
Serve:  python -m http.server 8765 --directory <repo>/runs   then open /listen/transplant.html
"""
import json
import os
import shutil
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
OUT = os.path.join(RUNS, 'listen', 'tp')
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, 'CB'))
with zipfile.ZipFile(os.path.join(RUNS, 'k05', 'clips_transplant.zip')) as z:
    z.extractall(OUT)

rows_df = pd.read_csv(os.path.join(RUNS, 'k05', 'transplant_rows.csv'))
ok = rows_df[rows_df['skipped'].isna()] if 'skipped' in rows_df else rows_df
rows = []
for r in ok.itertuples():
    ref = ''
    src = os.path.join(RUNS, 'k03', 'pilot', 'T2v', f'{r.id}.wav')
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, 'CB', f'{r.id}.wav'))
        ref = f'tp/CB/{r.id}.wav'
    clips = {n: f'tp/{n}/{r.id}.wav' for n in ('F1', 'TPd', 'TPf') if os.path.exists(os.path.join(OUT, n, f'{r.id}.wav'))}
    rows.append(dict(id=r.id, cat=r.category, text=r.norm, ref=ref, clips=clips))

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Rhythm transplant test</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#6fbfa6}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1000px}
h1{font-size:22px;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 14px}
#bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
.row{border-top:1px solid var(--line);padding:12px 0}.t{font-weight:600}.cat{font-size:12px;color:var(--mut);text-transform:uppercase}
.ref{display:flex;gap:10px;align-items:center;margin:6px 0;font-size:13px;color:var(--mut)}.ref audio{flex:1;max-width:420px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px;margin-top:6px}
.c{border:1px solid var(--line);border-radius:8px;padding:8px}.c.pick{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}.c.no{opacity:.55}
audio{width:100%;height:32px}.s{font-size:12px;color:var(--mut)}
</style></head><body>
<h1>Does borrowing Chatterbox's rhythm fix the Kokoro voice?</h1>
<p class="m">Each sentence shows the Chatterbox clip you liked as a labelled reference, then three unlabelled Kokoro versions. Pick the Kokoro version that sounds most like natural Malay. Tick "not acceptable" on any you would not use. Rate at least 12, then click Copy results and paste them to Claude.</p>
<div id="bar"><span id="tally"></span><button id="copy">Copy results</button></div><div id="list"></div>
<script>
const ROWS=__ROWS__;
const NAMES={F1:"F1 plain Kokoro",TPd:"TPd + Chatterbox timing",TPf:"TPf + timing and pitch"};
let st={};try{st=JSON.parse(localStorage.getItem("tp2")||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem("tp2",JSON.stringify(st))}catch(e){}};
const seed=s=>{let h=0;for(const ch of s)h=(h*31+ch.charCodeAt(0))|0;return h};
function tally(){const t={},no={};let n=0;for(const v of Object.values(st)){if(v.pick){n++;t[v.pick]=(t[v.pick]||0)+1}for(const k of v.no||[])no[k]=(no[k]||0)+1}
 document.getElementById("tally").textContent=`Picked ${n}/${ROWS.length} · picks: `+(Object.entries(t).map(([k,v])=>k+" "+v).join(", ")||"none")+" · not acceptable: "+(Object.entries(no).map(([k,v])=>k+" "+v).join(", ")||"none")}
function render(){const L=document.getElementById("list");L.innerHTML="";
 for(const r of ROWS){const s=st[r.id]||(st[r.id]={no:[]});const keys=Object.keys(r.clips).sort((a,b)=>((seed(a+r.id))&255)-((seed(b+r.id))&255));
  const d=document.createElement("div");d.className="row";d.innerHTML=`<div class="cat">${r.cat} · ${r.id}</div><div class="t"></div>`+(r.ref?`<div class="ref">Reference (Chatterbox) <audio controls preload="none" src="${r.ref}"></audio></div>`:"")+`<div class="g"></div>`;
  d.querySelector(".t").textContent=r.text;const g=d.querySelector(".g");
  keys.forEach((k,i)=>{const done=!!s.pick;const e=document.createElement("div");e.className="c"+(s.pick===k?" pick":"")+((s.no||[]).includes(k)?" no":"");
   e.innerHTML=`<label><input type="radio" name="${r.id}" ${s.pick===k?"checked":""}> <b>${"ABC"[i]}</b>${done?" — "+NAMES[k]:""}</label>
   <audio controls preload="none" src="${r.clips[k]}"></audio>
   <label class="s"><input type="checkbox" ${(s.no||[]).includes(k)?"checked":""}> not acceptable</label>`;
   e.querySelector("input[type=radio]").onchange=()=>{s.pick=k;save();render();tally()};
   e.querySelector("input[type=checkbox]").onchange=ev=>{s.no=ev.target.checked?[...(s.no||[]),k]:(s.no||[]).filter(z=>z!=k);save();render();tally()};
   g.appendChild(e)});L.appendChild(d)}}
document.getElementById("copy").onclick=()=>{const out=Object.entries(st).filter(([k,v])=>v.pick||(v.no||[]).length).map(([k,v])=>`${k}\t${v.pick||"-"}\t${(v.no||[]).join(",")}`).join("\n");
 navigator.clipboard.writeText(document.getElementById("tally").textContent+"\n"+out).then(()=>alert("Copied"))};
render();tally();
</script></body></html>'''
with open(os.path.join(RUNS, 'listen', 'transplant.html'), 'w', encoding='utf-8') as f:
    f.write(HTML.replace('__ROWS__', json.dumps(rows, ensure_ascii=False)))
print(len(rows), 'sentences ->', os.path.join(RUNS, 'listen', 'transplant.html'))
