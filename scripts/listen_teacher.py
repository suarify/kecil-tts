"""Build runs/listen/teacher.html (Gate A) from the teacher-search kernel output in runs/k04.

Per sentence, blind and shuffled: each finalist, the final_a variant of the top finalist, stock Bella,
and the Chatterbox voice-converted clip (anchor) where one exists.

Serve:  python -m http.server 8765 --directory <repo>/runs   then open /listen/teacher.html
"""
import json
import os
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
K4 = os.path.join(RUNS, 'k04')
OUT = os.path.join(RUNS, 'listen')
os.makedirs(os.path.join(OUT, 'teacher'), exist_ok=True)

with zipfile.ZipFile(os.path.join(K4, 'clips_finalists.zip')) as z:
    z.extractall(os.path.join(OUT, 'teacher'))

fin = json.load(open(os.path.join(K4, 'finalists.json'), encoding='utf-8'))
df = pd.read_csv(os.path.join(K4, 'search_finalists.csv'))
ref = pd.read_csv(os.path.join(ROOT, 'eval', 'ref_32.tsv'), sep='\t', dtype=str)
t40 = pd.read_csv(os.path.join(ROOT, 'eval', 'test_40.tsv'), sep='\t', dtype=str)
text = {**dict(zip(ref.id, ref.norm)), **dict(zip(t40.id, t40.text))}
cat = {**dict(zip(ref.id, ref.category)), **dict(zip(t40.id, t40.category))}

names = [f['name'] for f in fin] + ['B0']
main = df[df.config.isin(names)]
stat = main.groupby('config').agg(cer=('cer', 'mean'), acc=('acc', 'mean'), sim=('sim', 'mean')).round(4)

# 10 general + 3 code-switch + 3 numbers
gen = [i for i in ref.id if cat[i] == 'general']
cs = [i for i in ref.id if cat[i] == 'codeswitch']
nums = sorted({i for i in main.id if i in cat and cat[i] == 'numbers'})
pick = lambda xs, k: [xs[int(i * (len(xs) - 1) / max(k - 1, 1))] for i in range(k)]  # noqa: E731
ids = pick(gen, 10) + pick(cs, 3) + nums[:3]

rows = []
for i in ids:
    clips = {}
    for n in names:
        p = os.path.join(OUT, 'teacher', n, f'{i}.wav')
        if os.path.exists(p):
            clips[n] = f'teacher/{n}/{i}.wav'
    anchor = os.path.join(RUNS, 'k03', 'pilot', 'T2v', f'{i}.wav')
    if os.path.exists(anchor):
        os.makedirs(os.path.join(OUT, 'teacher', 'CB'), exist_ok=True)
        import shutil
        shutil.copy(anchor, os.path.join(OUT, 'teacher', 'CB', f'{i}.wav'))
        clips['CB'] = f'teacher/CB/{i}.wav'
    rows.append(dict(id=i, cat=cat[i], text=text[i], clips=clips))

info = {f['name']: f for f in fin}
info['B0'] = dict(name='B0', blend='stock Bella', conv='s0_r_a')
info['CB'] = dict(name='CB', blend='Chatterbox, voice-converted (the Malay you liked)', conv='')

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Malay teacher pick</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6f5e;--bad:#b4442c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--line:#2c2c2a;--acc:#6fbfa6;--bad:#e07a5f}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px 16px;max-width:1150px}
h1{font-size:22px;margin:0 0 4px}p.m{color:var(--mut);margin:0 0 14px}
#bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;cursor:pointer}
.row{border-top:1px solid var(--line);padding:12px 0}.t{font-weight:600}.cat{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.05em}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin-top:6px}
.c{border:1px solid var(--line);border-radius:8px;padding:8px}.c.pick{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}.c.no{opacity:.55}
audio{width:100%;height:32px}.s{font-size:12px;color:var(--mut)}
#key{font-size:13px;color:var(--mut);margin:0 0 12px}
</style></head><body>
<h1>Which voice sounds most like Malay?</h1>
<p class="m">Each sentence has several unlabelled clips (shuffled per sentence). Pick the one that sounds most like natural Malay in a pleasant female voice. Tick "not acceptable" on any you would not use. Labels appear after you pick. Rate at least 12 sentences, then click Copy results and paste them to Claude.</p>
<div id="bar"><span id="tally"></span><button id="copy">Copy results</button></div>
<div id="key"></div><div id="list"></div>
<script>
const ROWS=__ROWS__, INFO=__INFO__, STAT=__STAT__;
let st={};try{st=JSON.parse(localStorage.getItem("teacher")||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem("teacher",JSON.stringify(st))}catch(e){}};
const seed=s=>{let h=0;for(const ch of s)h=(h*31+ch.charCodeAt(0))|0;return h};
const label=n=>{const i=INFO[n]||{};return n+" · "+(i.blend||"")+(i.conv?" · "+i.conv:"")};
function tally(){const t={},no={};let n=0;for(const v of Object.values(st)){if(v.pick){n++;t[v.pick]=(t[v.pick]||0)+1}for(const k of v.no||[])no[k]=(no[k]||0)+1}
 document.getElementById("tally").textContent=`Picked ${n}/${ROWS.length} · picks: `+(Object.entries(t).map(([k,v])=>k+" "+v).join(", ")||"none")+" · not acceptable: "+(Object.entries(no).map(([k,v])=>k+" "+v).join(", ")||"none")}
function render(){const L=document.getElementById("list");L.innerHTML="";
 for(const r of ROWS){const s=st[r.id]||(st[r.id]={no:[]});const keys=Object.keys(r.clips).sort((a,b)=>((seed(a+r.id))&255)-((seed(b+r.id))&255));
  const d=document.createElement("div");d.className="row";d.innerHTML=`<div class="cat">${r.cat} · ${r.id}</div><div class="t"></div><div class="g"></div>`;
  d.querySelector(".t").textContent=r.text;const g=d.querySelector(".g");
  keys.forEach((k,i)=>{const done=!!s.pick;const e=document.createElement("div");e.className="c"+(s.pick===k?" pick":"")+((s.no||[]).includes(k)?" no":"");
   e.innerHTML=`<label><input type="radio" name="${r.id}" ${s.pick===k?"checked":""}> <b>${"ABCDEFGH"[i]}</b>${done?" — "+label(k):""}</label>
   <audio controls preload="none" src="${r.clips[k]}"></audio>
   <label class="s"><input type="checkbox" ${(s.no||[]).includes(k)?"checked":""}> not acceptable</label>`;
   e.querySelector("input[type=radio]").onchange=()=>{s.pick=k;save();render();tally()};
   e.querySelector("input[type=checkbox]").onchange=ev=>{s.no=ev.target.checked?[...(s.no||[]),k]:(s.no||[]).filter(z=>z!=k);save();render();tally()};
   g.appendChild(e)});L.appendChild(d)}}
document.getElementById("key").textContent="Automatic scores (all sentences): "+Object.entries(STAT).map(([k,v])=>`${k}: CER ${(v.cer*100).toFixed(1)}%, accent ${v.acc}, sim ${v.sim}`).join(" | ");
document.getElementById("copy").onclick=()=>{const out=Object.entries(st).filter(([k,v])=>v.pick||(v.no||[]).length).map(([k,v])=>`${k}\t${v.pick||"-"}\t${(v.no||[]).join(",")}`).join("\n");
 navigator.clipboard.writeText(document.getElementById("tally").textContent+"\n"+out).then(()=>alert("Copied"))};
render();tally();
</script></body></html>'''
with open(os.path.join(OUT, 'teacher.html'), 'w', encoding='utf-8') as f:
    f.write(HTML.replace('__ROWS__', json.dumps(rows, ensure_ascii=False))
            .replace('__INFO__', json.dumps(info, ensure_ascii=False))
            .replace('__STAT__', json.dumps(stat.to_dict(orient='index'), ensure_ascii=False)))
print(len(rows), 'sentences ->', os.path.join(OUT, 'teacher.html'))
print(stat.to_string())
