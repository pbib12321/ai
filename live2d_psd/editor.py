"""Browser editor for fixing parts by hand, served from a small local web server.

    python -m live2d_psd edit character_cubism.l2d      (or the image / the .psd)

Tools: AI select (MobileSAM: click to add, right-click to subtract, Enter to
apply), colour wand, brush, eraser and picker, plus undo, zoom and pan.
Parts can be renamed, re-foldered, reordered and created.  "Export PSD"
rebuilds the PSD from your edits.
"""
from __future__ import annotations

import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from .project import load_project, save_project


class _State:
    def __init__(self, folder, out_psd):
        self.folder = folder
        self.out = out_psd
        self.proj, self.labels, self.alpha = load_project(folder)
        self.rgb = np.array(Image.open(self.proj["source"]).convert("RGB"))
        self.sam = None
        self.emb = None
        self.lock = threading.Lock()

    def sam_mask(self, pos, neg):
        from . import models

        with self.lock:
            if self.sam is None:
                self.sam = models.MobileSAM.load() or False
            if not self.sam:
                return None
            if self.emb is None:
                self.emb = self.sam.embed(self.rgb)
        pts = np.array([list(pos) + list(neg)], np.float32)
        lab = np.array([[1] * len(pos) + [0] * len(neg)], np.float32)
        low, iou = self.sam.predict(self.emb, pts, lab)
        k = int(np.argmax(iou[0]))
        return self.sam.upscale(self.emb, low[0, k]) > 0


def _png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def serve(folder, out_psd, port=8765, open_browser=True):
    st = _State(folder, out_psd)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json", code=200):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n)

        def do_GET(self):
            if self.path == "/":
                return self._send(_HTML, "text/html; charset=utf-8")
            if self.path == "/image":
                a = (st.alpha * 255).astype(np.uint8)
                return self._send(_png(np.dstack([st.rgb, np.maximum(a, 255 if st.proj["mode"] == "anime" else a)])), "image/png")
            if self.path == "/alpha":
                return self._send(_png((st.alpha * 255).astype(np.uint8)), "image/png")
            if self.path == "/labels":
                return self._send(st.labels.astype("<u2").tobytes(), "application/octet-stream")
            if self.path == "/info":
                h, w = st.labels.shape
                return self._send(json.dumps({"w": w, "h": h, "parts": st.proj["parts"], "mode": st.proj["mode"], "out": st.out}))
            self._send("not found", "text/plain", 404)

        def do_POST(self):
            if self.path == "/sam":
                d = json.loads(self._body())
                m = st.sam_mask(d.get("pos", []), d.get("neg", []))
                if m is None:
                    return self._send(json.dumps({"error": "MobileSAM model not found"}), code=400)
                return self._send(_png((m * 255).astype(np.uint8)), "image/png")
            if self.path == "/labels":
                raw = np.frombuffer(self._body(), "<u2").astype(np.int32)
                st.labels = raw.reshape(st.labels.shape)
                return self._send("{}")
            if self.path == "/parts":
                st.proj["parts"] = json.loads(self._body())
                save_project(st.folder, st.proj["source"], st.proj["mode"], st.proj["parts"], st.labels, st.alpha, st.proj.get("settings", {}))
                return self._send("{}")
            if self.path == "/export":
                from . import build_project

                lines = []
                try:
                    build_project(st.folder, st.out, log=lines.append)
                except Exception as e:  # report to the page instead of dying
                    lines.append(f"error: {e}")
                return self._send(json.dumps({"log": lines}))
            self._send("not found", "text/plain", 404)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print(f"editor running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Live2D PSD part editor</title>
<style>
:root{--bg:#1d1f24;--panel:#262931;--line:#3a3e48;--text:#e6e6ea;--dim:#9aa0ac;--acc:#5b9cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px system-ui,sans-serif;display:flex;height:100vh;overflow:hidden}
#stage{flex:1;position:relative;overflow:hidden;cursor:crosshair;background:repeating-conic-gradient(#2b2e36 0 25%,#24262d 0 50%) 0 0/24px 24px}
canvas{position:absolute;left:0;top:0;image-rendering:pixelated}
#side{width:330px;background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column}
.bar{padding:8px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px;align-items:center}
button{background:#323642;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer;font:inherit}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}button:hover{border-color:var(--acc)}
#parts{flex:1;overflow:auto}
.p{display:flex;gap:6px;align-items:center;padding:4px 8px;border-bottom:1px solid #2f323b;cursor:pointer}
.p.sel{background:#2f3b52}.sw{width:14px;height:14px;border-radius:3px;flex:none}
.p input{background:transparent;border:1px solid transparent;color:var(--text);font:inherit;width:100%;padding:2px}
.p input:focus{border-color:var(--line);background:#1d1f24}.p .path{color:var(--dim);width:90px}
label{color:var(--dim)}input[type=range]{width:90px}
#log{height:110px;overflow:auto;padding:6px 8px;font:11px ui-monospace,monospace;color:var(--dim);border-top:1px solid var(--line);white-space:pre-wrap}
#hint{position:absolute;left:8px;bottom:8px;background:#000a;padding:6px 8px;border-radius:6px;color:#ccc;pointer-events:none}
</style></head><body>
<div id="stage"><canvas id="cv"></canvas><div id="hint"></div></div>
<div id="side">
 <div class="bar">
  <button data-t="sam" class="on" title="A">AI select</button><button data-t="wand" title="W">Wand</button>
  <button data-t="brush" title="B">Brush</button><button data-t="erase" title="E">Eraser</button><button data-t="pick" title="I">Pick</button>
 </div>
 <div class="bar"><label>size</label><input id="size" type="range" min="1" max="120" value="18">
  <label>wand tol</label><input id="tol" type="range" min="2" max="80" value="18">
  <label><input id="ov" type="checkbox" checked> overlay</label><label><input id="solo" type="checkbox"> only selected</label></div>
 <div class="bar"><button id="apply">Apply AI (Enter)</button><button id="clear">Clear points (Esc)</button><button id="undo">Undo (Ctrl+Z)</button></div>
 <div class="bar"><button id="add">+ New part</button><button id="up">Move up</button><button id="down">Move down</button><button id="del">Delete part</button></div>
 <div id="parts"></div>
 <div class="bar"><button id="save">Save</button><button id="export" class="on">Export PSD</button></div>
 <div id="log"></div>
</div>
<script>
const $=s=>document.querySelector(s), cv=$('#cv'), ctx=cv.getContext('2d');
let W,H,parts,labels,img,tool='sam',sel=null,scale=1,ox=0,oy=0,undo=[],pos=[],neg=[],samMask=null,dirty=false;
const colors={};
function col(id){if(!colors[id]){const h=(id*137.508)%360;colors[id]=`hsl(${h} 75% 55%)`;}return colors[id]}
function rgbOf(id){const c=document.createElement('canvas').getContext('2d');c.fillStyle=col(id);c.fillRect(0,0,1,1);return c.getImageData(0,0,1,1).data}
function log(s){const l=$('#log');l.textContent+=s+'\n';l.scrollTop=1e9}
async function init(){
 const info=await (await fetch('/info')).json();W=info.w;H=info.h;parts=info.parts;
 labels=new Uint16Array(await (await fetch('/labels')).arrayBuffer());
 img=new Image();img.src='/image';await img.decode();
 cv.width=W;cv.height=H;fit();sel=parts.length?parts[0].id:null;renderParts();draw();
 log(`${W}x${H}, ${parts.length} parts, ${info.mode} mode. Export writes ${info.out}`);hint();
}
function fit(){const r=$('#stage').getBoundingClientRect();scale=Math.min(r.width/W,r.height/H)*0.95;ox=(r.width-W*scale)/2;oy=(r.height-H*scale)/2;place()}
function place(){cv.style.width=W*scale+'px';cv.style.height=H*scale+'px';cv.style.left=ox+'px';cv.style.top=oy+'px'}
let ovl=null;
function draw(){
 ctx.clearRect(0,0,W,H);ctx.drawImage(img,0,0);
 if($('#ov').checked){
  const id=ctx.getImageData(0,0,W,H),d=id.data,cache={},solo=$('#solo').checked;
  for(let i=0;i<labels.length;i++){const l=labels[i];if(!l)continue;if(solo&&l!==sel){d[i*4]*=.35;d[i*4+1]*=.35;d[i*4+2]*=.35;continue}
   const c=cache[l]||(cache[l]=rgbOf(l)),a=l===sel?.55:.35;d[i*4]=d[i*4]*(1-a)+c[0]*a;d[i*4+1]=d[i*4+1]*(1-a)+c[1]*a;d[i*4+2]=d[i*4+2]*(1-a)+c[2]*a}
  if(samMask){for(let i=0;i<samMask.length;i++)if(samMask[i]){d[i*4]=d[i*4]*.4+255*.6;d[i*4+1]=d[i*4+1]*.4+255*.6;d[i*4+2]*=.4}}
  ctx.putImageData(id,0,0);
 }
 for(const [x,y] of pos){ctx.fillStyle='#2f2';ctx.beginPath();ctx.arc(x,y,5/scale+2,0,7);ctx.fill()}
 for(const [x,y] of neg){ctx.fillStyle='#f33';ctx.beginPath();ctx.arc(x,y,5/scale+2,0,7);ctx.fill()}
}
function renderParts(){
 const box=$('#parts');box.innerHTML='';
 const sorted=[...parts].sort((a,b)=>b.z-a.z);
 for(const p of sorted){
  const row=document.createElement('div');row.className='p'+(p.id===sel?' sel':'');
  row.innerHTML=`<span class="sw" style="background:${col(p.id)}"></span><input class="name" value="${p.name}"><input class="path" value="${p.path.join('/')}" title="folder (a/b)">`;
  row.onclick=e=>{if(e.target.tagName!=='INPUT'||sel!==p.id){sel=p.id;renderParts();draw()}};
  row.querySelector('.name').onchange=e=>{p.name=e.target.value;dirty=true};
  row.querySelector('.path').onchange=e=>{p.path=e.target.value.split('/').map(s=>s.trim()).filter(Boolean);dirty=true};
  box.appendChild(row);
 }
}
function hint(){$('#hint').textContent={sam:'AI select: click = add, right-click = remove, Enter = give to selected part',wand:'Wand: click a colour area to give it to the selected part',brush:'Brush: paint the selected part',erase:'Eraser: remove pixels from all parts',pick:'Pick: click to select the part under the cursor'}[tool]+' · wheel = zoom, middle/space-drag = pan'}
function pushUndo(){undo.push(labels.slice());if(undo.length>25)undo.shift();dirty=true}
function toImg(e){const r=cv.getBoundingClientRect();return [Math.floor((e.clientX-r.left)/scale),Math.floor((e.clientY-r.top)/scale)]}
document.querySelectorAll('[data-t]').forEach(b=>b.onclick=()=>{tool=b.dataset.t;document.querySelectorAll('[data-t]').forEach(x=>x.classList.toggle('on',x===b));hint()});
let painting=false,panning=false,last=null,space=false;
cv.oncontextmenu=e=>e.preventDefault();
cv.onmousedown=e=>{
 if(e.button===1||space){panning=true;last=[e.clientX,e.clientY];return}
 const [x,y]=toImg(e);if(x<0||y<0||x>=W||y>=H)return;
 if(tool==='pick'){const l=labels[y*W+x];if(l){sel=l;renderParts();draw()}return}
 if(tool==='sam'){(e.button===2?neg:pos).push([x,y]);runSam();return}
 if(tool==='wand'){if(!sel)return;pushUndo();wand(x,y);draw();return}
 if(tool==='brush'||tool==='erase'){if(tool==='brush'&&!sel)return;pushUndo();painting=true;dab(x,y);draw()}
};
window.onmousemove=e=>{
 if(panning){ox+=e.clientX-last[0];oy+=e.clientY-last[1];last=[e.clientX,e.clientY];place();return}
 if(painting){const [x,y]=toImg(e);dab(x,y);draw()}
};
window.onmouseup=()=>{painting=false;panning=false};
$('#stage').onwheel=e=>{e.preventDefault();const r=$('#stage').getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.15:1/1.15;ox=mx-(mx-ox)*f;oy=my-(my-oy)*f;scale*=f;place()};
function dab(x,y){const r=+$('#size').value,v=tool==='erase'?0:sel;for(let j=-r;j<=r;j++)for(let i=-r;i<=r;i++){if(i*i+j*j>r*r)continue;const X=x+i,Y=y+j;if(X<0||Y<0||X>=W||Y>=H)continue;labels[Y*W+X]=v}}
function wand(x,y){
 const c=document.createElement('canvas');c.width=W;c.height=H;const g=c.getContext('2d');g.drawImage(img,0,0);const d=g.getImageData(0,0,W,H).data;
 const t=+$('#tol').value,i0=(y*W+x)*4,r0=d[i0],g0=d[i0+1],b0=d[i0+2],seen=new Uint8Array(W*H),st=[y*W+x];seen[y*W+x]=1;
 while(st.length){const i=st.pop();labels[i]=sel;const X=i%W,Y=(i/W)|0;for(const n of [i-1,i+1,i-W,i+W]){if(n<0||n>=W*H||seen[n])continue;if((n===i-1&&X===0)||(n===i+1&&X===W-1))continue;const k=n*4;if(d[k+3]<128)continue;if(Math.abs(d[k]-r0)+Math.abs(d[k+1]-g0)+Math.abs(d[k+2]-b0)>t*3)continue;seen[n]=1;st.push(n)}}
}
async function runSam(){
 log('AI select…');const r=await fetch('/sam',{method:'POST',body:JSON.stringify({pos,neg})});
 if(!r.ok){log((await r.json()).error);return}
 const im=new Image();im.src=URL.createObjectURL(await r.blob());await im.decode();
 const c=document.createElement('canvas');c.width=W;c.height=H;const g=c.getContext('2d');g.drawImage(im,0,0);const d=g.getImageData(0,0,W,H).data;
 samMask=new Uint8Array(W*H);for(let i=0;i<W*H;i++)samMask[i]=d[i*4]>127?1:0;draw();
}
function applySam(){if(!samMask||!sel)return;pushUndo();for(let i=0;i<samMask.length;i++)if(samMask[i])labels[i]=sel;samMask=null;pos=[];neg=[];draw()}
$('#apply').onclick=applySam;$('#clear').onclick=()=>{samMask=null;pos=[];neg=[];draw()};
$('#undo').onclick=()=>{if(undo.length){labels=undo.pop();draw()}};
$('#ov').onchange=draw;$('#solo').onchange=draw;
$('#add').onclick=()=>{const id=Math.max(0,...parts.map(p=>p.id))+1,s=parts.find(p=>p.id===sel);parts.push({id,name:'New Part '+id,path:s?[...s.path]:['Body'],z:s?s.z+0.01:50,sweep:true});sel=id;dirty=true;renderParts()};
$('#del').onclick=()=>{if(!sel||!confirm('Delete this part? Its pixels become unassigned.'))return;pushUndo();for(let i=0;i<labels.length;i++)if(labels[i]===sel)labels[i]=0;parts=parts.filter(p=>p.id!==sel);sel=parts[0]?.id;renderParts();draw()};
function move(dir){const s=[...parts].sort((a,b)=>b.z-a.z),i=s.findIndex(p=>p.id===sel),j=i+dir;if(i<0||j<0||j>=s.length)return;const t=s[i].z;s[i].z=s[j].z;s[j].z=t;if(s[i].z===s[j].z)s[i].z+=dir<0?0.01:-0.01;dirty=true;renderParts()}
$('#up').onclick=()=>move(-1);$('#down').onclick=()=>move(1);
async function save(){await fetch('/labels',{method:'POST',body:labels.buffer});await fetch('/parts',{method:'POST',body:JSON.stringify(parts)});dirty=false;log('saved')}
$('#save').onclick=save;
$('#export').onclick=async()=>{await save();log('building PSD…');const r=await (await fetch('/export',{method:'POST'})).json();r.log.forEach(log)};
window.onkeydown=e=>{if(e.target.tagName==='INPUT')return;
 if(e.key==='Enter')applySam();else if(e.key==='Escape')$('#clear').onclick();else if(e.key===' '){space=true;e.preventDefault()}
 else if((e.ctrlKey||e.metaKey)&&e.key==='z')$('#undo').onclick();else if(e.key==='0')fit();
 else{const t={a:'sam',w:'wand',b:'brush',e:'erase',i:'pick'}[e.key];if(t)document.querySelector(`[data-t=${t}]`).click()}};
window.onkeyup=e=>{if(e.key===' ')space=false};
window.onbeforeunload=e=>{if(dirty){e.preventDefault();return ''}};
window.onresize=fit;init();
</script></body></html>"""
