import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
// SHIFU v2 EMBRYO ENGINE (embedded for standalone use)
// No dimensions. No channels. Structure emerges from exposure.
// ═���════════════��════════════════════════════════════════════════

const VERSION="2.0.0";
const OCR={"0,o":.1,"1,l":.2,"1,i":.2,"5,s":.3,"8,b":.3,"6,g":.4,"l,i":.2,"m,n":.4,"u,v":.5,"c,e":.5,"r,n":.3,"d,o":.3,"f,t":.4,"h,b":.4,"a,e":.4,"a,o":.4,"u,n":.4,"e,i":.4,"f,l":.4,"s,e":.5,"b,d":.4};
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
function editDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return editDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++)c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:1)));p=c;}return p[b.length];}
function ocrDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return ocrDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++){const k=[a[i],b[j]].sort().join(",");c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:(OCR[k]??1))));}p=c;}return p[b.length];}
function sharedBigrams(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<2||b.length<2)return 0;const bg=s=>{const m=new Map();for(let i=0;i<s.length-1;i++){const k=s.slice(i,i+2);m.set(k,(m.get(k)||0)+1);}return m;};const ba=bg(a),bb=bg(b);let inter=0;for(const[k,v]of ba)inter+=Math.min(v,bb.get(k)||0);const tot=[...ba.values()].reduce((a,b)=>a+b,0)+[...bb.values()].reduce((a,b)=>a+b,0);return tot?2*inter/tot:0;}
function tokenize(raw){return(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>1);}

class ShifuEmbryo{
constructor(){this.nodes={};this.sentenceCount=0;this.tokenCount=0;}
_nn(w){return{chars:w,freq:0,firstSeen:null,lastSeen:null,positions:[],gaps:[],sentLengths:[],neighbors:{},next:{},prev:{},next2:{}};}
feed(raw){const ws=tokenize(raw);if(ws.length<2)return 0;this.sentenceCount++;const len=ws.length;for(let i=0;i<ws.length;i++){const w=ws[i];const nd=this.nodes[w]??=this._nn(w);const rp=i/Math.max(ws.length-1,1);this.tokenCount++;nd.freq++;nd.firstSeen??=this.sentenceCount;if(nd.lastSeen!==null)nd.gaps.push(this.sentenceCount-nd.lastSeen);nd.lastSeen=this.sentenceCount;nd.positions.push(rp);nd.sentLengths.push(len);for(let j=Math.max(0,i-3);j<Math.min(ws.length,i+4);j++){if(j!==i)nd.neighbors[ws[j]]=(nd.neighbors[ws[j]]||0)+1;}if(i<ws.length-1)nd.next[ws[i+1]]=(nd.next[ws[i+1]]||0)+1;if(i>0)nd.prev[ws[i-1]]=(nd.prev[ws[i-1]]||0)+1;if(i<ws.length-2){const b=ws[i+1],c=ws[i+2];nd.next2[b]??={};nd.next2[b][c]=(nd.next2[b][c]||0)+1;}if(nd.positions.length>200)nd.positions=nd.positions.slice(-200);if(nd.gaps.length>100)nd.gaps=nd.gaps.slice(-100);if(nd.sentLengths.length>100)nd.sentLengths=nd.sentLengths.slice(-100);}return ws.length;}
feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0;for(const x of s)tk+=this.feed(x);return{sentences:s.length,tokens:tk};}
depth(w){const n=this.nodes[w?.toLowerCase?.()];if(!n)return{level:"unborn",evidence:0};if(n.freq===1)return{level:"surface",evidence:.1};if(n.freq<5)return{level:"shallow",evidence:.2};const nb=Object.keys(n.neighbors).length,sq=Object.keys(n.next).length+Object.keys(n.prev).length;const ev=Math.min((n.freq/50)*.3+(nb/20)*.3+(sq/10)*.2+(n.positions.length/50)*.2,1);if(ev<.3)return{level:"forming",evidence:ev};if(ev<.7)return{level:"structured",evidence:ev};return{level:"deep",evidence:ev};}
compare(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl],sig={},wt={};sig.editSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);sig.bigramSim=sharedBigrams(al,bl);sig.ocrSim=1-ocrDistance(al,bl)/Math.max(al.length,bl.length,1);wt.char=.05;if(!na||!nb)return{similarity:sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4,signals:sig,weights:wt,totalWeight:.05,depth:"surface"};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),union=new Set([...nbA,...nbBs]);if(union.size>0){const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...union].reduce((s,w)=>s+rw(w),0);sig.neighborOverlap=uw>0?sw/uw:0;wt.neighbor=Math.min(union.size/10,.35);}const totA=Object.values(na.next).reduce((s,v)=>s+v,0),totB=Object.values(nb.next).reduce((s,v)=>s+v,0);if(totA>0||totB>0){sig.expectsAB=totA?(na.next[bl]||0)/totA:0;sig.expectsBA=totB?(nb.next[al]||0)/totB:0;sig.directional=Math.abs(sig.expectsAB-sig.expectsBA);wt.seq=Math.min((totA+totB)/20,.25);}const nx2AB=na.next2?.[bl]?Object.keys(na.next2[bl]).length:0,nx2BA=nb.next2?.[al]?Object.keys(nb.next2[al]).length:0;if(nx2AB>0||nx2BA>0){sig.trajectoryAB=Math.min(nx2AB/5,1);sig.trajectoryBA=Math.min(nx2BA/5,1);wt.traj=.15;}if(na.positions.length>=3&&nb.positions.length>=3){sig.posSim=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*2,1);wt.pos=Math.min(Math.min(na.positions.length,nb.positions.length)/20,.10);}let indAB=0;for(const mid of Object.keys(na.next))if(this.nodes[mid]?.next[bl])indAB++;if(indAB>0){sig.indirectAB=Math.min(indAB/5,1);wt.indirect=Math.min(indAB/10,.15);}let sim=0,tw=0;if(wt.char){sim+=(sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4)*wt.char;tw+=wt.char;}if(wt.neighbor){sim+=sig.neighborOverlap*wt.neighbor;tw+=wt.neighbor;}if(wt.seq){sim+=Math.max(sig.expectsAB||0,sig.expectsBA||0)*wt.seq;tw+=wt.seq;}if(wt.traj){sim+=Math.max(sig.trajectoryAB||0,sig.trajectoryBA||0)*wt.traj;tw+=wt.traj;}if(wt.pos){sim+=sig.posSim*wt.pos;tw+=wt.pos;}if(wt.indirect){sim+=sig.indirectAB*wt.indirect;tw+=wt.indirect;}if(tw>0)sim/=tw;const d=tw>.5?"deep":tw>.3?"structured":tw>.15?"forming":"shallow";return{similarity:sim,signals:sig,weights:wt,totalWeight:tw,depth:d};}
affinity(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{a:al,b:bl,mutual:0,known:false};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1),sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...new Set([...nbA,...nbBs])].reduce((s,w)=>s+rw(w),0),orbit=uw>0?sw/uw:0;const totA=Object.values(na.next||{}).reduce((s,v)=>s+v,0),totB=Object.values(nb.next||{}).reduce((s,v)=>s+v,0),pullAB=totA?(na.next[bl]||0)/totA:0,pullBA=totB?(nb.next[al]||0)/totB:0;let indAB=0,indBA=0;for(const mid of Object.keys(na.next||{}))if(this.nodes[mid]?.next[bl])indAB++;for(const mid of Object.keys(nb.next||{}))if(this.nodes[mid]?.next[al])indBA++;indAB=Math.min(indAB/5,1);indBA=Math.min(indBA/5,1);const charSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);const fwA=new Set(Object.keys(na.next||{})),fwB=new Set(Object.keys(nb.next||{})),fwU=new Set([...fwA,...fwB]),expOvlp=fwU.size?[...fwA].filter(x=>fwB.has(x)).length/fwU.size:0;let posAlign=0;if(na.positions.length>=3&&nb.positions.length>=3)posAlign=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*3,1);const afAB=charSim*.05+orbit*.35+pullAB*.20+indAB*.20+expOvlp*.15+posAlign*.05,afBA=charSim*.05+orbit*.35+pullBA*.20+indBA*.20+expOvlp*.15+posAlign*.05;return{a:al,b:bl,orbit,pullAB,pullBA,indAB,indBA,charSim,expOvlp,posAlign,afAB,afBA,mutual:(afAB+afBA)/2,asym:Math.abs(afAB-afBA),known:true};}
scoreSentence(raw){const ws=tokenize(raw);if(ws.length<2)return{words:ws,steps:[],meanSurprise:0,coherence:0};const steps=[];let total=0;const field=new Map();for(let i=0;i<ws.length;i++){const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node};let sig=0,wts=0;if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*.35;wts+=.35;}}if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*.30;wts+=.30;}}if(i>0&&field.size>0){const fw=field.get(w)||0,mx=Math.max(...field.values(),1);step.fieldS=1-fw/mx;sig+=step.fieldS*.35;wts+=.35;}step.afGate=0;if(i>0&&node){const pn=this.nodes[ws[i-1]];if(pn?.neighbors&&node.neighbors){const pN=Object.keys(pn.neighbors),wN=new Set(Object.keys(node.neighbors)),sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;step.afGate=un?sh/un:0;}}step.surprise=wts>0?(sig/wts)*(1-step.afGate*.3):(node?.5:1);total+=step.surprise;step.cumS=total;steps.push(step);if(node?.neighbors)for(const[nb,cnt]of Object.entries(node.neighbors))field.set(nb,(field.get(nb)||0)+cnt);field.set(w,(field.get(w)||0)+10);}const ms=steps.length?total/steps.length:0;return{words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)};}
pressure(){const map=[];for(const[word,node]of Object.entries(this.nodes)){let inbound=0;for(const[,other]of Object.entries(this.nodes))if(other.next[word])inbound+=other.next[word];const actual=Object.keys(node.neighbors).length+Object.keys(node.next).length+Object.keys(node.prev).length;const p=actual-inbound;const nbrs=Object.keys(node.neighbors);let internal=0,pairs=0;for(let i=0;i<Math.min(nbrs.length,15);i++)for(let j=i+1;j<Math.min(nbrs.length,15);j++){pairs++;if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]])internal++;}const closure=pairs>0?internal/pairs:1;map.push({word,pressure:p,inbound,actual,closure,freq:node.freq,depth:this.depth(word).level});}return map.sort((a,b)=>a.pressure-b.pressure);}
vacuums(k=10){return this.pressure().filter(p=>p.pressure<0).slice(0,k);}
surpluses(k=10){return this.pressure().filter(p=>p.pressure>0).sort((a,b)=>b.pressure-a.pressure).slice(0,k);}
bridges(k=10){return this.pressure().filter(p=>p.closure<.3&&p.freq>=3).slice(0,k);}
unlearn(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{changed:false};let r=0;for(const[s,t]of[[na,bl],[nb,al]])for(const k of["neighbors","next","prev"])if(s[k][t]){s[k][t]=Math.floor(s[k][t]/2);if(s[k][t]<=0)delete s[k][t];r++;}if(na.next2[bl])delete na.next2[bl];if(nb.next2[al])delete nb.next2[al];return{changed:r>0,edgesWeakened:r};}
forget(word){const w=word.toLowerCase();if(!this.nodes[w])return{changed:false};for(const[,o]of Object.entries(this.nodes)){delete o.neighbors[w];delete o.next[w];delete o.prev[w];delete o.next2[w];for(const m of Object.keys(o.next2||{}))if(o.next2[m][w]){delete o.next2[m][w];if(!Object.keys(o.next2[m]).length)delete o.next2[m];}}delete this.nodes[w];return{changed:true,word:w};}
decay(threshold=1){let r=0;for(const[,n]of Object.entries(this.nodes)){for(const k of["neighbors","next","prev"])for(const[e,c]of Object.entries(n[k]))if(c<=threshold){delete n[k][e];r++;}for(const m of Object.keys(n.next2||{})){for(const[d,c]of Object.entries(n.next2[m]))if(c<=threshold){delete n.next2[m][d];r++;}if(!Object.keys(n.next2[m]).length)delete n.next2[m];}}return{edgesRemoved:r};}
correct(garbled,k=5){const g=garbled.toLowerCase(),cands=new Set();for(const w of Object.keys(this.nodes))if(Math.abs(w.length-g.length)<=3)cands.add(w);for(let i=0;i<g.length-1;i++){const bg=g.slice(i,i+2);for(const w of Object.keys(this.nodes))if(w.includes(bg))cands.add(w);}const scored=[...cands].map(w=>{const o=1-ocrDistance(g,w)/Math.max(g.length,w.length,1),b=sharedBigrams(g,w);return{word:w,score:o*.7+b*.3};});scored.sort((a,b)=>b.score-a.score||(this.nodes[b.word]?.freq||0)-(this.nodes[a.word]?.freq||0));const top=scored.slice(0,k),conf=top.length>=2?top[0].score-top[1].score:top.length?1:0;return{candidates:top,confidence:conf};}
similar(w,k=8){const wl=w.toLowerCase(),out=[];for(const c of Object.keys(this.nodes)){if(c===wl)continue;out.push({word:c,...this.compare(wl,c)});}return out.sort((a,b)=>b.similarity-a.similarity).slice(0,k);}
stats(){const v=Object.keys(this.nodes).length,depths={unborn:0,surface:0,shallow:0,forming:0,structured:0,deep:0};for(const w of Object.keys(this.nodes))depths[this.depth(w).level]++;return{version:VERSION,vocab:v,sentences:this.sentenceCount,tokens:this.tokenCount,depths};}
serialize(){return JSON.stringify({version:VERSION,nodes:Object.fromEntries(Object.entries(this.nodes).map(([w,n])=>[w,{chars:n.chars,freq:n.freq,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions.slice(-200),gaps:n.gaps.slice(-100),sentLengths:n.sentLengths.slice(-100),neighbors:n.neighbors,next:n.next,prev:n.prev,next2:n.next2}])),sentenceCount:this.sentenceCount,tokenCount:this.tokenCount});}
static deserialize(json){const d=JSON.parse(json),e=new ShifuEmbryo();e.sentenceCount=d.sentenceCount||0;e.tokenCount=d.tokenCount||0;for(const[w,n]of Object.entries(d.nodes||{})){e.nodes[w]={chars:n.chars||w,freq:n.freq||0,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions||[],gaps:n.gaps||[],sentLengths:n.sentLengths||[],neighbors:n.neighbors||{},next:n.next||{},prev:n.prev||{},next2:n.next2||{}};}return e;}}

// ═══════════════════════════════════════════════════════════════
// IndexedDB persistence
// ═══════════════════════════════════════════════════════════════

const IDB={
  open(){return new Promise((res,rej)=>{const r=indexedDB.open("shifu-v2",1);r.onupgradeneeded=()=>r.result.createObjectStore("state");r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error);});},
  async get(key){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readonly");const r=tx.objectStore("state").get(key);r.onsuccess=()=>res(r.result||null);r.onerror=()=>rej(r.error);});},
  async set(key,val){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readwrite");tx.objectStore("state").put(val,key);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error);});},
  async del(key){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readwrite");tx.objectStore("state").delete(key);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error);});}
};

// ═══════════════════════════════════════════════════════════════
// Visual components
// ═════════════════���═════════════════════════════════════════════

const DEPTH_COLORS = {unborn:"#555",surface:"#666",shallow:"#4a7eb5",forming:"#c9a227",structured:"#48b89a",deep:"#2ecc71"};
const Bar = ({value,max=1,color="#48b89a",label=""}) => (
  <div style={{display:"flex",alignItems:"center",gap:6,margin:"2px 0"}}>
    {label&&<span style={{width:90,fontSize:11,color:"#7a8a9a",textAlign:"right"}}>{label}</span>}
    <div style={{flex:1,height:14,background:"#1a2130",borderRadius:3,overflow:"hidden"}}>
      <div style={{width:`${Math.min(Math.max(value/max,0),1)*100}%`,height:"100%",background:color,borderRadius:3,transition:"width 0.3s"}}/>
    </div>
    <span style={{width:40,fontSize:11,color:"#7a8a9a",textAlign:"right"}}>{(value*100).toFixed(0)}%</span>
  </div>
);
const Badge = ({level}) => <span style={{display:"inline-block",padding:"1px 8px",borderRadius:9,fontSize:11,fontWeight:600,background:DEPTH_COLORS[level]||"#555",color:level==="forming"?"#1a2130":"#fff",marginLeft:4}}>{level}</span>;

// ═══════════════════════════════════════════════════════════════
// Intent parser
// ═══════════════════════════════════════════════════════════════

function parseIntent(input) {
  const t = input.trim();
  const tl = t.toLowerCase();
  if (tl==="help") return {cmd:"help"};
  if (tl==="stats") return {cmd:"stats"};
  if (tl==="reset") return {cmd:"reset"};
  if (tl==="pressure"||tl==="vacuums"||tl==="surpluses") return {cmd:tl};
  if (tl==="bridges") return {cmd:"bridges"};
  if (/^decay\s*(\d*)$/.test(tl)){const m=tl.match(/^decay\s*(\d*)$/);return{cmd:"decay",threshold:m[1]?parseInt(m[1]):1};}
  if (/^(feed:|learn:)\s*/i.test(t)) return {cmd:"feed",text:t.replace(/^(feed:|learn:)\s*/i,"")};
  if (/^(explore|word)\s+/i.test(tl)){const w=t.replace(/^(explore|word)\s+/i,"").trim();return{cmd:"explore",word:w};}
  if (/^(compare|cmp)\s+(\S+)\s+(and|vs|with)\s+(\S+)/i.test(tl)){const m=tl.match(/^(compare|cmp)\s+(\S+)\s+(?:and|vs|with)\s+(\S+)/);return{cmd:"compare",a:m[2],b:m[3]};}
  if (/^affinity\s+(\S+)\s+(and|vs|with)\s+(\S+)/i.test(tl)){const m=tl.match(/^affinity\s+(\S+)\s+(?:and|vs|with)\s+(\S+)/);return{cmd:"affinity",a:m[2],b:m[3]};}
  if (/^(correct|fix)\s+/i.test(tl)) return {cmd:"correct",text:t.replace(/^(correct|fix)\s+/i,"")};
  if (/^(score|coherence):\s*/i.test(t)) return {cmd:"score",text:t.replace(/^(score|coherence):\s*/i,"")};
  if (/^unlearn\s+(\S+)\s+(and|from)\s+(\S+)/i.test(tl)){const m=tl.match(/^unlearn\s+(\S+)\s+(?:and|from)\s+(\S+)/);return{cmd:"unlearn",a:m[1],b:m[2]};}
  if (/^forget\s+(\S+)/i.test(tl)){const m=tl.match(/^forget\s+(\S+)/);return{cmd:"forget",word:m[1]};}
  // Single word = explore
  if (/^\S+$/.test(t) && t.length>1) return {cmd:"explore",word:t};
  // Multi-word without prefix = feed
  if (t.length > 5) return {cmd:"feed",text:t};
  return {cmd:"unknown"};
}

// ═══════════════════════════════════════════════════════════════
// Response renderers
// ═══════════════���═══════════════════════════════════════════════

function renderExplore(eng, word) {
  const w = word.toLowerCase();
  const d = eng.depth(w);
  const n = eng.nodes[w];
  if (!n) return <div>"{word}" <Badge level="unborn"/> -- never seen. Feed me text containing this word.</div>;
  const topNb = Object.entries(n.neighbors).sort((a,b)=>b[1]-a[1]).slice(0,10);
  const topNx = Object.entries(n.next).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const sim = eng.similar(w, 5);
  return (<div>
    <div style={{marginBottom:8}}><strong>{word}</strong> <Badge level={d.level}/> freq={n.freq} evidence={d.evidence.toFixed(2)}</div>
    <div style={{fontSize:12,color:"#7a8a9a",marginBottom:4}}>neighbors ({Object.keys(n.neighbors).length}): {topNb.map(([w,c])=>`${w}(${c})`).join(", ")}</div>
    <div style={{fontSize:12,color:"#7a8a9a",marginBottom:4}}>next: {topNx.map(([w,c])=>`${w}(${c})`).join(", ")||"none"}</div>
    {sim.length>0&&<div style={{fontSize:12,color:"#7a8a9a"}}>similar: {sim.map(s=>`${s.word}(${s.similarity.toFixed(3)})`).join(", ")}</div>}
  </div>);
}

function renderCompare(eng, a, b) {
  const c = eng.compare(a, b);
  const s = c.signals;
  return (<div>
    <div style={{marginBottom:6}}><strong>{a}</strong> vs <strong>{b}</strong> -- similarity={c.similarity.toFixed(4)} depth={c.depth}</div>
    {s.neighborOverlap!==undefined&&<Bar value={s.neighborOverlap} label="orbit"/>}
    {s.expectsAB!==undefined&&<Bar value={s.expectsAB} color="#e8b339" label={`${a}\u2192${b}`}/>}
    {s.expectsBA!==undefined&&<Bar value={s.expectsBA} color="#e8b339" label={`${b}\u2192${a}`}/>}
    {s.trajectoryAB!==undefined&&<Bar value={s.trajectoryAB} color="#9b59b6" label={`traj ${a}\u2192`}/>}
    {s.trajectoryBA!==undefined&&<Bar value={s.trajectoryBA} color="#9b59b6" label={`traj ${b}\u2192`}/>}
    {s.posSim!==undefined&&<Bar value={s.posSim} color="#3498db" label="position"/>}
    {s.indirectAB!==undefined&&<Bar value={s.indirectAB} color="#e67e22" label="indirect"/>}
    <Bar value={s.editSim} color="#555" label="edit"/>
    <Bar value={s.ocrSim} color="#555" label="ocr"/>
    {s.directional!==undefined&&s.directional>0&&<div style={{fontSize:11,color:"#e8b339",marginTop:4}}>directional asymmetry: {(s.directional*100).toFixed(1)}%</div>}
  </div>);
}

function renderAffinity(eng, a, b) {
  const af = eng.affinity(a, b);
  if (!af.known) return <div>"{a}" or "{b}" unknown -- feed more text.</div>;
  return (<div>
    <div style={{marginBottom:6}}><strong>{a}</strong> {"<->"} <strong>{b}</strong> mutual={af.mutual.toFixed(4)} {af.asym>0.01&&`asym=${af.asym.toFixed(3)}`}</div>
    <Bar value={af.orbit} label="orbit" color="#48b89a"/>
    <Bar value={af.pullAB} label={`pull ${a}\u2192`} color="#e8b339"/>
    <Bar value={af.pullBA} label={`pull ${b}\u2192`} color="#e8b339"/>
    <Bar value={af.indAB} label={`ind ${a}\u2192`} color="#e67e22"/>
    <Bar value={af.indBA} label={`ind ${b}\u2192`} color="#e67e22"/>
    <Bar value={af.expOvlp} label="exp overlap" color="#9b59b6"/>
    <Bar value={af.charSim} label="char" color="#555"/>
    <Bar value={af.posAlign} label="position" color="#3498db"/>
  </div>);
}

function renderScore(eng, text) {
  const sc = eng.scoreSentence(text);
  if (sc.steps.length===0) return <div>Too short to score.</div>;
  return (<div>
    <div style={{marginBottom:8}}>coherence: <strong style={{color:sc.coherence>.5?"#2ecc71":sc.coherence>.2?"#e8b339":"#e74c3c"}}>{sc.coherence.toFixed(3)}</strong> mean surprise: {sc.meanSurprise.toFixed(3)}</div>
    {sc.steps.map((st,i)=>{
      const c=st.surprise<.3?"#2ecc71":st.surprise<.7?"#e8b339":"#e74c3c";
      return <div key={i} style={{display:"flex",alignItems:"center",gap:6,margin:"2px 0"}}>
        <span style={{width:80,fontSize:12,color:st.known?"#a8b4c4":"#555",textAlign:"right"}}>{st.afGate>.1?"* ":""}{st.word}</span>
        <div style={{flex:1,height:12,background:"#1a2130",borderRadius:3,overflow:"hidden"}}>
          <div style={{width:`${st.surprise*100}%`,height:"100%",background:c,borderRadius:3}}/>
        </div>
        <span style={{width:35,fontSize:10,color:"#7a8a9a"}}>{st.surprise.toFixed(2)}</span>
      </div>;
    })}
    <div style={{fontSize:11,color:"#7a8a9a",marginTop:4}}>* = affinity-gated (reduced surprise)</div>
  </div>);
}

function renderPressure(eng, mode) {
  let items;
  if (mode==="vacuums") items=eng.vacuums(12);
  else if (mode==="surpluses") items=eng.surpluses(12);
  else if (mode==="bridges") items=eng.bridges(12);
  else items=[...eng.vacuums(6),...eng.surpluses(6)];
  if (!items.length) return <div>No {mode} yet -- feed more text.</div>;
  return (<div>
    <div style={{marginBottom:6}}><strong>{mode||"pressure"}</strong> ({items.length} words)</div>
    {items.map((p,i)=>{
      const c=p.pressure<0?"#e74c3c":"#2ecc71";
      const w=mode==="bridges"?(1-p.closure):Math.min(Math.abs(p.pressure)/50,1);
      return <div key={i} style={{display:"flex",alignItems:"center",gap:6,margin:"2px 0"}}>
        <span style={{width:100,fontSize:12,color:"#a8b4c4",textAlign:"right"}}>{p.word}</span>
        <div style={{flex:1,height:12,background:"#1a2130",borderRadius:3,overflow:"hidden"}}>
          <div style={{width:`${w*100}%`,height:"100%",background:c,borderRadius:3}}/>
        </div>
        <span style={{width:60,fontSize:10,color:"#7a8a9a"}}>{mode==="bridges"?`cl=${p.closure.toFixed(2)}`:`p=${p.pressure}`}</span>
        <Badge level={p.depth}/>
      </div>;
    })}
  </div>);
}

const HELP = `Commands:
  [text]                    feed text (learn it)
  [word]                    explore a word
  compare A and B           compare two words
  affinity A and B          pre-contact attraction
  score: [sentence]         walk with surprise
  correct [garbled]         OCR correction
  pressure / vacuums / surpluses / bridges
  unlearn A and B           weaken connection
  forget [word]             remove completely
  decay / decay [n]         prune weak edges
  stats                     vocabulary & depths
  reset                     clear engine
  help                      this message`;

// ═���═════════════════════════════════════════════════════════════
// Main App
// ═══════════════════════════════════════════════════════════════

export default function ShifuChat() {
  const [eng, setEng] = useState(null);
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [vocab, setVocab] = useState(0);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  // Load engine from IndexedDB on mount
  useEffect(() => {
    (async () => {
      const saved = await IDB.get("engine");
      const e = saved ? ShifuEmbryo.deserialize(saved) : new ShifuEmbryo();
      setEng(e);
      setVocab(Object.keys(e.nodes).length);
      setMsgs([{from:"bot",content:<div style={{color:"#7a8a9a"}}>Shifu v2 embryo loaded. {Object.keys(e.nodes).length} words. Type <strong>help</strong> for commands.</div>}]);
    })();
  }, []);

  // Auto-save to IndexedDB after every command
  const save = useCallback(async (e) => {
    setVocab(Object.keys(e.nodes).length);
    await IDB.set("engine", e.serialize());
  }, []);

  // Scroll to bottom
  useEffect(() => { endRef.current?.scrollIntoView({behavior:"smooth"}); }, [msgs]);

  const send = useCallback(async () => {
    if (!input.trim() || !eng) return;
    const userMsg = input.trim();
    setInput("");
    setMsgs(m => [...m, {from:"user", content:userMsg}]);

    const intent = parseIntent(userMsg);
    let response;

    switch(intent.cmd) {
      case "help":
        response = <pre style={{margin:0,fontSize:12,whiteSpace:"pre-wrap"}}>{HELP}</pre>;
        break;
      case "stats": {
        const st = eng.stats();
        response = (<div>
          <div>v{st.version} -- {st.vocab} words, {st.tokens} tokens, {st.sentences} sentences</div>
          <div style={{fontSize:12,color:"#7a8a9a",marginTop:4}}>
            {Object.entries(st.depths).map(([l,c])=>c>0?<span key={l} style={{marginRight:8}}><Badge level={l}/> {c}</span>:null)}
          </div>
        </div>);
        break;
      }
      case "reset":
        const fresh = new ShifuEmbryo();
        setEng(fresh);
        await IDB.del("engine");
        setVocab(0);
        response = <div>Engine reset. Empty stone.</div>;
        break;
      case "feed": {
        const r = eng.feedText(intent.text);
        await save(eng);
        response = <div>Fed {r.sentences} sentence{r.sentences!==1?"s":""}, {r.tokens} tokens. Vocab: {Object.keys(eng.nodes).length}</div>;
        break;
      }
      case "explore":
        response = renderExplore(eng, intent.word);
        break;
      case "compare":
        response = renderCompare(eng, intent.a, intent.b);
        break;
      case "affinity":
        response = renderAffinity(eng, intent.a, intent.b);
        break;
      case "score":
        response = renderScore(eng, intent.text);
        break;
      case "correct": {
        const words = intent.text.split(/\s+/);
        response = (<div>{words.map((w,i)=>{
          const c=eng.correct(w);
          const top=c.candidates[0];
          if(!top||top.word===w.toLowerCase())return <span key={i} style={{marginRight:6}}>{w}</span>;
          return <span key={i} style={{marginRight:6}}><s style={{color:"#e74c3c"}}>{w}</s> <strong style={{color:"#2ecc71"}}>{top.word}</strong><span style={{fontSize:10,color:"#7a8a9a"}}>({top.score.toFixed(2)})</span> </span>;
        })}</div>);
        break;
      }
      case "pressure": case "vacuums": case "surpluses": case "bridges":
        response = renderPressure(eng, intent.cmd);
        break;
      case "unlearn": {
        const r = eng.unlearn(intent.a, intent.b);
        await save(eng);
        response = r.changed ? <div>Weakened {r.edgesWeakened} edges between "{intent.a}" and "{intent.b}"</div> : <div>No edges to weaken.</div>;
        break;
      }
      case "forget": {
        const r = eng.forget(intent.word);
        await save(eng);
        response = r.changed ? <div>Forgot "{intent.word}" completely.</div> : <div>"{intent.word}" was already unknown.</div>;
        break;
      }
      case "decay": {
        const r = eng.decay(intent.threshold||1);
        await save(eng);
        response = <div>Pruned {r.edgesRemoved} weak edges (threshold={intent.threshold||1}).</div>;
        break;
      }
      default:
        response = <div style={{color:"#7a8a9a"}}>Not sure what to do. Type <strong>help</strong> for commands.</div>;
    }

    setMsgs(m => [...m, {from:"bot", content:response}]);
  }, [input, eng, save]);

  if (!eng) return <div style={{color:"#a8b4c4",padding:40}}>Loading...</div>;

  return (
    <div style={{display:"flex",flexDirection:"column",height:"100vh",background:"#0a0e14",color:"#a8b4c4",fontFamily:"'IBM Plex Mono',monospace"}}>
      {/* Header */}
      <div style={{padding:"12px 20px",borderBottom:"1px solid #1a2130",display:"flex",alignItems:"baseline",gap:12}}>
        <span style={{fontFamily:"'Instrument Serif',Georgia,serif",fontSize:22,color:"#e0e6ed"}}>Shifu v2</span>
        <span style={{fontSize:12,color:"#48b89a"}}>embryo -- pressure-driven</span>
        <span style={{fontSize:12,color:"#555",marginLeft:"auto"}}>{vocab} words</span>
      </div>

      {/* Messages */}
      <div style={{flex:1,overflowY:"auto",padding:"12px 20px"}}>
        {msgs.map((m,i) => (
          <div key={i} style={{display:"flex",justifyContent:m.from==="user"?"flex-end":"flex-start",marginBottom:8}}>
            <div style={{
              maxWidth:"85%",padding:"8px 14px",borderRadius:12,fontSize:13,lineHeight:1.5,
              background:m.from==="user"?"#1a2a3a":"#0e1219",
              border:`1px solid ${m.from==="user"?"#1a3a5a":"#1a2130"}`,
            }}>
              {typeof m.content==="string"?m.content:m.content}
            </div>
          </div>
        ))}
        <div ref={endRef}/>
      </div>

      {/* Input */}
      <div style={{padding:"12px 20px",borderTop:"1px solid #1a2130",display:"flex",gap:8}}>
        <input
          ref={inputRef}
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>{if(e.key==="Enter")send();}}
          placeholder="Type a word, paste text, or ask a question..."
          style={{flex:1,padding:"8px 14px",background:"#0e1219",border:"1px solid #1a2130",borderRadius:8,color:"#e0e6ed",fontSize:13,fontFamily:"inherit",outline:"none"}}
          autoFocus
        />
        <button onClick={send} style={{padding:"8px 18px",background:"#48b89a",color:"#0a0e14",border:"none",borderRadius:8,fontWeight:600,fontSize:13,cursor:"pointer"}}>Send</button>
      </div>
    </div>
  );
}
