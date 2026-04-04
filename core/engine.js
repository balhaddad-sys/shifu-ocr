// SHIFU v2.1 — The Embryo (Brain-like consolidation)
// No arrays. Position is a running mean that hardens with exposure.
// Deep words resist change. New connections weaken with depth.
// First exposure = plastic (learns everything). 100th = rigid (barely moves).
const VERSION="2.1.0";
const OCR={"0,o":.1,"1,l":.2,"1,i":.2,"5,s":.3,"8,b":.3,"6,g":.4,"l,i":.2,"m,n":.4,"u,v":.5,"c,e":.5,"r,n":.3,"d,o":.3,"f,t":.4,"h,b":.4,"a,e":.4,"a,o":.4,"u,n":.4,"e,i":.4,"f,l":.4,"s,e":.5,"b,d":.4};
function editDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return editDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++)c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:1)));p=c;}return p[b.length];}
function ocrDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return ocrDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++){const k=[a[i],b[j]].sort().join(",");c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:(OCR[k]??1))));}p=c;}return p[b.length];}
function sharedBigrams(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<2||b.length<2)return 0;const bg=s=>{const m=new Map();for(let i=0;i<s.length-1;i++){const k=s.slice(i,i+2);m.set(k,(m.get(k)||0)+1);}return m;};const ba=bg(a),bb=bg(b);let inter=0;for(const[k,v]of ba)inter+=Math.min(v,bb.get(k)||0);const tot=[...ba.values()].reduce((a,b)=>a+b,0)+[...bb.values()].reduce((a,b)=>a+b,0);return tot?2*inter/tot:0;}
function tokenize(raw){return(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>1);}

class ShifuEmbryo{
constructor(){this.nodes={};this.sentenceCount=0;this.tokenCount=0;this._idxLen={};this._idxBg={};}

// ─── Node: no arrays. Running statistics that consolidate. ────────
// pos: settled mean position (0→1). Hardens with freq.
// posV: variance — how much pos still moves. Shrinks with exposure.
// lastGap: most recent gap between encounters.
// plasticity: 1/freq. First exposure = 1.0. 100th = 0.01.
_nn(w){return{chars:w,freq:0,firstSeen:null,lastSeen:null,pos:0.5,posV:1.0,lastGap:0,nbrCount:0,frozen:false,neighbors:{},next:{},prev:{},next2:{}};}

_idx(w){const l=w.length;(this._idxLen[l]??=[]).push(w);for(let i=0;i<w.length-1;i++){const bg=w.slice(i,i+2);(this._idxBg[bg]??=[]).push(w);}}

// ─── Feed: consolidation, not accumulation. ───────────────────────
feed(raw){
  const ws=tokenize(raw);if(ws.length<2)return 0;
  this.sentenceCount++;const len=ws.length;
  for(let i=0;i<ws.length;i++){
    const w=ws[i];
    const isNew=!(w in this.nodes);
    const nd=this.nodes[w]??=this._nn(w);
    if(isNew)this._idx(w);
    this.tokenCount++;
    nd.freq++;
    nd.firstSeen??=this.sentenceCount;
    if(nd.lastSeen!==null)nd.lastGap=this.sentenceCount-nd.lastSeen;
    nd.lastSeen=this.sentenceCount;
    const plasticity=1/nd.freq;
    const rp=i/Math.max(len-1,1);
    if(nd.freq===1){
      nd.pos=rp;nd.posV=0.25;
    }else{
      const delta=rp-nd.pos;
      nd.pos+=delta*plasticity;
      nd.posV+=(delta*delta-nd.posV)*plasticity;
    }
    // Window + frozen — all cheap checks first
    if(!nd.frozen){
      const win=nd.freq<=2?6:nd.freq<=8?4:nd.freq<=32?3:2;
      const cap=nd.freq/3;const hasStr=nd.nbrCount>=5;
      const prevW=i>0?ws[i-1]:null;const nextW=i<len-1?ws[i+1]:null;
      const jMin=i-win<0?0:i-win;const jMax=i+win+1>len?len:i+win+1;
      for(let j=jMin;j<jMax;j++){
        if(j===i)continue;const nb=ws[j];
        if(!nd.neighbors)nd.neighbors={};
        if(nd.neighbors[nb]!==undefined){
          if(nd.neighbors[nb]<cap)nd.neighbors[nb]++;
        }else{
          let add=nd.freq<=5?1:plasticity>0.5?plasticity*2:0.5;
          if(hasStr){const nbNb=this.nodes[nb]?.neighbors;
            if(nbNb&&!((prevW&&nbNb[prevW])||(nextW&&nbNb[nextW])))add=plasticity*0.5;}
          nd.neighbors[nb]=add;nd.nbrCount++;
        }
      }
      if(nd.freq>30&&nd.posV<0.05&&nd.nbrCount>20)nd.frozen=true;
    }
    // Guard: ensure properties exist (V8 can lose them under memory pressure at 100K+ nodes)
    if(!nd.next)nd.next={};if(!nd.prev)nd.prev={};if(!nd.next2)nd.next2={};
    if(i<len-1)nd.next[ws[i+1]]=(nd.next[ws[i+1]]||0)+1;
    if(i>0)nd.prev[ws[i-1]]=(nd.prev[ws[i-1]]||0)+1;
    if(i<len-2){const b=ws[i+1],c=ws[i+2];nd.next2[b]??={};nd.next2[b][c]=(nd.next2[b][c]||0)+1;}
  }
  return ws.length;
}

feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0;for(const x of s)tk+=this.feed(x);return{sentences:s.length,tokens:tk};}

depth(w){
  const n=this.nodes[w];if(!n)return{level:"unborn",evidence:0};
  if(n.freq===1)return{level:"surface",evidence:0.1};
  if(n.freq<5)return{level:"shallow",evidence:0.2};
  const nb=n.nbrCount;
  const sq=Object.keys(n.next).length+Object.keys(n.prev).length;
  const stability=1-Math.min(n.posV*4,1);
  const ev=Math.min((n.freq/50)*0.25+(nb/20)*0.25+(sq/10)*0.2+stability*0.3,1.0);
  if(ev<0.3)return{level:"forming",evidence:ev};
  if(ev<0.7)return{level:"structured",evidence:ev};
  return{level:"deep",evidence:ev};
}

compare(a,b){
  const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl],sig={},wt={};
  sig.editSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);
  sig.bigramSim=sharedBigrams(al,bl);
  sig.ocrSim=1-ocrDistance(al,bl)/Math.max(al.length,bl.length,1);
  wt.char=0.05;
  if(!na||!nb)return{similarity:sig.editSim*0.3+sig.bigramSim*0.3+sig.ocrSim*0.4,signals:sig,weights:wt,totalWeight:0.05,depth:"surface"};
  const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),union=new Set([...nbA,...nbBs]);
  if(union.size>0){const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...union].reduce((s,w)=>s+rw(w),0);sig.neighborOverlap=uw>0?sw/uw:0;wt.neighbor=Math.min(union.size/10,0.35);}
  const totA=Object.values(na.next).reduce((s,v)=>s+v,0),totB=Object.values(nb.next).reduce((s,v)=>s+v,0);
  if(totA>0||totB>0){sig.expectsAB=totA?(na.next[bl]||0)/totA:0;sig.expectsBA=totB?(nb.next[al]||0)/totB:0;sig.directional=Math.abs(sig.expectsAB-sig.expectsBA);wt.seq=Math.min((totA+totB)/20,0.25);}
  const nx2AB=na.next2?.[bl]?Object.keys(na.next2[bl]).length:0,nx2BA=nb.next2?.[al]?Object.keys(nb.next2[al]).length:0;
  if(nx2AB>0||nx2BA>0){sig.trajectoryAB=Math.min(nx2AB/5,1);sig.trajectoryBA=Math.min(nx2BA/5,1);wt.traj=0.15;}
  const bothSettled=na.freq>=3&&nb.freq>=3;
  if(bothSettled){sig.posSim=1-Math.min(Math.abs(na.pos-nb.pos)*2,1);const settledWeight=Math.min((1-na.posV)*(1-nb.posV)*4,1);wt.pos=settledWeight*0.15;}
  let indAB=0;for(const mid of Object.keys(na.next))if(this.nodes[mid]?.next[bl])indAB++;
  if(indAB>0){sig.indirectAB=Math.min(indAB/5,1);wt.indirect=Math.min(indAB/10,0.15);}
  let sim=0,tw=0;
  if(wt.char){sim+=(sig.editSim*0.3+sig.bigramSim*0.3+sig.ocrSim*0.4)*wt.char;tw+=wt.char;}
  if(wt.neighbor){sim+=sig.neighborOverlap*wt.neighbor;tw+=wt.neighbor;}
  if(wt.seq){sim+=Math.max(sig.expectsAB||0,sig.expectsBA||0)*wt.seq;tw+=wt.seq;}
  if(wt.traj){sim+=Math.max(sig.trajectoryAB||0,sig.trajectoryBA||0)*wt.traj;tw+=wt.traj;}
  if(wt.pos){sim+=sig.posSim*wt.pos;tw+=wt.pos;}
  if(wt.indirect){sim+=sig.indirectAB*wt.indirect;tw+=wt.indirect;}
  if(tw>0)sim/=tw;
  const d=tw>0.5?"deep":tw>0.3?"structured":tw>0.15?"forming":"shallow";
  return{similarity:sim,signals:sig,weights:wt,totalWeight:tw,depth:d};
}

affinity(a,b){
  const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];
  if(!na||!nb)return{a:al,b:bl,mutual:0,known:false};
  const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x));
  const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);
  const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...new Set([...nbA,...nbBs])].reduce((s,w)=>s+rw(w),0);
  const orbit=uw>0?sw/uw:0;
  const totA=Object.values(na.next||{}).reduce((s,v)=>s+v,0),totB=Object.values(nb.next||{}).reduce((s,v)=>s+v,0);
  const pullAB=totA?(na.next[bl]||0)/totA:0,pullBA=totB?(nb.next[al]||0)/totB:0;
  let indAB=0,indBA=0;
  for(const mid of Object.keys(na.next||{}))if(this.nodes[mid]?.next[bl])indAB++;
  for(const mid of Object.keys(nb.next||{}))if(this.nodes[mid]?.next[al])indBA++;
  indAB=Math.min(indAB/5,1);indBA=Math.min(indBA/5,1);
  const charSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);
  const fwA=new Set(Object.keys(na.next||{})),fwB=new Set(Object.keys(nb.next||{})),fwU=new Set([...fwA,...fwB]);
  const expOvlp=fwU.size?[...fwA].filter(x=>fwB.has(x)).length/fwU.size:0;
  let posAlign=0;
  if(na.freq>=3&&nb.freq>=3)posAlign=1-Math.min(Math.abs(na.pos-nb.pos)*3,1);
  const afAB=charSim*0.05+orbit*0.35+pullAB*0.20+indAB*0.20+expOvlp*0.15+posAlign*0.05;
  const afBA=charSim*0.05+orbit*0.35+pullBA*0.20+indBA*0.20+expOvlp*0.15+posAlign*0.05;
  return{a:al,b:bl,orbit,pullAB,pullBA,indAB,indBA,charSim,expOvlp,posAlign,afAB,afBA,mutual:(afAB+afBA)/2,asym:Math.abs(afAB-afBA),known:true};
}

// ─── Score sentence — depth-gated ─────────────────────────────────
scoreSentence(raw){
  const ws=tokenize(raw);if(ws.length<2)return{words:ws,steps:[],meanSurprise:0,coherence:0};
  const steps=[];let total=0;const field=new Map();let fieldMax=1;
  for(let i=0;i<ws.length;i++){
    const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node};let sig=0,wts=0;
    const wDeep=node&&(node.freq>=5&&node.nbrCount>=5);
    if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*0.35;wts+=0.35;}}
    if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*0.30;wts+=0.30;}}
    if(i>0&&field.size>0){const fw=field.get(w)||0;step.fieldS=1-fw/fieldMax;sig+=step.fieldS*0.35;wts+=0.35;}
    step.afGate=0;
    if(i>0&&node){
      const prevNode=this.nodes[ws[i-1]];
      const prevDeep=prevNode&&(prevNode.freq>=5&&prevNode.nbrCount>=5);
      if((wDeep||prevDeep)&&prevNode?.neighbors&&node.neighbors){
        const pN=Object.keys(prevNode.neighbors),wN=new Set(Object.keys(node.neighbors));
        const sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;
        step.afGate=un?sh/un:0;
      }
    }
    step.surprise=wts>0?(sig/wts)*(1-step.afGate*0.3):(node?0.5:1);
    total+=step.surprise;step.cumS=total;steps.push(step);
    if(node?.neighbors){
      const mass=wDeep?1:0.2;
      for(const[nb,cnt]of Object.entries(node.neighbors)){const nv=(field.get(nb)||0)+cnt*mass;field.set(nb,nv);if(nv>fieldMax)fieldMax=nv;}
    }
    {const wv=(field.get(w)||0)+(wDeep?10:2);field.set(w,wv);if(wv>fieldMax)fieldMax=wv;}
  }
  const ms=steps.length?total/steps.length:0;
  return{words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)};
}

// ─── Pressure ─────────────────────────────────────────────────────
pressureOf(w){const node=this.nodes[w];if(!node)return null;const inbound=Object.values(node.prev||{}).reduce((s,v)=>s+v,0);const actual=node.nbrCount+Object.keys(node.next).length+Object.keys(node.prev).length;const p=actual-inbound;const nbrs=Object.keys(node.neighbors);let internal=0,pairs=0;for(let i=0;i<Math.min(nbrs.length,15);i++)for(let j=i+1;j<Math.min(nbrs.length,15);j++){pairs++;if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]])internal++;}const closure=pairs>0?internal/pairs:1;return{word:w,pressure:p,inbound,actual,closure,freq:node.freq,depth:this.depth(w).level};}
pressure(minFreq=2){const map=[];for(const[w,n]of Object.entries(this.nodes)){if(n.freq<minFreq||n.nbrCount<3)continue;map.push(this.pressureOf(w));}return map.sort((a,b)=>a.pressure-b.pressure);}
vacuums(k=10){return this.pressure().filter(p=>p.pressure<0).slice(0,k);}
surpluses(k=10){return this.pressure().filter(p=>p.pressure>0).sort((a,b)=>b.pressure-a.pressure).slice(0,k);}
bridges(k=10){return this.pressure().filter(p=>p.closure<0.3&&p.freq>=3).slice(0,k);}

// ─── Tensions + Collapse ──────────────────────────────────────────
tensions(k=10){const fT=Math.max(...Object.values(this.nodes).map(n=>n.freq))*0.3;const res=[];for(const[w,nd]of Object.entries(this.nodes)){if(nd.freq<3)continue;const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<fT);if(nbrs.length<4)continue;const adj={};for(const n of nbrs)adj[n]=[];for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}const vis=new Set();const comps=[];for(const n of nbrs){if(vis.has(n))continue;const comp=[n],q=[n];vis.add(n);while(q.length){const cur=q.shift();for(const nx of(adj[cur]||[]))if(!vis.has(nx)){vis.add(nx);q.push(nx);comp.push(nx);}}comps.push(comp);}const real=comps.filter(c=>c.length>=2);if(real.length>=2)res.push({word:w,components:real.length,sizes:real.map(c=>c.length),clusters:real.map(c=>c.slice(0,4)),tension:1-real[0].length/nbrs.length,freq:nd.freq});}return res.sort((a,b)=>b.tension-a.tension).slice(0,k);}
collapse(w){w=w.toLowerCase();const nd=this.nodes[w];if(!nd)return{ok:false,reason:"unknown"};const fT=Math.max(...Object.values(this.nodes).map(n=>n.freq))*0.3;const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<fT);if(nbrs.length<4)return{ok:false,reason:"not enough structure"};const adj={};for(const n of nbrs)adj[n]=[];for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}const vis=new Set();const comps=[];for(const n of nbrs){if(vis.has(n))continue;const comp=[n],q=[n];vis.add(n);while(q.length){const cur=q.shift();for(const nx of(adj[cur]||[]))if(!vis.has(nx)){vis.add(nx);q.push(nx);comp.push(nx);}}comps.push(comp);}const real=comps.filter(c=>c.length>=2);if(real.length<2)return{ok:false,reason:"no split"};real.sort((a,b)=>b.length-a.length);const keep=new Set(real[0]);let pruned=0;for(const nb of nbrs){if(!keep.has(nb)){delete nd.neighbors[nb];nd.nbrCount--;delete nd.next[nb];delete nd.prev[nb];pruned++;}}return{ok:true,word:w,kept:real[0].length,pruned,surviving:real[0].slice(0,5),removed:real.slice(1).map(c=>c.slice(0,3))};}

// ─── Forgetting ───────────────────────────────────────────────────
unlearn(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{changed:false};let r=0;for(const[src,tgt]of[[na,bl],[nb,al]]){if(src.neighbors[tgt]){src.neighbors[tgt]=Math.floor(src.neighbors[tgt]/2);if(src.neighbors[tgt]<=0){delete src.neighbors[tgt];src.nbrCount--;}r++;}if(src.next[tgt]){src.next[tgt]=Math.floor(src.next[tgt]/2);if(src.next[tgt]<=0)delete src.next[tgt];r++;}if(src.prev[tgt]){src.prev[tgt]=Math.floor(src.prev[tgt]/2);if(src.prev[tgt]<=0)delete src.prev[tgt];r++;}if(src.next2&&src.next2[tgt])delete src.next2[tgt];}return{changed:r>0,edgesWeakened:r};}
forget(word){const w=word.toLowerCase();if(!this.nodes[w])return{changed:false};for(const[,other]of Object.entries(this.nodes)){if(other.neighbors[w]){delete other.neighbors[w];other.nbrCount--;}delete other.next[w];delete other.prev[w];if(other.next2)delete other.next2[w];for(const mid of Object.keys(other.next2||{}))if(other.next2[mid]&&other.next2[mid][w]){delete other.next2[mid][w];if(!Object.keys(other.next2[mid]).length)delete other.next2[mid];}}delete this.nodes[w];return{changed:true,word:w};}
decay(threshold=1){let removed=0;for(const[,node]of Object.entries(this.nodes)){for(const[nb,cnt]of Object.entries(node.neighbors))if(cnt<=threshold){delete node.neighbors[nb];node.nbrCount--;removed++;}for(const[nb,cnt]of Object.entries(node.next))if(cnt<=threshold){delete node.next[nb];removed++;}for(const[nb,cnt]of Object.entries(node.prev))if(cnt<=threshold){delete node.prev[nb];removed++;}}return{edgesRemoved:removed};}

// ─── Compact ─────────────────────────────────────────────────────
compact(maxNbrs=30){let stripped=0;
  for(const[w,nd]of Object.entries(this.nodes)){
    if(nd.nbrCount<=maxNbrs)continue;
    const sorted=Object.entries(nd.neighbors).sort((a,b)=>b[1]-a[1]);
    const keep=new Set(sorted.slice(0,maxNbrs).map(([k])=>k));
    for(const[k]of sorted){if(!keep.has(k)){delete nd.neighbors[k];nd.nbrCount--;stripped++;}}
  }
  return{stripped};}

// ─── Correct (indexed) ───────────────────────────────────────────
correct(garbled,k=5){const g=garbled.toLowerCase(),cands=new Set();for(let d=-2;d<=2;d++){const ws=this._idxLen[g.length+d];if(ws)for(const w of ws)cands.add(w);}for(let i=0;i<g.length-1;i++){const ws=this._idxBg[g.slice(i,i+2)];if(ws)for(const w of ws)cands.add(w);}const scored=[...cands].map(w=>{const o=1-ocrDistance(g,w)/Math.max(g.length,w.length,1),b=sharedBigrams(g,w);return{word:w,score:o*0.7+b*0.3};});scored.sort((a,b)=>b.score-a.score||(this.nodes[b.word]?.freq||0)-(this.nodes[a.word]?.freq||0));const top=scored.slice(0,k),conf=top.length>=2?top[0].score-top[1].score:top.length?1:0;return{candidates:top,confidence:conf};}

// ─── Similar (2-hop) ──────────────────────────────────────────────
similar(w,k=8){const wl=w.toLowerCase(),node=this.nodes[wl];if(!node)return[];const cands=new Set();for(const nb of Object.keys(node.neighbors)){cands.add(nb);const nbNode=this.nodes[nb];if(nbNode)for(const nb2 of Object.keys(nbNode.neighbors))cands.add(nb2);}for(const nb of Object.keys(node.next||{}))cands.add(nb);for(const nb of Object.keys(node.prev||{}))cands.add(nb);cands.delete(wl);const out=[];for(const c of cands){if(!this.nodes[c])continue;out.push({word:c,...this.compare(wl,c)});}return out.sort((a,b)=>b.similarity-a.similarity).slice(0,k);}

// ─── Path (Dijkstra) — content-weighted cost ─────────────────────
path(start,goal,maxLen=15){const sl=start.toLowerCase(),gl=goal.toLowerCase();if(!this.nodes[sl]||!this.nodes[gl])return{ok:false,reason:"unknown word"};const dist={},prev={},visited=new Set();dist[sl]=0;prev[sl]=null;const pq=[sl];while(pq.length){pq.sort((a,b)=>(dist[a]||Infinity)-(dist[b]||Infinity));const u=pq.shift();if(visited.has(u))continue;visited.add(u);if(u===gl)break;const nd=this.nodes[u];if(!nd||!nd.next)continue;const total=Object.values(nd.next).reduce((s,v)=>s+v,0);let pathLen=0,trace=u;while(prev[trace]){pathLen++;trace=prev[trace];}if(pathLen>=maxLen)continue;for(const[nb,cnt]of Object.entries(nd.next)){if(visited.has(nb))continue;const prob=cnt/total;const surprise=-Math.log(prob+1e-10);const freqPenalty=nb===gl?1:Math.max(Math.log2((this.nodes[nb]?.freq||1)+1),1);const cost=surprise*freqPenalty;const newDist=dist[u]+cost;if(newDist<(dist[nb]??Infinity)){dist[nb]=newDist;prev[nb]=u;pq.push(nb);}}}if(dist[gl]===undefined)return{ok:false,reason:'no path'};const words=[];let cur=gl;while(cur!==null){words.unshift(cur);cur=prev[cur];}const text=words.join(' ');const score=this.scoreSentence(text);return{ok:true,words,text,energy:dist[gl],coherence:score.coherence,steps:score.steps};}

// ─── Generate: start → bridge → goal ──────────────────────────────
generate(startWord,goalWord,bridgeWord,maxLen=12){
  let w=startWord?startWord.toLowerCase():null;
  const goal=goalWord?goalWord.toLowerCase():null;
  const bridge=bridgeWord?bridgeWord.toLowerCase():null;
  if(!w||!this.nodes[w]){const cs=Object.entries(this.nodes).filter(([,n])=>n.freq>=3&&n.freq<=50&&Object.keys(n.next).length>=2);if(!cs.length)return{words:[],text:"",coherence:0};w=cs[Math.floor(Math.random()*cs.length)][0];}
  const words=[w],used=new Set([w]);const field=new Map();
  const scope=new Set();const startNode=this.nodes[w];
  if(startNode){for(const nb of Object.keys(startNode.neighbors)){scope.add(nb);const nbNode=this.nodes[nb];if(nbNode)for(const nb2 of Object.keys(nbNode.neighbors))scope.add(nb2);}}
  if(goal&&this.nodes[goal])for(const nb of Object.keys(this.nodes[goal].neighbors))scope.add(nb);
  if(bridge&&this.nodes[bridge])for(const nb of Object.keys(this.nodes[bridge].neighbors))scope.add(nb);
  const addToField=(word)=>{const nd=this.nodes[word];if(!nd)return;for(const[nb,cnt]of Object.entries(nd.neighbors)){const rr=1/Math.max(Math.log2((this.nodes[nb]?.freq||1)+1),0.5);field.set(nb,(field.get(nb)||0)+cnt*rr);}field.set(word,(field.get(word)||0)+5);};
  addToField(w);
  if(bridge&&this.nodes[bridge])for(const[nb,cnt]of Object.entries(this.nodes[bridge].neighbors)){const rr=1/Math.max(Math.log2((this.nodes[nb]?.freq||1)+1),0.5);field.set(nb,(field.get(nb)||0)+cnt*rr*0.5);}
  if(goal&&this.nodes[goal])for(const[nb,cnt]of Object.entries(this.nodes[goal].neighbors)){const rr=1/Math.max(Math.log2((this.nodes[nb]?.freq||1)+1),0.5);field.set(nb,(field.get(nb)||0)+cnt*rr*0.2);}
  let bridgeReached=!bridge;
  for(let i=0;i<maxLen-1;i++){
    const nd=this.nodes[w];if(!nd)break;
    const entries=Object.entries(nd.next).filter(([nw])=>!used.has(nw)||words.length>6);if(!entries.length)break;
    const seqTotal=entries.reduce((s,[,c])=>s+c,0);const fieldMax=Math.max(1,...field.values());const progress=words.length/maxLen;
    const target=(!bridgeReached&&bridge)?bridge:goal;
    const prevWord=words.length>=2?words[words.length-2]:null;const prevNode=prevWord?this.nodes[prevWord]:null;
    const trigrams=prevNode?.next2?.[w]||null;const trigTotal=trigrams?Object.values(trigrams).reduce((s,v)=>s+v,0):0;
    const scored=entries.map(([nw,cnt])=>{const nNode=this.nodes[nw];const freq=nNode?.freq||1;const seq=cnt/seqTotal;const traj=trigrams&&trigrams[nw]?trigrams[nw]/trigTotal:0;const fld=(field.get(nw)||0)/fieldMax*(1/Math.max(Math.log2(freq+1),0.5));const mass=nNode&&nNode.nbrCount>=5&&freq>=5?1.2:freq>=3?0.8:0.4;let posfit=1;if(nNode&&nNode.freq>=3&&nNode.posV<0.2){posfit=1-Math.abs(nNode.pos-progress)*0.8;}let pull=0;if(target&&nNode?.next){if(nw===target)pull=3;else if(nNode.next[target])pull=1.5;else if(nNode.neighbors[target])pull=0.5;}const inScope=scope.size===0||scope.has(nw)||freq>50?1:0.3;return{w:nw,score:((seq*0.20+traj*0.15+fld*0.20+pull*0.20+posfit*0.10)*mass+seq*0.15)*inScope};});
    const totalScore=scored.reduce((s,x)=>s+Math.max(x.score,0.01),0);let r=Math.random()*totalScore,pick=scored[0].w;for(const x of scored){r-=Math.max(x.score,0.01);if(r<=0){pick=x.w;break;}}
    words.push(pick);used.add(pick);addToField(pick);w=pick;if(pick===bridge)bridgeReached=true;if(pick===goal)break;
  }
  const text=words.join(' ');const sc=this.scoreSentence(text);return{words,text,coherence:sc.coherence};
}

// ─── Identity ─────────────────────────────────────────────────────
identity(word,k=8){
  const wl=word.toLowerCase(),nd=this.nodes[wl];if(!nd)return[];
  const topicNbrs=new Set(Object.keys(nd.neighbors));const candidates=[];
  for(const[nb,edgeWeight]of Object.entries(nd.neighbors)){
    const nbNode=this.nodes[nb];if(!nbNode||nbNode.freq<3)continue;
    const theirNbrs=Object.keys(nbNode.neighbors);const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);
    let shared=0,total=0;for(const n of theirNbrs){total+=rw(n);if(topicNbrs.has(n))shared+=rw(n);}for(const n of [...topicNbrs])if(!nbNode.neighbors[n])total+=rw(n);
    const orbit=total>0?shared/total:0;const specificity=edgeWeight/nbNode.freq;const pull=(nd.next[nb]||0)+(nd.prev[nb]||0);
    const heat=orbit*0.35+specificity*0.35+Math.min(pull/10,1)*0.30;
    if(heat>0.01)candidates.push({word:nb,orbit,specificity,pull,heat,freq:nbNode.freq,depth:this.depth(nb).level});
  }
  candidates.sort((a,b)=>b.heat-a.heat);return candidates.slice(0,k);
}

// ─── Speak ────────────────────────────────────────────────────────
speak(topic,count=5,maxLen=12){
  const res=[];const tl=topic?.toLowerCase();const tn=this.nodes[tl];
  if(!tn)return Array.from({length:count},()=>this.generate(null,null,null,maxLen));
  const freqs=Object.values(this.nodes).map(n=>n.freq).sort((a,b)=>b-a);const ceiling=freqs[Math.floor(freqs.length*0.02)]||100;
  const orbit=this.identity(tl,20);const innerPlanets=orbit.filter(p=>p.freq<ceiling).map(p=>p.word);
  const seqNbrs=Object.entries(tn.next||{}).filter(([w])=>{const nd=this.nodes[w];return nd&&nd.freq>=3&&nd.freq<ceiling;}).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([w])=>w);
  const allGoals=[...new Set([...innerPlanets,...seqNbrs])].filter(w=>w!==tl);const starts=[tl,...innerPlanets.slice(0,5)];
  for(let i=allGoals.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[allGoals[i],allGoals[j]]=[allGoals[j],allGoals[i]];}
  let gi=0;const seen=new Set();
  for(let i=0;i<count;i++){
    const start=starts[i%starts.length];const goal=allGoals.length?allGoals[gi++%allGoals.length]:null;
    if(goal){const p=this.path(start,goal,maxLen);if(p.ok&&p.words.length>=3&&!seen.has(p.text)){seen.add(p.text);res.push({words:p.words,text:p.text,coherence:p.coherence});continue;}}
    let bridge=null;
    if(goal&&this.nodes[start]&&this.nodes[goal]){let best=0;for(const[cand,cnt]of Object.entries(this.nodes[start].next)){if(cand===goal||cand===start)continue;const cN=this.nodes[cand];if(!cN||cN.freq<3)continue;const toGoal=(cN.next[goal]||0)+(cN.neighbors[goal]||0)+(this.nodes[goal].prev[cand]||0);if(toGoal>0){const sc=cnt*toGoal*(1/Math.max(Math.log2(cN.freq+1),0.5));if(sc>best){best=sc;bridge=cand;}}}}
    const g=this.generate(start,goal,bridge,maxLen);if(g.text&&!seen.has(g.text)){seen.add(g.text);res.push(g);}
  }
  return res;
}

// ─── Query + Ask ─────────────────────────────────────────────────��
query(pattern,k=5){const ph='QBLANK';const raw=pattern.replace(/_/g,ph);const words=tokenize(raw).map(w=>w==='qblank'?'_':w);const bi=words.indexOf('_');if(bi===-1)return{ok:false,reason:"use _ for the blank"};const cands=new Set();for(let i=0;i<words.length;i++){if(i===bi)continue;const nd=this.nodes[words[i]];if(!nd)continue;for(const nb of Object.keys(nd.neighbors))cands.add(nb);for(const nb of Object.keys(nd.next||{}))cands.add(nb);for(const nb of Object.keys(nd.prev||{}))cands.add(nb);}if(bi>0){const pv=this.nodes[words[bi-1]];if(pv)for(const nb of Object.keys(pv.next||{}))cands.add(nb);}if(bi<words.length-1){const nx=this.nodes[words[bi+1]];if(nx)for(const nb of Object.keys(nx.prev||{}))cands.add(nb);}for(const w of words)cands.delete(w);cands.delete('_');const results=[];for(const c of cands){const filled=[...words];filled[bi]=c;const sc=this.scoreSentence(filled.join(' '));results.push({word:c,coherence:sc.coherence,freq:this.nodes[c]?.freq||0,depth:this.depth(c).level});}results.sort((a,b)=>b.coherence-a.coherence);return{ok:true,pattern,blank:bi,results:results.slice(0,k),total:cands.size};}
ask(question){let q=question.toLowerCase().replace(/[?!.]/g,'').trim();q=q.replace(/^(what|which|who)\s+/i,'_ ');q=q.replace(/\s+(what|which|who)\s+/gi,' _ ');if(!q.includes('_')){q=q.replace(/^(does|do|did|can|will|is|are|was|were)\s+/i,'');const ws=q.split(/\s+/);if(ws.length>=2){ws.splice(1,0,'_');q=ws.join(' ');}else return{ok:false,reason:"use _ for the blank"};}return this.query(q);}

// ─── Respond ──────────────────────────────────────────────────────
respond(input){
  const words=tokenize(input);if(!words.length)return{type:"empty",text:"..."};
  const qW=new Set(['what','who','how','why','when','where','which','does','do','did','can','will','is','are','was','were','tell','me','about','define','describe','explain','the','an','it','related','between','compare','connection','relationship','and']);
  const stripped=words.filter(w=>!qW.has(w));const known=stripped.filter(w=>this.nodes[w]);const unknown=stripped.filter(w=>!this.nodes[w]);
  const content=known;const hasRel=words.some(w=>w==='how'||w==='between'||w==='related'||w==='compare'||w==='and');
  if(unknown.length>content.length&&stripped.length>3){const b=Object.keys(this.nodes).length;this.feedText(input);const a=Object.keys(this.nodes).length;return{type:"learned",text:"Learned. +"+(a-b)+" new.",vocab:a};}
  if(content.length>=2&&hasRel){const a=content[0],b=content[content.length>2?content.length-1:1];const af=this.affinity(a,b);const p=this.path(a,b,12);let text="";if(af.known&&af.mutual>0.01)text+=a+" \u2194 "+b+": "+(af.mutual*100).toFixed(0)+"% affinity\n";if(p.ok&&p.words.length>=3)text+='"'+p.text+'"\n';const br=p.ok&&p.words.length>3?p.words[Math.floor(p.words.length/2)]:null;const g=this.generate(a,b,br,12);if(g.text&&g.words.length>=3)text+=g.text+"\n";if(!text.trim()){const s=this.speak(a,2,12);for(const x of s)if(x.text)text+=x.text+"\n";}return{type:"relationship",a,b,affinity:af,path:p,text:text.trim()};}
  if(content.length>=1){const topic=content[0];const id=this.identity(topic,6);const bolus=id.filter(p=>p.specificity>0.05);const sents=this.speak(topic,3,12);let text=topic;if(bolus.length)text+=" \u2192 "+bolus.map(p=>p.word).join(", ");text+="\n";for(const s of sents)if(s.text&&s.words.length>=3)text+=s.text+"\n";return{type:"identity",topic,identity:bolus,sentences:sents,text:text.trim()};}
  if(known.length>0){const deepest=known.sort((a,b)=>this.depth(b).evidence-this.depth(a).evidence)[0];const sents=this.speak(deepest,4,12);let text="";for(const s of sents)if(s.text&&s.words.length>=3)text+=s.text+"\n";if(!text)text='I know "'+deepest+'" but need more data.';return{type:"speak",topic:deepest,sentences:sents,text:text.trim()};}
  this.feedText(input);return{type:"absorbed",text:"Absorbed. Vocab: "+Object.keys(this.nodes).length+"."};
}

// ─── Converse ─────────────────────────────────────────────────────
converse(input){
  this.feedText(input);const resp=this.respond(input);
  const topics=[];if(resp.topic)topics.push(resp.topic);if(resp.a)topics.push(resp.a);if(resp.b)topics.push(resp.b);
  if(!topics.length){const qW=new Set(['what','who','how','why','when','where','which','does','do','did','can','will','is','are','was','were','tell','me','about','define','describe','explain','the','an','it','related','between','compare','connection','relationship','and']);const words=tokenize(input).filter(w=>!qW.has(w)&&this.nodes[w]);if(words.length)topics.push(words[0]);}
  let question=null;
  for(const topic of topics){const nd=this.nodes[topic];if(!nd)continue;
    const shallow=[];for(const[nb]of Object.entries(nd.neighbors)){const nbd=this.nodes[nb];if(!nbd)continue;const d=this.depth(nb);if(d.level==="surface"||d.level==="shallow")shallow.push(nb);}
    const unreachable=[];for(const[nb]of Object.entries(nd.neighbors)){if(!this.nodes[nb])continue;const p=this.path(topic,nb,8);if(!p.ok)unreachable.push(nb);}
    const id=this.identity(topic,10);const mysterious=id.filter(p=>p.specificity>0.1&&p.orbit<0.1);
    if(mysterious.length>0){question=`What connects ${topic} to ${mysterious[0].word}? I see them together but don't understand why.`;}
    else if(unreachable.length>0){const pick=unreachable[Math.floor(Math.random()*Math.min(unreachable.length,3))];question=`How does ${topic} relate to ${pick}? I know they're connected but can't trace the path.`;}
    else if(shallow.length>0){const pick=shallow[Math.floor(Math.random()*Math.min(shallow.length,5))];question=`Tell me more about ${pick}. I've seen it near ${topic} but don't know it well.`;}
    if(question)break;
  }
  if(!question&&topics.length>0)question=`I understand ${topics[0]} well. What else should I learn about?`;
  return{...resp,question,text:resp.text+(question?'\n\n'+question:'')};
}

// ─── Binding: temporal field persistence across sentence gaps ─────
// "The doctor prescribed medication. The patient recovered."
// Field from sentence 1 carries into sentence 2 (decayed).
// "patient" surprise drops because "doctor" and "prescribed" are still in the field.
scoreDocument(text, decay=0.5){
  const sents=text.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);
  if(!sents.length)return{sentences:[],coherence:0,bindings:[]};
  const field=new Map();let fMax=1;const results=[];const bindings=[];
  for(let si=0;si<sents.length;si++){
    const ws=tokenize(sents[si]);if(ws.length<2)continue;
    const steps=[];let total=0;
    for(let i=0;i<ws.length;i++){
      const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node,sentence:si};
      let sig=0,wts=0;
      const wDeep=node&&(node.freq>=5&&node.nbrCount>=5);
      if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*0.35;wts+=0.35;}}
      if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*0.25;wts+=0.25;}}
      if(field.size>0){const fw=field.get(w)||0;step.fieldS=1-fw/fMax;sig+=step.fieldS*0.40;wts+=0.40;
        if(si>0&&fw>0)bindings.push({word:w,sentence:si,fieldStrength:fw/fMax});}
      step.afGate=0;
      if(i>0&&node){const prevNode=this.nodes[ws[i-1]];const prevDeep=prevNode&&(prevNode.freq>=5&&prevNode.nbrCount>=5);if((wDeep||prevDeep)&&prevNode?.neighbors&&node.neighbors){const pN=Object.keys(prevNode.neighbors),wN=new Set(Object.keys(node.neighbors));const sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;step.afGate=un?sh/un:0;}}
      step.surprise=wts>0?(sig/wts)*(1-step.afGate*0.3):(node?0.5:1);
      total+=step.surprise;step.cumS=total;steps.push(step);
      if(node?.neighbors){const mass=wDeep?1:0.2;for(const[nb,cnt]of Object.entries(node.neighbors)){const nv=(field.get(nb)||0)+cnt*mass;field.set(nb,nv);if(nv>fMax)fMax=nv;}}
      {const wv=(field.get(w)||0)+(wDeep?10:2);field.set(w,wv);if(wv>fMax)fMax=wv;}
    }
    const ms=steps.length?total/steps.length:0;
    results.push({text:sents[si],words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)});
    // Decay field at sentence boundary — recent context fades but persists
    for(const[k,v]of field)field.set(k,v*decay);
  }
  const overall=results.length?results.reduce((s,r)=>s+r.coherence,0)/results.length:0;
  return{sentences:results,coherence:overall,bindings};
}

// ─── Self-Teach: the engine does its own homework ─────────────────
selfTeach(){
  const st=this.stats();if(st.vocab<20)return{action:"waiting",detail:"need more data"};
  const report={resolved:[],strengthened:[],collapsed:[],decayed:0,action:"studied"};
  const planets=Object.keys(this.nodes).filter(w=>{
    const d=this.depth(w);return(d.level==="structured"||d.level==="deep")&&this.nodes[w].freq>=5;
  });
  if(!planets.length)return{action:"resting",detail:"no planets yet"};
  const topic=planets[Math.floor(Math.random()*planets.length)];
  const nd=this.nodes[topic];if(!nd)return{action:"resting"};
  const id=this.identity(topic,10);
  for(const planet of id.slice(0,5)){
    if(planet.specificity<0.05)continue;
    const p=this.path(topic,planet.word,10);
    if(p.ok&&p.words.length>=3){this.feed(p.text);report.resolved.push({from:topic,to:planet.word,via:p.text});}
    else{const bridge=Object.keys(nd.next).find(w=>{const wn=this.nodes[w];return wn&&(wn.next[planet.word]||wn.neighbors[planet.word]);});
      if(bridge){const gen=this.generate(topic,planet.word,bridge,10);if(gen.text&&gen.words.length>=3){this.feed(gen.text);report.strengthened.push({from:topic,to:planet.word,bridge,text:gen.text});}}}
  }
  const shallow=Object.keys(nd.neighbors).filter(w=>{const d=this.depth(w);return d.level==="shallow"||d.level==="surface";}).slice(0,3);
  for(const sw of shallow){const gen=this.generate(topic,sw,null,8);if(gen.text&&gen.words.length>=3){this.feed(gen.text);report.strengthened.push({from:topic,to:sw,text:gen.text});}}
  const tens=this.tensions(2);
  for(const t of tens){const c=this.collapse(t.word);if(c.ok)report.collapsed.push({word:t.word,kept:c.kept,pruned:c.pruned});}
  if(Math.random()<0.3){const dc=this.decay(1);report.decayed=dc.edgesRemoved;}
  return report;
}

// ─── Study: run N rounds of self-teaching ─────────────────────────
study(rounds=10,onRound){
  const results=[];
  for(let i=0;i<rounds;i++){
    const r=this.selfTeach();r.round=i+1;results.push(r);
    if(onRound)onRound(r);
    if(r.action==="waiting"||r.action==="resting")break;
  }
  return results;
}

// ─── Stats ────────────────────────────────────────────────────────
stats(){const v=Object.keys(this.nodes).length,depths={unborn:0,surface:0,shallow:0,forming:0,structured:0,deep:0};for(const w of Object.keys(this.nodes))depths[this.depth(w).level]++;return{version:VERSION,vocab:v,sentences:this.sentenceCount,tokens:this.tokenCount,depths};}

// ─── Serialize ────────────────────────────────────────────────────
serialize(){return JSON.stringify({version:VERSION,nodes:Object.fromEntries(Object.entries(this.nodes).map(([w,n])=>[w,{c:n.chars,f:n.freq,fs:n.firstSeen,ls:n.lastSeen,p:n.pos,pv:n.posV,lg:n.lastGap,nc:n.nbrCount,nb:n.neighbors,nx:n.next,px:n.prev,n2:n.next2}])),sc:this.sentenceCount,tc:this.tokenCount});}
static deserialize(json){const d=JSON.parse(json),e=new ShifuEmbryo();e.sentenceCount=d.sc||d.sentenceCount||0;e.tokenCount=d.tc||d.tokenCount||0;for(const[w,n]of Object.entries(d.nodes||d.n||{})){const nb=n.nb||n.neighbors||{};const nc=n.nc||Object.keys(nb).length;const freq=n.f||n.freq||0;const pv=n.pv??n.posV??1;e.nodes[w]={chars:n.c||n.chars||w,freq,firstSeen:n.fs||n.firstSeen,lastSeen:n.ls||n.lastSeen,pos:n.p??n.pos??0.5,posV:pv,lastGap:n.lg??n.lastGap??0,nbrCount:nc,frozen:freq>30&&pv<0.05&&nc>20,neighbors:nb,next:n.nx||n.next||{},prev:n.px||n.prev||{},next2:n.n2||n.next2||{}};e._idx(w);}return e;}
}
module.exports={ShifuEmbryo,VERSION,editDistance,ocrDistance,sharedBigrams,tokenize};
