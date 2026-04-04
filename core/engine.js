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
_nn(w){return{chars:w,freq:0,firstSeen:null,lastSeen:null,pos:0.5,posV:1.0,lastGap:0,neighbors:{},next:{},prev:{},next2:{}};}

_idx(w){const l=w.length;(this._idxLen[l]??=[]);if(!this._idxLen[l].includes(w))this._idxLen[l].push(w);for(let i=0;i<w.length-1;i++){const bg=w.slice(i,i+2);(this._idxBg[bg]??=[]);if(!this._idxBg[bg].includes(w))this._idxBg[bg].push(w);}}

// ─── Feed: consolidation, not accumulation. ───────────────────────
// Position updates via running mean weighted by plasticity.
// Neighbor edges strengthen existing connections; new connections
// are added at plasticity-scaled strength (deep words resist new links).
// Phase conflict still active for incompatible neighborhoods.
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

    // Plasticity: how much this word can still change
    // freq=1 → 1.0 (fully plastic). freq=10 → 0.1. freq=100 → 0.01.
    const plasticity=1/nd.freq;

    // Position: running mean that hardens
    const rp=i/Math.max(len-1,1);
    if(nd.freq===1){
      nd.pos=rp;nd.posV=0.25; // first encounter: set directly, high variance
    }else{
      const delta=rp-nd.pos;
      nd.pos+=delta*plasticity;              // move toward new position (less each time)
      nd.posV+=(delta*delta-nd.posV)*plasticity; // variance shrinks with exposure
    }

    // Variable window: rare words reach deeper
    const win=Math.min(Math.max(Math.ceil(4/Math.max(Math.log2(nd.freq+1),1)),2),6);

    // Deep rigid words: skip neighbor accumulation entirely.
    // "the" doesn't need to know it neighbors everything. Only next/prev matters.
    // Threshold: freq > 30 AND posV < 0.05 AND neighbors > 20 → frozen neighbors.
    const frozen=nd.freq>30&&nd.posV<0.05&&Object.keys(nd.neighbors).length>20;

    if(!frozen){
      const myNbrs=Object.keys(nd.neighbors);const hasStructure=myNbrs.length>=5;
      for(let j=Math.max(0,i-win);j<Math.min(len,i+win+1);j++){
        if(j===i)continue;const nb=ws[j];
        const existing=nd.neighbors[nb];
        if(existing){
          // Existing: reinforce but cap at freq/3 (prevents unbounded growth)
          nd.neighbors[nb]=Math.min(existing+1,nd.freq/3);
        }else{
          if(hasStructure&&this.nodes[nb]){
            const theirNbrs=new Set(Object.keys(this.nodes[nb].neighbors||{}));
            const overlap=myNbrs.filter(x=>theirNbrs.has(x)).length;
            const phase=myNbrs.length>0?overlap/myNbrs.length:1;
            if(phase<0.05){
              nd.neighbors[nb]=plasticity*0.5;
              const weakest=Object.entries(nd.neighbors).filter(([k])=>k!==nb).sort((a,b)=>a[1]-b[1]).slice(0,2);
              for(const[k]of weakest){nd.neighbors[k]*=0.95;if(nd.neighbors[k]<0.5)delete nd.neighbors[k];}
            }else{
              nd.neighbors[nb]=Math.max(plasticity,0.5);
            }
          }else{
            nd.neighbors[nb]=nd.freq<=5?1:Math.max(plasticity*2,0.5);
          }
        }
      }
    }

    // Sequential: always full strength (word order is rigid)
    if(i<len-1)nd.next[ws[i+1]]=(nd.next[ws[i+1]]||0)+1;
    if(i>0)nd.prev[ws[i-1]]=(nd.prev[ws[i-1]]||0)+1;
    if(i<len-2){const b=ws[i+1],c=ws[i+2];nd.next2[b]??={};nd.next2[b][c]=(nd.next2[b][c]||0)+1;}
  }
  return ws.length;
}

feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0;for(const x of s)tk+=this.feed(x);return{sentences:s.length,tokens:tk};}

// ─── Depth: uses pos variance as signal ───────────────────────────
// Low posV = position is settled = deep knowledge. High posV = still moving.
depth(w){
  const n=this.nodes[w];if(!n)return{level:"unborn",evidence:0};
  if(n.freq===1)return{level:"surface",evidence:0.1};
  if(n.freq<5)return{level:"shallow",evidence:0.2};
  const nb=Object.keys(n.neighbors).length;
  const sq=Object.keys(n.next).length+Object.keys(n.prev).length;
  const stability=1-Math.min(n.posV*4,1); // posV→0 = stable → high evidence
  const ev=Math.min(
    (n.freq/50)*0.25 +
    (nb/20)*0.25 +
    (sq/10)*0.2 +
    stability*0.3,    // position stability is the strongest signal
    1.0
  );
  if(ev<0.3)return{level:"forming",evidence:ev};
  if(ev<0.7)return{level:"structured",evidence:ev};
  return{level:"deep",evidence:ev};
}

// ─── Compare: uses consolidated pos, not arrays ───────────────────
compare(a,b){
  const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl],sig={},wt={};
  sig.editSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);
  sig.bigramSim=sharedBigrams(al,bl);
  sig.ocrSim=1-ocrDistance(al,bl)/Math.max(al.length,bl.length,1);
  wt.char=0.05;
  if(!na||!nb)return{similarity:sig.editSim*0.3+sig.bigramSim*0.3+sig.ocrSim*0.4,signals:sig,weights:wt,totalWeight:0.05,depth:"surface"};

  // Neighbors
  const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),union=new Set([...nbA,...nbBs]);
  if(union.size>0){const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...union].reduce((s,w)=>s+rw(w),0);sig.neighborOverlap=uw>0?sw/uw:0;wt.neighbor=Math.min(union.size/10,0.35);}

  // Sequential
  const totA=Object.values(na.next).reduce((s,v)=>s+v,0),totB=Object.values(nb.next).reduce((s,v)=>s+v,0);
  if(totA>0||totB>0){sig.expectsAB=totA?(na.next[bl]||0)/totA:0;sig.expectsBA=totB?(nb.next[al]||0)/totB:0;sig.directional=Math.abs(sig.expectsAB-sig.expectsBA);wt.seq=Math.min((totA+totB)/20,0.25);}

  // Trajectory
  const nx2AB=na.next2?.[bl]?Object.keys(na.next2[bl]).length:0,nx2BA=nb.next2?.[al]?Object.keys(nb.next2[al]).length:0;
  if(nx2AB>0||nx2BA>0){sig.trajectoryAB=Math.min(nx2AB/5,1);sig.trajectoryBA=Math.min(nx2BA/5,1);wt.traj=0.15;}

  // Positional similarity: consolidated pos, weighted by how settled both are
  const bothSettled=na.freq>=3&&nb.freq>=3;
  if(bothSettled){
    sig.posSim=1-Math.min(Math.abs(na.pos-nb.pos)*2,1);
    const settledWeight=Math.min((1-na.posV)*(1-nb.posV)*4,1); // only trust if both are settled
    wt.pos=settledWeight*0.15;
  }

  // Indirect
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

// ─── Affinity: consolidated position ──────────────────────────────
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
  // Positional alignment: only if both settled
  let posAlign=0;
  if(na.freq>=3&&nb.freq>=3)posAlign=1-Math.min(Math.abs(na.pos-nb.pos)*3,1);
  const afAB=charSim*0.05+orbit*0.35+pullAB*0.20+indAB*0.20+expOvlp*0.15+posAlign*0.05;
  const afBA=charSim*0.05+orbit*0.35+pullBA*0.20+indBA*0.20+expOvlp*0.15+posAlign*0.05;
  return{a:al,b:bl,orbit,pullAB,pullBA,indAB,indBA,charSim,expOvlp,posAlign,afAB,afBA,mutual:(afAB+afBA)/2,asym:Math.abs(afAB-afBA),known:true};
}

// ─── Score sentence ───────────────────────────────────────────────
scoreSentence(raw){
  const ws=tokenize(raw);if(ws.length<2)return{words:ws,steps:[],meanSurprise:0,coherence:0};
  const steps=[];let total=0;const field=new Map();
  for(let i=0;i<ws.length;i++){
    const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node};let sig=0,wts=0;
    if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*0.35;wts+=0.35;}}
    if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*0.30;wts+=0.30;}}
    if(i>0&&field.size>0){const fw=field.get(w)||0,mx=Math.max(...field.values(),1);step.fieldS=1-fw/mx;sig+=step.fieldS*0.35;wts+=0.35;}
    step.afGate=0;if(i>0&&node){const pn=this.nodes[ws[i-1]];if(pn?.neighbors&&node.neighbors){const pN=Object.keys(pn.neighbors),wN=new Set(Object.keys(node.neighbors)),sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;step.afGate=un?sh/un:0;}}
    step.surprise=wts>0?(sig/wts)*(1-step.afGate*0.3):(node?0.5:1);
    total+=step.surprise;step.cumS=total;steps.push(step);
    if(node?.neighbors)for(const[nb,cnt]of Object.entries(node.neighbors))field.set(nb,(field.get(nb)||0)+cnt);
    field.set(w,(field.get(w)||0)+10);
  }
  const ms=steps.length?total/steps.length:0;
  return{words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)};
}

// ─── Pressure ─────────────────────────────────────────────────────
pressureOf(w){const node=this.nodes[w];if(!node)return null;const inbound=Object.values(node.prev||{}).reduce((s,v)=>s+v,0);const actual=Object.keys(node.neighbors).length+Object.keys(node.next).length+Object.keys(node.prev).length;const p=actual-inbound;const nbrs=Object.keys(node.neighbors);let internal=0,pairs=0;for(let i=0;i<Math.min(nbrs.length,15);i++)for(let j=i+1;j<Math.min(nbrs.length,15);j++){pairs++;if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]])internal++;}const closure=pairs>0?internal/pairs:1;return{word:w,pressure:p,inbound,actual,closure,freq:node.freq,depth:this.depth(w).level};}
pressure(minFreq=2){const map=[];for(const[w,n]of Object.entries(this.nodes)){if(n.freq<minFreq)continue;map.push(this.pressureOf(w));}return map.sort((a,b)=>a.pressure-b.pressure);}
vacuums(k=10){return this.pressure().filter(p=>p.pressure<0).slice(0,k);}
surpluses(k=10){return this.pressure().filter(p=>p.pressure>0).sort((a,b)=>b.pressure-a.pressure).slice(0,k);}
bridges(k=10){return this.pressure().filter(p=>p.closure<0.3&&p.freq>=3).slice(0,k);}

// ─── Tensions + Collapse ──────────────────────────────────────────
tensions(k=10){const fT=Math.max(...Object.values(this.nodes).map(n=>n.freq))*0.3;const res=[];for(const[w,nd]of Object.entries(this.nodes)){if(nd.freq<3)continue;const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<fT);if(nbrs.length<4)continue;const adj={};for(const n of nbrs)adj[n]=[];for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}const vis=new Set();const comps=[];for(const n of nbrs){if(vis.has(n))continue;const comp=[n],q=[n];vis.add(n);while(q.length){const cur=q.shift();for(const nx of(adj[cur]||[]))if(!vis.has(nx)){vis.add(nx);q.push(nx);comp.push(nx);}}comps.push(comp);}const real=comps.filter(c=>c.length>=2);if(real.length>=2)res.push({word:w,components:real.length,sizes:real.map(c=>c.length),clusters:real.map(c=>c.slice(0,4)),tension:1-real[0].length/nbrs.length,freq:nd.freq});}return res.sort((a,b)=>b.tension-a.tension).slice(0,k);}
collapse(w){w=w.toLowerCase();const nd=this.nodes[w];if(!nd)return{ok:false,reason:"unknown"};const fT=Math.max(...Object.values(this.nodes).map(n=>n.freq))*0.3;const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<fT);if(nbrs.length<4)return{ok:false,reason:"not enough structure"};const adj={};for(const n of nbrs)adj[n]=[];for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}const vis=new Set();const comps=[];for(const n of nbrs){if(vis.has(n))continue;const comp=[n],q=[n];vis.add(n);while(q.length){const cur=q.shift();for(const nx of(adj[cur]||[]))if(!vis.has(nx)){vis.add(nx);q.push(nx);comp.push(nx);}}comps.push(comp);}const real=comps.filter(c=>c.length>=2);if(real.length<2)return{ok:false,reason:"no split"};real.sort((a,b)=>b.length-a.length);const keep=new Set(real[0]);let pruned=0;for(const nb of nbrs){if(!keep.has(nb)){delete nd.neighbors[nb];delete nd.next[nb];delete nd.prev[nb];pruned++;}}return{ok:true,word:w,kept:real[0].length,pruned,surviving:real[0].slice(0,5),removed:real.slice(1).map(c=>c.slice(0,3))};}

// ─── Forgetting ───────────────────────────────────────────────────
unlearn(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{changed:false};let r=0;for(const[src,tgt]of[[na,bl],[nb,al]]){if(src.neighbors[tgt]){src.neighbors[tgt]=Math.floor(src.neighbors[tgt]/2);if(src.neighbors[tgt]<=0)delete src.neighbors[tgt];r++;}if(src.next[tgt]){src.next[tgt]=Math.floor(src.next[tgt]/2);if(src.next[tgt]<=0)delete src.next[tgt];r++;}if(src.prev[tgt]){src.prev[tgt]=Math.floor(src.prev[tgt]/2);if(src.prev[tgt]<=0)delete src.prev[tgt];r++;}if(src.next2&&src.next2[tgt])delete src.next2[tgt];}return{changed:r>0,edgesWeakened:r};}
forget(word){const w=word.toLowerCase();if(!this.nodes[w])return{changed:false};for(const[,other]of Object.entries(this.nodes)){delete other.neighbors[w];delete other.next[w];delete other.prev[w];if(other.next2)delete other.next2[w];for(const mid of Object.keys(other.next2||{}))if(other.next2[mid]&&other.next2[mid][w]){delete other.next2[mid][w];if(!Object.keys(other.next2[mid]).length)delete other.next2[mid];}}delete this.nodes[w];return{changed:true,word:w};}
decay(threshold=1){let removed=0;for(const[,node]of Object.entries(this.nodes)){for(const[nb,cnt]of Object.entries(node.neighbors))if(cnt<=threshold){delete node.neighbors[nb];removed++;}for(const[nb,cnt]of Object.entries(node.next))if(cnt<=threshold){delete node.next[nb];removed++;}for(const[nb,cnt]of Object.entries(node.prev))if(cnt<=threshold){delete node.prev[nb];removed++;}}return{edgesRemoved:removed};}

// ─── Compact: strip bloated neighbor tables from function words ───
// Words with freq>50 and >100 neighbors are function words. Keep only
// the top N neighbors by weight. The rest is noise.
compact(maxNbrs=30){let stripped=0;
  for(const[w,nd]of Object.entries(this.nodes)){
    const nbrCount=Object.keys(nd.neighbors).length;
    if(nbrCount<=maxNbrs)continue;
    // Keep only top maxNbrs by edge weight
    const sorted=Object.entries(nd.neighbors).sort((a,b)=>b[1]-a[1]);
    const keep=new Set(sorted.slice(0,maxNbrs).map(([k])=>k));
    for(const[k]of sorted){if(!keep.has(k)){delete nd.neighbors[k];stripped++;}}
  }
  return{stripped};}

// ─── Correct (indexed) ───────────────────────────────────────────
correct(garbled,k=5){const g=garbled.toLowerCase(),cands=new Set();for(let d=-2;d<=2;d++){const ws=this._idxLen[g.length+d];if(ws)for(const w of ws)cands.add(w);}for(let i=0;i<g.length-1;i++){const ws=this._idxBg[g.slice(i,i+2)];if(ws)for(const w of ws)cands.add(w);}const scored=[...cands].map(w=>{const o=1-ocrDistance(g,w)/Math.max(g.length,w.length,1),b=sharedBigrams(g,w);return{word:w,score:o*0.7+b*0.3};});scored.sort((a,b)=>b.score-a.score||(this.nodes[b.word]?.freq||0)-(this.nodes[a.word]?.freq||0));const top=scored.slice(0,k),conf=top.length>=2?top[0].score-top[1].score:top.length?1:0;return{candidates:top,confidence:conf};}

// ─── Similar (2-hop) ──────────────────────────────────────────────
similar(w,k=8){const wl=w.toLowerCase(),node=this.nodes[wl];if(!node)return[];const cands=new Set();for(const nb of Object.keys(node.neighbors)){cands.add(nb);const nbNode=this.nodes[nb];if(nbNode)for(const nb2 of Object.keys(nbNode.neighbors))cands.add(nb2);}for(const nb of Object.keys(node.next||{}))cands.add(nb);for(const nb of Object.keys(node.prev||{}))cands.add(nb);cands.delete(wl);const out=[];for(const c of cands){if(!this.nodes[c])continue;out.push({word:c,...this.compare(wl,c)});}return out.sort((a,b)=>b.similarity-a.similarity).slice(0,k);}

// ─── Path (Dijkstra) ──────────────────────────────────────────────
path(start,goal,maxLen=15){const sl=start.toLowerCase(),gl=goal.toLowerCase();if(!this.nodes[sl]||!this.nodes[gl])return{ok:false,reason:"unknown word"};const dist={},prev={},visited=new Set();dist[sl]=0;prev[sl]=null;const pq=[sl];while(pq.length){pq.sort((a,b)=>(dist[a]||Infinity)-(dist[b]||Infinity));const u=pq.shift();if(visited.has(u))continue;visited.add(u);if(u===gl)break;const nd=this.nodes[u];if(!nd||!nd.next)continue;const total=Object.values(nd.next).reduce((s,v)=>s+v,0);let pathLen=0,trace=u;while(prev[trace]){pathLen++;trace=prev[trace];}if(pathLen>=maxLen)continue;for(const[nb,cnt]of Object.entries(nd.next)){if(visited.has(nb))continue;const prob=cnt/total;const cost=-Math.log(prob+1e-10);const newDist=dist[u]+cost;if(newDist<(dist[nb]??Infinity)){dist[nb]=newDist;prev[nb]=u;pq.push(nb);}}}if(dist[gl]===undefined)return{ok:false,reason:'no path'};const words=[];let cur=gl;while(cur!==null){words.unshift(cur);cur=prev[cur];}const text=words.join(' ');const score=this.scoreSentence(text);return{ok:true,words,text,energy:dist[gl],coherence:score.coherence,steps:score.steps};}

// ─── Generate ─────────────────────────────────────────────────────
generate(startWord,maxLen=12){let w=startWord?startWord.toLowerCase():null;if(!w||!this.nodes[w]){const cs=Object.entries(this.nodes).filter(([,n])=>n.freq>=3&&n.freq<=50&&Object.keys(n.next).length>=2);if(!cs.length)return{words:[],text:"",coherence:0};w=cs[Math.floor(Math.random()*cs.length)][0];}const words=[w],used=new Set([w]);for(let i=0;i<maxLen-1;i++){const nd=this.nodes[w];if(!nd||!Object.keys(nd.next).length)break;const entries=Object.entries(nd.next).filter(([nw])=>!used.has(nw)||words.length>6);if(!entries.length)break;const weighted=entries.map(([nw,cnt])=>({w:nw,wt:cnt*(1/Math.max(Math.log2((this.nodes[nw]?.freq||1)+1),0.5))}));const totalWt=weighted.reduce((s,x)=>s+x.wt,0);let r=Math.random()*totalWt,pick=weighted[0].w;for(const x of weighted){r-=x.wt;if(r<=0){pick=x.w;break;}}words.push(pick);used.add(pick);w=pick;}const text=words.join(' ');const sc=this.scoreSentence(text);return{words,text,coherence:sc.coherence};}
speak(topic,count=5,maxLen=12){const res=[];const starts=[];const tn=this.nodes[topic?.toLowerCase()];if(tn){starts.push(topic.toLowerCase());const nbrs=Object.entries(tn.neighbors).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([w])=>w);for(const nb of nbrs)if(this.nodes[nb]?.next&&Object.keys(this.nodes[nb].next).length>=2)starts.push(nb);}for(let i=0;i<count;i++)res.push(this.generate(starts.length?starts[i%starts.length]:null,maxLen));return res;}

// ─── Query + Ask ──────────────────────────────────────────────────
query(pattern,k=5){const ph='QBLANK';const raw=pattern.replace(/_/g,ph);const words=tokenize(raw).map(w=>w==='qblank'?'_':w);const bi=words.indexOf('_');if(bi===-1)return{ok:false,reason:"use _ for the blank"};const cands=new Set();for(let i=0;i<words.length;i++){if(i===bi)continue;const nd=this.nodes[words[i]];if(!nd)continue;for(const nb of Object.keys(nd.neighbors))cands.add(nb);for(const nb of Object.keys(nd.next||{}))cands.add(nb);for(const nb of Object.keys(nd.prev||{}))cands.add(nb);}if(bi>0){const pv=this.nodes[words[bi-1]];if(pv)for(const nb of Object.keys(pv.next||{}))cands.add(nb);}if(bi<words.length-1){const nx=this.nodes[words[bi+1]];if(nx)for(const nb of Object.keys(nx.prev||{}))cands.add(nb);}for(const w of words)cands.delete(w);cands.delete('_');const results=[];for(const c of cands){const filled=[...words];filled[bi]=c;const sc=this.scoreSentence(filled.join(' '));results.push({word:c,coherence:sc.coherence,freq:this.nodes[c]?.freq||0,depth:this.depth(c).level});}results.sort((a,b)=>b.coherence-a.coherence);return{ok:true,pattern,blank:bi,results:results.slice(0,k),total:cands.size};}
ask(question){let q=question.toLowerCase().replace(/[?!.]/g,'').trim();q=q.replace(/^(what|which|who)\s+/i,'_ ');q=q.replace(/\s+(what|which|who)\s+/gi,' _ ');if(!q.includes('_')){q=q.replace(/^(does|do|did|can|will|is|are|was|were)\s+/i,'');const ws=q.split(/\s+/);if(ws.length>=2){ws.splice(1,0,'_');q=ws.join(' ');}else return{ok:false,reason:"use _ for the blank"};}return this.query(q);}

// ─── Stats ────────────────────────────────────────────────────────
stats(){const v=Object.keys(this.nodes).length,depths={unborn:0,surface:0,shallow:0,forming:0,structured:0,deep:0};for(const w of Object.keys(this.nodes))depths[this.depth(w).level]++;return{version:VERSION,vocab:v,sentences:this.sentenceCount,tokens:this.tokenCount,depths};}

// ─── Serialize: compact, no arrays ────────────────────────────────
serialize(){return JSON.stringify({version:VERSION,nodes:Object.fromEntries(Object.entries(this.nodes).map(([w,n])=>[w,{c:n.chars,f:n.freq,fs:n.firstSeen,ls:n.lastSeen,p:n.pos,pv:n.posV,lg:n.lastGap,nb:n.neighbors,nx:n.next,px:n.prev,n2:n.next2}])),sc:this.sentenceCount,tc:this.tokenCount});}
static deserialize(json){const d=JSON.parse(json),e=new ShifuEmbryo();e.sentenceCount=d.sc||d.sentenceCount||0;e.tokenCount=d.tc||d.tokenCount||0;for(const[w,n]of Object.entries(d.nodes||d.n||{})){e.nodes[w]={chars:n.c||n.chars||w,freq:n.f||n.freq||0,firstSeen:n.fs||n.firstSeen,lastSeen:n.ls||n.lastSeen,pos:n.p??n.pos??0.5,posV:n.pv??n.posV??1,lastGap:n.lg??n.lastGap??0,neighbors:n.nb||n.neighbors||{},next:n.nx||n.next||{},prev:n.px||n.prev||{},next2:n.n2||n.next2||{}};e._idx(w);}return e;}
}
module.exports={ShifuEmbryo,VERSION,editDistance,ocrDistance,sharedBigrams,tokenize};
