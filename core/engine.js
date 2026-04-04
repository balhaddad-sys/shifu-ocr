// SHIFU v2.0 — The Embryo
// One cell. No dimensions. No channels. Structure emerges from exposure.
// Pressure replaces curiosity: the graph knows its own shape.
const VERSION="2.0.0";
const OCR={"0,o":.1,"1,l":.2,"1,i":.2,"5,s":.3,"8,b":.3,"6,g":.4,"l,i":.2,"m,n":.4,"u,v":.5,"c,e":.5,"r,n":.3,"d,o":.3,"f,t":.4,"h,b":.4,"a,e":.4,"a,o":.4,"u,n":.4,"e,i":.4,"f,l":.4,"s,e":.5,"b,d":.4};
const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
function editDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return editDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++)c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:1)));p=c;}return p[b.length];}
function ocrDistance(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return ocrDistance(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++){const k=[a[i],b[j]].sort().join(",");c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:(OCR[k]??1))));}p=c;}return p[b.length];}
function sharedBigrams(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<2||b.length<2)return 0;const bg=s=>{const m=new Map();for(let i=0;i<s.length-1;i++){const k=s.slice(i,i+2);m.set(k,(m.get(k)||0)+1);}return m;};const ba=bg(a),bb=bg(b);let inter=0;for(const[k,v]of ba)inter+=Math.min(v,bb.get(k)||0);const tot=[...ba.values()].reduce((a,b)=>a+b,0)+[...bb.values()].reduce((a,b)=>a+b,0);return tot?2*inter/tot:0;}
function tokenize(raw){return(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>1);}

class ShifuEmbryo{
constructor(){this.nodes={};this.sentenceCount=0;this.tokenCount=0;this._idxLen={};this._idxBg={};}
_nn(w){return{chars:w,freq:0,firstSeen:null,lastSeen:null,positions:[],gaps:[],sentLengths:[],neighbors:{},next:{},prev:{},next2:{}};}
_idx(w){const l=w.length;(this._idxLen[l]??=[]);if(!this._idxLen[l].includes(w))this._idxLen[l].push(w);for(let i=0;i<w.length-1;i++){const bg=w.slice(i,i+2);(this._idxBg[bg]??=[]);if(!this._idxBg[bg].includes(w))this._idxBg[bg].push(w);}}
// ─── The perturbation: variable depth + phase conflict. ──────────
// Rare words penetrate deeper. Incompatible edges weaken each other.
// When a new neighbor has zero overlap with existing neighbors,
// the connection is "out of phase" — added at half strength,
// and the weakest existing edges decay slightly.
feed(raw){const ws=tokenize(raw);if(ws.length<2)return 0;this.sentenceCount++;const len=ws.length;for(let i=0;i<ws.length;i++){const w=ws[i];const isNew=!(w in this.nodes);const nd=this.nodes[w]??=this._nn(w);if(isNew)this._idx(w);const rp=i/Math.max(ws.length-1,1);this.tokenCount++;nd.freq++;nd.firstSeen??=this.sentenceCount;if(nd.lastSeen!==null)nd.gaps.push(this.sentenceCount-nd.lastSeen);nd.lastSeen=this.sentenceCount;nd.positions.push(rp);nd.sentLengths.push(len);
    const window=Math.min(Math.max(Math.ceil(4/Math.max(Math.log2(nd.freq+1),1)),2),6);
    const myNbrs=Object.keys(nd.neighbors);const hasStructure=myNbrs.length>=5;
    for(let j=Math.max(0,i-window);j<Math.min(ws.length,i+window+1);j++){if(j===i)continue;const nb=ws[j];
      // Phase conflict: does this new neighbor fit my existing structure?
      if(hasStructure&&this.nodes[nb]){const theirNbrs=new Set(Object.keys(this.nodes[nb].neighbors||{}));const overlap=myNbrs.filter(x=>theirNbrs.has(x)).length;const phase=myNbrs.length>0?overlap/myNbrs.length:1;
        if(phase<0.05){
          // Destructive interference: out of phase. Half-strength edge.
          nd.neighbors[nb]=(nd.neighbors[nb]||0)+0.5;
          // Slight decay on weakest existing edges (the conflict pushes back)
          const weakest=Object.entries(nd.neighbors).filter(([k])=>k!==nb).sort((a,b)=>a[1]-b[1]).slice(0,2);
          for(const[k]of weakest){nd.neighbors[k]*=0.95;if(nd.neighbors[k]<0.5)delete nd.neighbors[k];}
        }else{nd.neighbors[nb]=(nd.neighbors[nb]||0)+1;}
      }else{nd.neighbors[nb]=(nd.neighbors[nb]||0)+1;}}
    if(i<ws.length-1)nd.next[ws[i+1]]=(nd.next[ws[i+1]]||0)+1;if(i>0)nd.prev[ws[i-1]]=(nd.prev[ws[i-1]]||0)+1;if(i<ws.length-2){const b=ws[i+1],c=ws[i+2];nd.next2[b]??={};nd.next2[b][c]=(nd.next2[b][c]||0)+1;}if(nd.positions.length>200)nd.positions=nd.positions.slice(-200);if(nd.gaps.length>100)nd.gaps=nd.gaps.slice(-100);if(nd.sentLengths.length>100)nd.sentLengths=nd.sentLengths.slice(-100);}return ws.length;}
feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0;for(const x of s)tk+=this.feed(x);return{sentences:s.length,tokens:tk};}
depth(w){const n=this.nodes[w];if(!n)return{level:"unborn",evidence:0};if(n.freq===1)return{level:"surface",evidence:.1};if(n.freq<5)return{level:"shallow",evidence:.2};const nb=Object.keys(n.neighbors).length,sq=Object.keys(n.next).length+Object.keys(n.prev).length;const ev=Math.min((n.freq/50)*.3+(nb/20)*.3+(sq/10)*.2+(n.positions.length/50)*.2,1);if(ev<.3)return{level:"forming",evidence:ev};if(ev<.7)return{level:"structured",evidence:ev};return{level:"deep",evidence:ev};}
compare(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl],sig={},wt={};sig.editSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);sig.bigramSim=sharedBigrams(al,bl);sig.ocrSim=1-ocrDistance(al,bl)/Math.max(al.length,bl.length,1);wt.char=.05;if(!na||!nb)return{similarity:sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4,signals:sig,weights:wt,totalWeight:.05,depth:"surface"};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),union=new Set([...nbA,...nbBs]);if(union.size>0){const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...union].reduce((s,w)=>s+rw(w),0);sig.neighborOverlap=uw>0?sw/uw:0;wt.neighbor=Math.min(union.size/10,.35);}const totA=Object.values(na.next).reduce((s,v)=>s+v,0),totB=Object.values(nb.next).reduce((s,v)=>s+v,0);if(totA>0||totB>0){sig.expectsAB=totA?(na.next[bl]||0)/totA:0;sig.expectsBA=totB?(nb.next[al]||0)/totB:0;sig.directional=Math.abs(sig.expectsAB-sig.expectsBA);wt.seq=Math.min((totA+totB)/20,.25);}const nx2AB=na.next2?.[bl]?Object.keys(na.next2[bl]).length:0,nx2BA=nb.next2?.[al]?Object.keys(nb.next2[al]).length:0;if(nx2AB>0||nx2BA>0){sig.trajectoryAB=Math.min(nx2AB/5,1);sig.trajectoryBA=Math.min(nx2BA/5,1);wt.traj=.15;}if(na.positions.length>=3&&nb.positions.length>=3){sig.posSim=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*2,1);wt.pos=Math.min(Math.min(na.positions.length,nb.positions.length)/20,.10);}let indAB=0;for(const mid of Object.keys(na.next))if(this.nodes[mid]?.next[bl])indAB++;if(indAB>0){sig.indirectAB=Math.min(indAB/5,1);wt.indirect=Math.min(indAB/10,.15);}let sim=0,tw=0;if(wt.char){sim+=(sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4)*wt.char;tw+=wt.char;}if(wt.neighbor){sim+=sig.neighborOverlap*wt.neighbor;tw+=wt.neighbor;}if(wt.seq){sim+=Math.max(sig.expectsAB||0,sig.expectsBA||0)*wt.seq;tw+=wt.seq;}if(wt.traj){sim+=Math.max(sig.trajectoryAB||0,sig.trajectoryBA||0)*wt.traj;tw+=wt.traj;}if(wt.pos){sim+=sig.posSim*wt.pos;tw+=wt.pos;}if(wt.indirect){sim+=sig.indirectAB*wt.indirect;tw+=wt.indirect;}if(tw>0)sim/=tw;const d=tw>.5?"deep":tw>.3?"structured":tw>.15?"forming":"shallow";return{similarity:sim,signals:sig,weights:wt,totalWeight:tw,depth:d};}
affinity(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{a:al,b:bl,mutual:0,known:false};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1),sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...new Set([...nbA,...nbBs])].reduce((s,w)=>s+rw(w),0),orbit=uw>0?sw/uw:0;const totA=Object.values(na.next||{}).reduce((s,v)=>s+v,0),totB=Object.values(nb.next||{}).reduce((s,v)=>s+v,0),pullAB=totA?(na.next[bl]||0)/totA:0,pullBA=totB?(nb.next[al]||0)/totB:0;let indAB=0,indBA=0;for(const mid of Object.keys(na.next||{}))if(this.nodes[mid]?.next[bl])indAB++;for(const mid of Object.keys(nb.next||{}))if(this.nodes[mid]?.next[al])indBA++;indAB=Math.min(indAB/5,1);indBA=Math.min(indBA/5,1);const charSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);const fwA=new Set(Object.keys(na.next||{})),fwB=new Set(Object.keys(nb.next||{})),fwU=new Set([...fwA,...fwB]),expOvlp=fwU.size?[...fwA].filter(x=>fwB.has(x)).length/fwU.size:0;let posAlign=0;if(na.positions.length>=3&&nb.positions.length>=3)posAlign=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*3,1);const afAB=charSim*.05+orbit*.35+pullAB*.20+indAB*.20+expOvlp*.15+posAlign*.05,afBA=charSim*.05+orbit*.35+pullBA*.20+indBA*.20+expOvlp*.15+posAlign*.05;return{a:al,b:bl,orbit,pullAB,pullBA,indAB,indBA,charSim,expOvlp,posAlign,afAB,afBA,mutual:(afAB+afBA)/2,asym:Math.abs(afAB-afBA),known:true};}
scoreSentence(raw){const ws=tokenize(raw);if(ws.length<2)return{words:ws,steps:[],meanSurprise:0,coherence:0};const steps=[];let total=0;const field=new Map();for(let i=0;i<ws.length;i++){const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node};let sig=0,wts=0;if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*.35;wts+=.35;}}if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*.30;wts+=.30;}}if(i>0&&field.size>0){const fw=field.get(w)||0,mx=Math.max(...field.values(),1);step.fieldS=1-fw/mx;sig+=step.fieldS*.35;wts+=.35;}step.afGate=0;if(i>0&&node){const pn=this.nodes[ws[i-1]];if(pn?.neighbors&&node.neighbors){const pN=Object.keys(pn.neighbors),wN=new Set(Object.keys(node.neighbors)),sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;step.afGate=un?sh/un:0;}}step.surprise=wts>0?(sig/wts)*(1-step.afGate*.3):(node?0.5:1);total+=step.surprise;step.cumS=total;steps.push(step);if(node?.neighbors)for(const[nb,cnt]of Object.entries(node.neighbors))field.set(nb,(field.get(nb)||0)+cnt);field.set(w,(field.get(w)||0)+10);}const ms=steps.length?total/steps.length:0;return{words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)};}
// ─── Pressure: LOCAL. O(neighbors), not O(V²). ─────────────────
// A molecule feels its neighborhood, not the universe.
// Inbound = sum of prev table (already stored). No full scan.
pressureOf(w){const node=this.nodes[w];if(!node)return null;const inbound=Object.values(node.prev||{}).reduce((s,v)=>s+v,0);const actual=Object.keys(node.neighbors).length+Object.keys(node.next).length+Object.keys(node.prev).length;const p=actual-inbound;const nbrs=Object.keys(node.neighbors);let internal=0,pairs=0;for(let i=0;i<Math.min(nbrs.length,15);i++)for(let j=i+1;j<Math.min(nbrs.length,15);j++){pairs++;if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]])internal++;}const closure=pairs>0?internal/pairs:1;return{word:w,pressure:p,inbound,actual,closure,freq:node.freq,depth:this.depth(w).level};}
pressure(minFreq=2){const map=[];for(const[w,n]of Object.entries(this.nodes)){if(n.freq<minFreq)continue;map.push(this.pressureOf(w));}return map.sort((a,b)=>a.pressure-b.pressure);}
vacuums(k=10){return this.pressure().filter(p=>p.pressure<0).slice(0,k);}
surpluses(k=10){return this.pressure().filter(p=>p.pressure>0).sort((a,b)=>b.pressure-a.pressure).slice(0,k);}
bridges(k=10){return this.pressure().filter(p=>p.closure<.3&&p.freq>=3).slice(0,k);}
// ─── Tensions: words with split neighborhoods (phase conflict) ────
// Finds words where neighbors form 2+ disconnected clusters.
// Filters out high-frequency function words that connect everything.
tensions(k=10){let maxFreq=0;for(const n of Object.values(this.nodes))if(n.freq>maxFreq)maxFreq=n.freq;const freqThresh=maxFreq*0.3;
  const results=[];for(const[w,nd]of Object.entries(this.nodes)){if(nd.freq<3)continue;
  const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<freqThresh);
  if(nbrs.length<4||nbrs.length>80)continue; // skip huge hubs — too expensive, too generic
  const adj={};for(const n of nbrs)adj[n]=[];
  for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}
  const visited=new Set();const components=[];
  for(const n of nbrs){if(visited.has(n))continue;const comp=[n];const queue=[n];visited.add(n);while(queue.length){const cur=queue.shift();for(const next of(adj[cur]||[])){if(!visited.has(next)){visited.add(next);queue.push(next);comp.push(next);}}}components.push(comp);}
  const real=components.filter(c=>c.length>=2);
  if(real.length>=2){
    const tension=1-real[0].length/nbrs.length;
    results.push({word:w,components:real.length,sizes:real.map(c=>c.length),clusters:real.map(c=>c.slice(0,4)),tension,freq:nd.freq});
  }}
  return results.sort((a,b)=>b.tension-a.tension).slice(0,k);}
// ─── Collapse: force resolution of ambiguous word ─────────────────
// Keeps the dominant cluster, prunes the minority.
collapse(w){w=w.toLowerCase();const nd=this.nodes[w];if(!nd)return{ok:false,reason:"unknown"};
  let maxFreq=0;for(const n of Object.values(this.nodes))if(n.freq>maxFreq)maxFreq=n.freq;const freqThresh=maxFreq*0.3;
  const nbrs=Object.keys(nd.neighbors).filter(n=>this.nodes[n]&&this.nodes[n].freq<freqThresh);
  if(nbrs.length<4)return{ok:false,reason:"not enough content neighbors"};
  const adj={};for(const n of nbrs)adj[n]=[];
  for(let i=0;i<nbrs.length;i++)for(let j=i+1;j<nbrs.length;j++){if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]]){adj[nbrs[i]].push(nbrs[j]);adj[nbrs[j]].push(nbrs[i]);}}
  const visited=new Set();const components=[];
  for(const n of nbrs){if(visited.has(n))continue;const comp=[n];const queue=[n];visited.add(n);while(queue.length){const cur=queue.shift();for(const next of(adj[cur]||[])){if(!visited.has(next)){visited.add(next);queue.push(next);comp.push(next);}}}components.push(comp);}
  const real=components.filter(c=>c.length>=2);
  if(real.length<2)return{ok:false,reason:"no split detected"};
  real.sort((a,b)=>b.length-a.length);const keep=new Set(real[0]);let pruned=0;
  for(const nb of nbrs){if(!keep.has(nb)){delete nd.neighbors[nb];delete nd.next[nb];delete nd.prev[nb];pruned++;}}
  return{ok:true,word:w,kept:real[0].length,pruned,surviving:real[0].slice(0,5),removed:real.slice(1).map(c=>c.slice(0,3))};}

// ─── Active forgetting: weaken or sever connections ───────────────
// unlearn(a, b): told these don't belong together. Weaken all edges.
// forget(word): remove a node entirely. As if it was never seen.
// decay(threshold): passive forgetting. Remove all edges below threshold.
unlearn(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{changed:false,reason:"unknown word"};let removed=0;
  // Weaken co-occurrence (halve, delete if ≤1)
  if(na.neighbors[bl]){na.neighbors[bl]=Math.floor(na.neighbors[bl]/2);if(na.neighbors[bl]<=0){delete na.neighbors[bl];removed++;}else removed++;}
  if(nb.neighbors[al]){nb.neighbors[al]=Math.floor(nb.neighbors[al]/2);if(nb.neighbors[al]<=0){delete nb.neighbors[al];removed++;}else removed++;}
  // Weaken sequential (halve, delete if ≤1)
  if(na.next[bl]){na.next[bl]=Math.floor(na.next[bl]/2);if(na.next[bl]<=0)delete na.next[bl];removed++;}
  if(nb.next[al]){nb.next[al]=Math.floor(nb.next[al]/2);if(nb.next[al]<=0)delete nb.next[al];removed++;}
  if(na.prev[bl]){na.prev[bl]=Math.floor(na.prev[bl]/2);if(na.prev[bl]<=0)delete na.prev[bl];removed++;}
  if(nb.prev[al]){nb.prev[al]=Math.floor(nb.prev[al]/2);if(nb.prev[al]<=0)delete nb.prev[al];removed++;}
  // Weaken second-order (delete paths through the other)
  if(na.next2[bl])delete na.next2[bl];
  if(nb.next2[al])delete nb.next2[al];
  // Clean nx2 entries that point to the other as a destination
  for(const mid of Object.keys(na.next2||{})){if(na.next2[mid][bl]){delete na.next2[mid][bl];if(!Object.keys(na.next2[mid]).length)delete na.next2[mid];}}
  for(const mid of Object.keys(nb.next2||{})){if(nb.next2[mid][al]){delete nb.next2[mid][al];if(!Object.keys(nb.next2[mid]).length)delete nb.next2[mid];}}
  return{changed:removed>0,edgesWeakened:removed,a:al,b:bl};}
forget(word){const w=word.toLowerCase();if(!this.nodes[w])return{changed:false,reason:"unknown word"};
  // Remove all references to this word from every other node
  for(const[,other]of Object.entries(this.nodes)){
    delete other.neighbors[w];delete other.next[w];delete other.prev[w];
    delete other.next2[w];
    for(const mid of Object.keys(other.next2||{})){if(other.next2[mid][w]){delete other.next2[mid][w];if(!Object.keys(other.next2[mid]).length)delete other.next2[mid];}}}
  delete this.nodes[w];return{changed:true,word:w};}
decay(threshold=1){let removed=0;
  for(const[,node]of Object.entries(this.nodes)){
    for(const[nb,cnt]of Object.entries(node.neighbors)){if(cnt<=threshold){delete node.neighbors[nb];removed++;}}
    for(const[nb,cnt]of Object.entries(node.next)){if(cnt<=threshold){delete node.next[nb];removed++;}}
    for(const[nb,cnt]of Object.entries(node.prev)){if(cnt<=threshold){delete node.prev[nb];removed++;}}
    for(const mid of Object.keys(node.next2||{})){for(const[dest,cnt]of Object.entries(node.next2[mid])){if(cnt<=threshold){delete node.next2[mid][dest];removed++;}}if(!Object.keys(node.next2[mid]).length)delete node.next2[mid];}}
  return{edgesRemoved:removed};}
// ─── Correct: indexed candidate lookup, not full scan. ────────────
correct(garbled,k=5){const g=garbled.toLowerCase(),cands=new Set();for(let d=-2;d<=2;d++){const ws=this._idxLen[g.length+d];if(ws)for(const w of ws)cands.add(w);}for(let i=0;i<g.length-1;i++){const ws=this._idxBg[g.slice(i,i+2)];if(ws)for(const w of ws)cands.add(w);}const scored=[...cands].map(w=>{const o=1-ocrDistance(g,w)/Math.max(g.length,w.length,1),b=sharedBigrams(g,w);return{word:w,score:o*.7+b*.3};});scored.sort((a,b)=>b.score-a.score||(this.nodes[b.word]?.freq||0)-(this.nodes[a.word]?.freq||0));const top=scored.slice(0,k),conf=top.length>=2?top[0].score-top[1].score:top.length?1:0;return{candidates:top,confidence:conf};}
// ─── Similar: 2-hop neighborhood, not universe scan. O(nbrs²) not O(V). ──
similar(w,k=8){const wl=w.toLowerCase(),node=this.nodes[wl];if(!node)return[];const cands=new Set();for(const nb of Object.keys(node.neighbors)){cands.add(nb);const nbNode=this.nodes[nb];if(nbNode)for(const nb2 of Object.keys(nbNode.neighbors))cands.add(nb2);}for(const nb of Object.keys(node.next||{}))cands.add(nb);for(const nb of Object.keys(node.prev||{}))cands.add(nb);cands.delete(wl);const out=[];for(const c of cands){if(!this.nodes[c])continue;out.push({word:c,...this.compare(wl,c)});}return out.sort((a,b)=>b.similarity-a.similarity).slice(0,k);}
// ─── Path: least-action sentence generation (Dijkstra) ───────────
// Cost of edge A→B = -log(P(B|A)) * resistance(B).
// Each molecule has its own energy barrier to disturb.
// Deep words = heavy molecules = high resistance = expensive to route through.
// Rare words = light molecules = low resistance = cheap detours.
// resistance(B) = 1 / log2(freq+2). Logarithmic because energy scales
// with structure, not linearly with exposure count.
path(start,goal,maxLen=15){
  const sl=start.toLowerCase(),gl=goal.toLowerCase();
  if(!this.nodes[sl]||!this.nodes[gl])return{ok:false,reason:"unknown word"};
  const dist={},prev={},visited=new Set();
  dist[sl]=0;prev[sl]=null;
  const pq=[sl];
  while(pq.length){
    pq.sort((a,b)=>(dist[a]||Infinity)-(dist[b]||Infinity));
    const u=pq.shift();
    if(visited.has(u))continue;visited.add(u);
    if(u===gl)break;
    const nd=this.nodes[u];if(!nd||!nd.next)continue;
    const total=Object.values(nd.next).reduce((s,v)=>s+v,0);
    let pathLen=0;let trace=u;while(prev[trace]){pathLen++;trace=prev[trace];}
    if(pathLen>=maxLen)continue;
    for(const[nb,cnt]of Object.entries(nd.next)){
      if(visited.has(nb))continue;
      const prob=cnt/total;
      const surprise=-Math.log(prob+1e-10);
      // Each node's resistance: how much energy it takes to disturb this molecule
      const nbFreq=this.nodes[nb]?.freq||1;
      const resistance=1/Math.log2(nbFreq+2);
      const cost=surprise*resistance;
      const newDist=dist[u]+cost;
      if(newDist<(dist[nb]??Infinity)){dist[nb]=newDist;prev[nb]=u;pq.push(nb);}
    }
  }
  // Reconstruct path
  if(dist[gl]===undefined)return{ok:false,reason:`no path from "${start}" to "${goal}"`};
  const words=[];let cur=gl;
  while(cur!==null){words.unshift(cur);cur=prev[cur];}
  const text=words.join(' ');
  const score=this.scoreSentence(text);
  return{ok:true,words,text,energy:dist[gl],coherence:score.coherence,steps:score.steps};}
// ─── Generate: open-ended walk (weighted random, prefers rare) ────
generate(startWord,maxLen=12){
  let w=startWord?startWord.toLowerCase():null;
  if(!w||!this.nodes[w]){const cs=Object.entries(this.nodes).filter(([,n])=>n.freq>=3&&n.freq<=50&&Object.keys(n.next).length>=2);if(!cs.length)return{words:[],text:"(not enough data)"};w=cs[Math.floor(Math.random()*cs.length)][0];}
  const words=[w],used=new Set([w]);
  for(let i=0;i<maxLen-1;i++){const nd=this.nodes[w];if(!nd||!Object.keys(nd.next).length)break;
    const entries=Object.entries(nd.next).filter(([nw])=>!used.has(nw)||words.length>6);if(!entries.length)break;
    const weighted=entries.map(([nw,cnt])=>({w:nw,wt:cnt*(1/Math.max(Math.log2((this.nodes[nw]?.freq||1)+1),0.5))}));
    const totalWt=weighted.reduce((s,x)=>s+x.wt,0);let r=Math.random()*totalWt,pick=weighted[0].w;
    for(const x of weighted){r-=x.wt;if(r<=0){pick=x.w;break;}}
    words.push(pick);used.add(pick);w=pick;}
  const text=words.join(' ');const score=this.scoreSentence(text);
  return{words,text,coherence:score.coherence};}
// ─── Query: fill-in-the-blank from graph structure ───────────────
// "doctor prescribed _ for seizure" → finds best word for the blank.
// Combines: forward expectation (word before → next table),
//           backward expectation (word after → prev table),
//           trajectory (word 2 before → next2 table),
//           neighborhood support (all context words' neighbors vote).
// No neural network. Just the edges that exist.
query(raw,k=5){
  // Tokenize but preserve _ as a placeholder
  const ws=raw.toLowerCase().replace(/_+/g,' BLANK ').match(/[a-z0-9]+|BLANK/g)||[];
  const blankIdx=ws.indexOf('BLANK');if(blankIdx===-1)return{ok:false,reason:"use _ for the blank"};
  const before=blankIdx>0?ws[blankIdx-1]:null;
  const after=blankIdx<ws.length-1?ws[blankIdx+1]:null;
  const before2=blankIdx>1?ws[blankIdx-2]:null;
  const context=ws.filter(w=>w!=='BLANK');
  const scores={};
  const addScore=(w,val,src)=>{if(w==='BLANK'||context.includes(w))return;scores[w]??={word:w,total:0,signals:{}};scores[w].total+=val;scores[w].signals[src]=(scores[w].signals[src]||0)+val;};
  // Signal 1: forward expectation — what does the word before predict?
  if(before&&this.nodes[before]?.next){const nd=this.nodes[before];const total=Object.values(nd.next).reduce((s,v)=>s+v,0);
    if(total>0)for(const[w,cnt]of Object.entries(nd.next))addScore(w,(cnt/total)*0.35,'fwd');}
  // Signal 2: backward expectation — what does the word after expect before it?
  if(after&&this.nodes[after]?.prev){const nd=this.nodes[after];const total=Object.values(nd.prev).reduce((s,v)=>s+v,0);
    if(total>0)for(const[w,cnt]of Object.entries(nd.prev))addScore(w,(cnt/total)*0.25,'bwd');}
  // Signal 3: trajectory — does word_before2 → word_before → ? predict anything?
  if(before2&&before&&this.nodes[before2]?.next2?.[before]){const nx2=this.nodes[before2].next2[before];const total=Object.values(nx2).reduce((s,v)=>s+v,0);
    if(total>0)for(const[w,cnt]of Object.entries(nx2))addScore(w,(cnt/total)*0.20,'traj');}
  // Signal 4: neighborhood support — do context words' neighbors vote for any candidate?
  const nbrVotes={};for(const cw of context){const nd=this.nodes[cw];if(!nd?.neighbors)continue;
    for(const[nb,cnt]of Object.entries(nd.neighbors)){if(context.includes(nb)||nb==='BLANK')continue;nbrVotes[nb]=(nbrVotes[nb]||0)+cnt;}}
  const maxVote=Math.max(...Object.values(nbrVotes),1);
  for(const[w,v]of Object.entries(nbrVotes))addScore(w,(v/maxVote)*0.20,'nbr');
  const ranked=Object.values(scores).sort((a,b)=>b.total-a.total).slice(0,k);
  if(!ranked.length)return{ok:false,reason:"not enough data to fill blank"};
  const filled=ws.map(w=>w==='BLANK'?ranked[0].word:w).join(' ');
  const score=this.scoreSentence(filled);
  return{ok:true,blank:blankIdx,candidates:ranked,filled,coherence:score.coherence};}
stats(){const v=Object.keys(this.nodes).length,depths={unborn:0,surface:0,shallow:0,forming:0,structured:0,deep:0};for(const w of Object.keys(this.nodes))depths[this.depth(w).level]++;return{version:VERSION,vocab:v,sentences:this.sentenceCount,tokens:this.tokenCount,depths};}
serialize(){return JSON.stringify({version:VERSION,nodes:Object.fromEntries(Object.entries(this.nodes).map(([w,n])=>[w,{chars:n.chars,freq:n.freq,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions.slice(-200),gaps:n.gaps.slice(-100),sentLengths:n.sentLengths.slice(-100),neighbors:n.neighbors,next:n.next,prev:n.prev,next2:n.next2}])),sentenceCount:this.sentenceCount,tokenCount:this.tokenCount});}
static deserialize(json){const d=JSON.parse(json),e=new ShifuEmbryo();e.sentenceCount=d.sentenceCount||0;e.tokenCount=d.tokenCount||0;for(const[w,n]of Object.entries(d.nodes||{})){e.nodes[w]={chars:n.chars||w,freq:n.freq||0,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions||[],gaps:n.gaps||[],sentLengths:n.sentLengths||[],neighbors:n.neighbors||{},next:n.next||{},prev:n.prev||{},next2:n.next2||{}};e._idx(w);}return e;}}
module.exports={ShifuEmbryo,VERSION,editDistance,ocrDistance,sharedBigrams,tokenize};
