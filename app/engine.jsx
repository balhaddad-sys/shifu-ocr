import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════
// SHIFU v2.0 — The Embryo Engine (embedded)
// No dimensions. No channels. Structure emerges from exposure.
// Pressure: the graph knows its own shape.
// ═══════════════════════════════════════════════════════════════
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
depth(w){const n=this.nodes[w];if(!n)return{level:"unborn",evidence:0};if(n.freq===1)return{level:"surface",evidence:.1};if(n.freq<5)return{level:"shallow",evidence:.2};const nb=Object.keys(n.neighbors).length,sq=Object.keys(n.next).length+Object.keys(n.prev).length;const ev=Math.min((n.freq/50)*.3+(nb/20)*.3+(sq/10)*.2+(n.positions.length/50)*.2,1);if(ev<.3)return{level:"forming",evidence:ev};if(ev<.7)return{level:"structured",evidence:ev};return{level:"deep",evidence:ev};}
compare(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl],sig={},wt={};sig.editSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);sig.bigramSim=sharedBigrams(al,bl);sig.ocrSim=1-ocrDistance(al,bl)/Math.max(al.length,bl.length,1);wt.char=.05;if(!na||!nb)return{similarity:sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4,signals:sig,weights:wt,totalWeight:.05,depth:"surface"};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),union=new Set([...nbA,...nbBs]);if(union.size>0){const rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1);const sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...union].reduce((s,w)=>s+rw(w),0);sig.neighborOverlap=uw>0?sw/uw:0;wt.neighbor=Math.min(union.size/10,.35);}const totA=Object.values(na.next).reduce((s,v)=>s+v,0),totB=Object.values(nb.next).reduce((s,v)=>s+v,0);if(totA>0||totB>0){sig.expectsAB=totA?(na.next[bl]||0)/totA:0;sig.expectsBA=totB?(nb.next[al]||0)/totB:0;sig.directional=Math.abs(sig.expectsAB-sig.expectsBA);wt.seq=Math.min((totA+totB)/20,.25);}const nx2AB=na.next2?.[bl]?Object.keys(na.next2[bl]).length:0,nx2BA=nb.next2?.[al]?Object.keys(nb.next2[al]).length:0;if(nx2AB>0||nx2BA>0){sig.trajectoryAB=Math.min(nx2AB/5,1);sig.trajectoryBA=Math.min(nx2BA/5,1);wt.traj=.15;}if(na.positions.length>=3&&nb.positions.length>=3){sig.posSim=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*2,1);wt.pos=Math.min(Math.min(na.positions.length,nb.positions.length)/20,.10);}let indAB=0;for(const mid of Object.keys(na.next))if(this.nodes[mid]?.next[bl])indAB++;if(indAB>0){sig.indirectAB=Math.min(indAB/5,1);wt.indirect=Math.min(indAB/10,.15);}let sim=0,tw=0;if(wt.char){sim+=(sig.editSim*.3+sig.bigramSim*.3+sig.ocrSim*.4)*wt.char;tw+=wt.char;}if(wt.neighbor){sim+=sig.neighborOverlap*wt.neighbor;tw+=wt.neighbor;}if(wt.seq){sim+=Math.max(sig.expectsAB||0,sig.expectsBA||0)*wt.seq;tw+=wt.seq;}if(wt.traj){sim+=Math.max(sig.trajectoryAB||0,sig.trajectoryBA||0)*wt.traj;tw+=wt.traj;}if(wt.pos){sim+=sig.posSim*wt.pos;tw+=wt.pos;}if(wt.indirect){sim+=sig.indirectAB*wt.indirect;tw+=wt.indirect;}if(tw>0)sim/=tw;const d=tw>.5?"deep":tw>.3?"structured":tw>.15?"forming":"shallow";return{similarity:sim,signals:sig,weights:wt,totalWeight:tw,depth:d};}
affinity(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{a:al,b:bl,mutual:0,known:false};const nbA=Object.keys(na.neighbors),nbBs=new Set(Object.keys(nb.neighbors)),shared=nbA.filter(x=>nbBs.has(x)),rw=w=>1/Math.max(Math.log2((this.nodes[w]?.freq||0)+1),1),sw=shared.reduce((s,w)=>s+rw(w),0),uw=[...new Set([...nbA,...nbBs])].reduce((s,w)=>s+rw(w),0),orbit=uw>0?sw/uw:0;const totA=Object.values(na.next||{}).reduce((s,v)=>s+v,0),totB=Object.values(nb.next||{}).reduce((s,v)=>s+v,0),pullAB=totA?(na.next[bl]||0)/totA:0,pullBA=totB?(nb.next[al]||0)/totB:0;let indAB=0,indBA=0;for(const mid of Object.keys(na.next||{}))if(this.nodes[mid]?.next[bl])indAB++;for(const mid of Object.keys(nb.next||{}))if(this.nodes[mid]?.next[al])indBA++;indAB=Math.min(indAB/5,1);indBA=Math.min(indBA/5,1);const charSim=1-editDistance(al,bl)/Math.max(al.length,bl.length,1);const fwA=new Set(Object.keys(na.next||{})),fwB=new Set(Object.keys(nb.next||{})),fwU=new Set([...fwA,...fwB]),expOvlp=fwU.size?[...fwA].filter(x=>fwB.has(x)).length/fwU.size:0;let posAlign=0;if(na.positions.length>=3&&nb.positions.length>=3)posAlign=1-Math.min(Math.abs(mean(na.positions)-mean(nb.positions))*3,1);const afAB=charSim*.05+orbit*.35+pullAB*.20+indAB*.20+expOvlp*.15+posAlign*.05,afBA=charSim*.05+orbit*.35+pullBA*.20+indBA*.20+expOvlp*.15+posAlign*.05;return{a:al,b:bl,orbit,pullAB,pullBA,indAB,indBA,charSim,expOvlp,posAlign,afAB,afBA,mutual:(afAB+afBA)/2,asym:Math.abs(afAB-afBA),known:true};}
scoreSentence(raw){const ws=tokenize(raw);if(ws.length<2)return{words:ws,steps:[],meanSurprise:0,coherence:0};const steps=[];let total=0;const field=new Map();for(let i=0;i<ws.length;i++){const w=ws[i],node=this.nodes[w],step={word:w,pos:i,known:!!node};let sig=0,wts=0;if(i>0){const prev=this.nodes[ws[i-1]];if(prev?.next){const tot=Object.values(prev.next).reduce((a,b)=>a+b,0);step.seqS=1-((prev.next[w]||0)/Math.max(tot,1));sig+=step.seqS*.35;wts+=.35;}}if(i>=2){const pp=this.nodes[ws[i-2]],nx2=pp?.next2?.[ws[i-1]];if(nx2){const tot=Object.values(nx2).reduce((a,b)=>a+b,0);step.trajS=1-((nx2[w]||0)/Math.max(tot,1));sig+=step.trajS*.30;wts+=.30;}}if(i>0&&field.size>0){const fw=field.get(w)||0,mx=Math.max(...field.values(),1);step.fieldS=1-fw/mx;sig+=step.fieldS*.35;wts+=.35;}step.afGate=0;if(i>0&&node){const pn=this.nodes[ws[i-1]];if(pn?.neighbors&&node.neighbors){const pN=Object.keys(pn.neighbors),wN=new Set(Object.keys(node.neighbors)),sh=pN.filter(x=>wN.has(x)).length,un=new Set([...pN,...wN]).size;step.afGate=un?sh/un:0;}}step.surprise=wts>0?(sig/wts)*(1-step.afGate*.3):(node?.5:1);total+=step.surprise;step.cumS=total;steps.push(step);if(node?.neighbors)for(const[nb,cnt]of Object.entries(node.neighbors))field.set(nb,(field.get(nb)||0)+cnt);field.set(w,(field.get(w)||0)+10);}const ms=steps.length?total/steps.length:0;return{words:ws,steps,meanSurprise:ms,coherence:1-Math.min(ms,1)};}
pressure(){const map=[];for(const[word,node]of Object.entries(this.nodes)){let inbound=0;for(const[,other]of Object.entries(this.nodes))if(other.next[word])inbound+=other.next[word];const actual=Object.keys(node.neighbors).length+Object.keys(node.next).length+Object.keys(node.prev).length;const p=actual-inbound;const nbrs=Object.keys(node.neighbors);let internal=0,pairs=0;for(let i=0;i<Math.min(nbrs.length,15);i++)for(let j=i+1;j<Math.min(nbrs.length,15);j++){pairs++;if(this.nodes[nbrs[i]]?.neighbors[nbrs[j]])internal++;}const closure=pairs>0?internal/pairs:1;map.push({word,pressure:p,inbound,actual,closure,freq:node.freq,depth:this.depth(word).level});}return map.sort((a,b)=>a.pressure-b.pressure);}
vacuums(k=10){return this.pressure().filter(p=>p.pressure<0).slice(0,k);}
surpluses(k=10){return this.pressure().filter(p=>p.pressure>0).sort((a,b)=>b.pressure-a.pressure).slice(0,k);}
bridges(k=10){return this.pressure().filter(p=>p.closure<.3&&p.freq>=3).slice(0,k);}
generate(seed,maxLen=20){const s=seed.toLowerCase();const nd=this.nodes[s];if(!nd)return{words:[],text:"",reason:"unknown word"};const words=[s];let cur=s,prev=null;for(let i=0;i<maxLen-1;i++){const n=this.nodes[cur];if(!n)break;let pool=null;const prevNode=prev?this.nodes[prev]:null;if(prevNode?.next2?.[cur])pool=Object.entries(prevNode.next2[cur]);if(!pool||!pool.length)pool=Object.entries(n.next||{});if(!pool.length)pool=Object.entries(n.neighbors||{});if(!pool.length)break;const tot=pool.reduce((s,[,w])=>s+w,0);let r=Math.random()*tot,picked=pool[0][0];for(const[w,wt]of pool){r-=wt;if(r<=0){picked=w;break;}}if(words.length>=3&&words.slice(-3).includes(picked))break;if(words.length>=2&&picked===words[words.length-2])break;words.push(picked);prev=cur;cur=picked;}return{words,text:words.join(" ")};}
decay(threshold=1){let removed=0;for(const[,node]of Object.entries(this.nodes)){for(const[nb,cnt]of Object.entries(node.neighbors)){if(cnt<=threshold){delete node.neighbors[nb];removed++;}}for(const[nb,cnt]of Object.entries(node.next)){if(cnt<=threshold){delete node.next[nb];removed++;}}for(const[nb,cnt]of Object.entries(node.prev)){if(cnt<=threshold){delete node.prev[nb];removed++;}}for(const mid of Object.keys(node.next2||{})){for(const[dest,cnt]of Object.entries(node.next2[mid])){if(cnt<=threshold){delete node.next2[mid][dest];removed++;}}if(!Object.keys(node.next2[mid]).length)delete node.next2[mid];}}return{edgesRemoved:removed};}
describe(a,b){const al=a.toLowerCase(),bl=b.toLowerCase(),na=this.nodes[al],nb=this.nodes[bl];if(!na||!nb)return{a:al,b:bl,description:null,candidates:[],reason:"unknown word"};const bridges=new Map();for(const[mid,w]of Object.entries(na.next||{})){if(this.nodes[mid]?.next[bl])bridges.set(mid,(bridges.get(mid)||0)+w+this.nodes[mid].next[bl]);}for(const[mid,w]of Object.entries(nb.next||{})){if(this.nodes[mid]?.next[al])bridges.set(mid,(bridges.get(mid)||0)+w+this.nodes[mid].next[al]);}for(const[mid,targets]of Object.entries(na.next2||{})){if(targets[bl])bridges.set(mid,(bridges.get(mid)||0)+targets[bl]*3);}for(const[mid,targets]of Object.entries(nb.next2||{})){if(targets[al])bridges.set(mid,(bridges.get(mid)||0)+targets[al]*3);}const shared=[];const nbA=na.neighbors||{},nbB=nb.neighbors||{};for(const w of Object.keys(nbA)){if(nbB[w]){const freq=this.nodes[w]?.freq||1;shared.push({word:w,weight:(nbA[w]+nbB[w])/Math.log2(freq+2)});}}shared.sort((a,b)=>b.weight-a.weight);const nxA=Object.entries(na.next||{}).sort((a,b)=>b[1]-a[1]).slice(0,10).map(e=>e[0]);const nxB=Object.entries(nb.next||{}).sort((a,b)=>b[1]-a[1]).slice(0,10).map(e=>e[0]);const pxA=Object.entries(na.prev||{}).sort((a,b)=>b[1]-a[1]).slice(0,10).map(e=>e[0]);const pxB=Object.entries(nb.prev||{}).sort((a,b)=>b[1]-a[1]).slice(0,10).map(e=>e[0]);const candidates=new Set();const topBridges=[...bridges.entries()].sort((a,b)=>b[1]-a[1]).slice(0,8);for(const[mid]of topBridges){candidates.add(`${al} ${mid} ${bl}`);candidates.add(`${bl} ${mid} ${al}`);}for(const s of shared.slice(0,6)){candidates.add(`${al} and ${bl} ${s.word}`);candidates.add(`${al} ${s.word} ${bl}`);}for(const wa of nxA.slice(0,4)){candidates.add(`${al} ${wa} ${bl}`);for(const wb of nxB.slice(0,3)){if(wa!==wb)candidates.add(`${al} ${wa} ${bl} ${wb}`);}}for(const pa of pxA.slice(0,3)){for(const pb of pxB.slice(0,3)){if(pa===pb)candidates.add(`${pa} ${al} and ${bl}`);}}const scored=[];for(const sentence of candidates){const result=this.scoreSentence(sentence);if(result.words.length>=2)scored.push({sentence,coherence:result.coherence,surprise:result.meanSurprise});}scored.sort((a,b)=>b.coherence-a.coherence);return{a:al,b:bl,description:scored[0]?.sentence||null,coherence:scored[0]?.coherence||0,candidates:scored.slice(0,5),bridges:topBridges.map(([w,s])=>({word:w,strength:s})),shared:shared.slice(0,8).map(s=>({word:s.word,weight:+s.weight.toFixed(2)}))};}
correct(garbled,k=5){const g=garbled.toLowerCase(),cands=new Set();for(const w of Object.keys(this.nodes))if(Math.abs(w.length-g.length)<=3)cands.add(w);for(let i=0;i<g.length-1;i++){const bg=g.slice(i,i+2);for(const w of Object.keys(this.nodes))if(w.includes(bg))cands.add(w);}const scored=[...cands].map(w=>{const o=1-ocrDistance(g,w)/Math.max(g.length,w.length,1),b=sharedBigrams(g,w);return{word:w,score:o*.7+b*.3};});scored.sort((a,b)=>b.score-a.score||(this.nodes[b.word]?.freq||0)-(this.nodes[a.word]?.freq||0));const top=scored.slice(0,k),conf=top.length>=2?top[0].score-top[1].score:top.length?1:0;return{candidates:top,confidence:conf};}
similar(w,k=8){const wl=w.toLowerCase(),out=[];for(const c of Object.keys(this.nodes)){if(c===wl)continue;out.push({word:c,...this.compare(wl,c)});}return out.sort((a,b)=>b.similarity-a.similarity).slice(0,k);}
stats(){const v=Object.keys(this.nodes).length,depths={unborn:0,surface:0,shallow:0,forming:0,structured:0,deep:0};for(const w of Object.keys(this.nodes))depths[this.depth(w).level]++;return{version:VERSION,vocab:v,sentences:this.sentenceCount,tokens:this.tokenCount,depths};}
serialize(){return JSON.stringify({version:VERSION,nodes:Object.fromEntries(Object.entries(this.nodes).map(([w,n])=>[w,{chars:n.chars,freq:n.freq,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions.slice(-200),gaps:n.gaps.slice(-100),sentLengths:n.sentLengths.slice(-100),neighbors:n.neighbors,next:n.next,prev:n.prev,next2:n.next2}])),sentenceCount:this.sentenceCount,tokenCount:this.tokenCount});}
static deserialize(json){const d=JSON.parse(json),e=new ShifuEmbryo();e.sentenceCount=d.sentenceCount||0;e.tokenCount=d.tokenCount||0;for(const[w,n]of Object.entries(d.nodes||{})){e.nodes[w]={chars:n.chars||w,freq:n.freq||0,firstSeen:n.firstSeen,lastSeen:n.lastSeen,positions:n.positions||[],gaps:n.gaps||[],sentLengths:n.sentLengths||[],neighbors:n.neighbors||{},next:n.next||{},prev:n.prev||{},next2:n.next2||{}};}return e;}}

// ═══════════════════════════════════════════════════════════════
// IndexedDB
// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════
// Teacher — closes the feedback loop
// ═══════════════════════════════════════════════════════════════
class Teacher{
constructor(engine){this.engine=engine;this.log=[];this.corrections=0;this.lessons=0;}
diagnose(){const vacs=this.engine.vacuums(10),surps=this.engine.surpluses(10),brs=this.engine.bridges(10),st=this.engine.stats();const underexposed=[];for(const[w,n]of Object.entries(this.engine.nodes)){const d=this.engine.depth(w);if(n.freq>=3&&d.level==="shallow")underexposed.push({word:w,freq:n.freq,evidence:d.evidence});}underexposed.sort((a,b)=>b.freq-a.freq);const starved=[];for(const[w,n]of Object.entries(this.engine.nodes)){let inbound=0;for(const[,other]of Object.entries(this.engine.nodes))if(other.next[w])inbound+=other.next[w];const actual=Object.keys(n.neighbors).length;if(inbound>actual&&inbound>=3)starved.push({word:w,inbound,actual,deficit:inbound-actual});}starved.sort((a,b)=>b.deficit-a.deficit);const needs=[];if(st.vocab===0){needs.push("Empty engine. Feed any text to begin.");}else{if(st.depths.deep<3)needs.push("Very few deep words. Feed more repeated sentences.");if(underexposed.length>5)needs.push(`${underexposed.length} shallow words seen 3+ times. Feed contexts for: ${underexposed.slice(0,5).map(w=>w.word).join(", ")}.`);if(starved.length>0)needs.push(`${starved.length} words predicted but structurally thin. Feed contexts for: ${starved.slice(0,5).map(w=>w.word).join(", ")}.`);if(brs.length>0)needs.push(`${brs.length} bridges connecting disconnected neighborhoods: ${brs.slice(0,3).map(b=>b.word).join(", ")}.`);if(needs.length===0)needs.push("Engine looks balanced. Feed new domain text to expand.");}return{vocab:st.vocab,depths:st.depths,vacuums:vacs,surpluses:surps,bridges:brs,underexposed:underexposed.slice(0,10),starved:starved.slice(0,10),needs};}
lesson(text,domain="general"){const before=this.engine.stats();const bp=this.engine.pressure();const fed=this.engine.feedText(text);const decayed=before.vocab>50?this.engine.decay(1):{edgesRemoved:0};const after=this.engine.stats();const ap=this.engine.pressure();const newWords=after.vocab-before.vocab;const nd={};for(const l of Object.keys(after.depths))nd[l]=after.depths[l]-before.depths[l];const bvw=new Set(bp.filter(p=>p.pressure<0).map(p=>p.word));const avw=new Set(ap.filter(p=>p.pressure<0).map(p=>p.word));const filled=[...bvw].filter(w=>!avw.has(w));const newV=[...avw].filter(w=>!bvw.has(w));this.lessons++;const r={lesson:this.lessons,domain,fed,newWords,newDepths:nd,decayed:decayed.edgesRemoved,filledVacuums:filled,newVacuums:newV};this.log.push(r);return r;}
drill(sentence){this.engine.feed(sentence);const fw=this.engine.scoreSentence(sentence);const words=sentence.toLowerCase().match(/[a-z0-9]+/g)||[];const rev=[...words].reverse().join(" ");const bw=this.engine.scoreSentence(rev);return{sentence,reversed:rev,forwardCoherence:fw.coherence,backwardCoherence:bw.coherence,asymmetry:fw.coherence-bw.coherence,learned:fw.coherence-bw.coherence>0.05};}
plan(){const d=this.diagnose();const steps=[];if(d.starved.length>0)steps.push({priority:1,action:"feed_targeted",targets:d.starved.slice(0,3).map(s=>s.word),instruction:`Feed varied sentences containing: ${d.starved.slice(0,3).map(s=>s.word).join(", ")}`});if(d.underexposed.length>0)steps.push({priority:2,action:"reinforce",targets:d.underexposed.slice(0,5).map(u=>u.word),instruction:`Reinforce shallow words: ${d.underexposed.slice(0,5).map(u=>u.word).join(", ")}`});if(d.bridges.length>0)steps.push({priority:3,action:"investigate",targets:d.bridges.slice(0,3).map(b=>b.word),instruction:`Investigate bridges: ${d.bridges.slice(0,3).map(b=>b.word).join(", ")}`});if(steps.length===0)steps.push({priority:1,action:"expand",instruction:"Engine balanced. Feed new domain text."});return{diagnosis:d,steps};}
cycle(text,domain="general"){const before=this.diagnose();const lesson=this.lesson(text,domain);const sents=text.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>10);let drill=null;if(sents.length>0)drill=this.drill(sents[0]);const after=this.diagnose();return{before:{vocab:before.vocab,depths:before.depths,needs:before.needs},lesson,drill,after:{vocab:after.vocab,depths:after.depths,needs:after.needs}};}
_walkFrom(start,maxLen){const nd=this.engine.nodes[start];if(!nd)return null;const words=[start];let cur=start;for(let i=0;i<maxLen-1;i++){const n=this.engine.nodes[cur];if(!n)break;// Try next first, fall back to neighbors
let pool=Object.entries(n.next||{});if(!pool.length)pool=Object.entries(n.neighbors||{});if(!pool.length)break;const tot=pool.reduce((s,[,w])=>s+w,0);let r=Math.random()*tot,picked=pool[0][0];for(const[w,wt]of pool){r-=wt;if(r<=0){picked=w;break;}}if(words.includes(picked)&&words.length>2)break;words.push(picked);cur=picked;}return words.length>=3?words.join(" "):null;}
_walkBetween(src,dst,maxLen){const eng=this.engine;if(!eng.nodes[src]||!eng.nodes[dst])return null;const words=[src];let cur=src;for(let i=0;i<maxLen-1;i++){const nd=eng.nodes[cur];if(!nd?.next)break;if(nd.next[dst]){words.push(dst);break;}const dstNb=new Set(Object.keys(eng.nodes[dst]?.neighbors||{}));let best=null,bestS=-1;for(const[c,w]of Object.entries(nd.next)){if(words.includes(c))continue;const cNb=Object.keys(eng.nodes[c]?.neighbors||{});const ov=cNb.filter(x=>dstNb.has(x)).length;const s=ov+w*0.1;if(s>bestS){bestS=s;best=c;}}if(!best)break;words.push(best);cur=best;}return words.length>=3?words.join(" "):null;}
spontaneous(){const eng=this.engine;const entries=Object.entries(eng.nodes);const vocab=entries.length;if(vocab<10)return{action:"wait",reason:"too few words"};const actions=[];
// Find words where neighbor count is low relative to frequency (structurally thin)
const thin=[];for(const[w,n]of entries){const nbCount=Object.keys(n.neighbors).length;const nxCount=Object.keys(n.next).length;if(n.freq>=2&&(nbCount<n.freq*0.5||nxCount<2))thin.push({word:w,freq:n.freq,nb:nbCount});}
thin.sort((a,b)=>b.freq-a.freq);
// Reinforce top thin words by walking from them
for(const t of thin.slice(0,8)){const g=this._walkFrom(t.word,8);if(g){eng.feed(g);actions.push({type:"reinforce",word:t.word,sentence:g});}}
// Bridge: connect two random high-freq words that don't share neighbors
if(vocab>50){const freq=entries.filter(([,n])=>n.freq>10).map(([w])=>w);if(freq.length>=2){const a=freq[Math.floor(Math.random()*freq.length)];const b=freq[Math.floor(Math.random()*freq.length)];if(a!==b&&!eng.nodes[a]?.neighbors[b]){const bridge=this._walkBetween(a,b,8);if(bridge){eng.feed(bridge);actions.push({type:"bridge",from:a,to:b,sentence:bridge});}}}}
// Decay noise
const decayed=vocab>100?eng.decay(1):{edgesRemoved:0};
this.lessons++;return{action:"spontaneous",lesson:this.lessons,generated:actions,decayed:decayed.edgesRemoved,vocab:Object.keys(eng.nodes).length};}}

const IDB={
  open(){return new Promise((res,rej)=>{const r=indexedDB.open("shifu-v2",1);r.onupgradeneeded=()=>r.result.createObjectStore("state");r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error);});},
  async get(key){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readonly");const r=tx.objectStore("state").get(key);r.onsuccess=()=>res(r.result||null);r.onerror=()=>rej(r.error);});},
  async set(key,val){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readwrite");tx.objectStore("state").put(val,key);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error);});},
  async del(key){const db=await this.open();return new Promise((res,rej)=>{const tx=db.transaction("state","readwrite");tx.objectStore("state").delete(key);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error);});}
};

// ═══════════════════════════════════════════════════════════════
// Intent parser
// ═══════════════════════════════════════════════════════════════
function parseIntent(input, engine) {
  const t = input.trim(), tl = t.toLowerCase();

  // OCR-correct the first word if the engine exists
  // so "descrbie", "comprae", "explroe" still match commands
  const words = tl.split(/\s+/);
  let cmd0 = words[0] || "";
  const COMMANDS = ["help","stats","reset","pressure","vacuums","vacuum","surpluses","surplus",
    "bridges","feed","score","correct","explore","compare","affinity","describe",
    "generate","speak","say","diagnose","lesson","drill","plan","teach","cycle"];
  if (engine && !COMMANDS.includes(cmd0)) {
    let best = cmd0, bestD = Infinity;
    for (const c of COMMANDS) {
      const d = editDistance(cmd0, c);
      if (d < bestD && d <= Math.ceil(c.length * 0.4)) { bestD = d; best = c; }
    }
    cmd0 = best;
  }

  if (cmd0 === "help" && words.length === 1) return { cmd: "help" };
  if (cmd0 === "stats" && words.length === 1) return { cmd: "stats" };
  if (cmd0 === "reset" && words.length === 1) return { cmd: "reset" };
  if ((cmd0 === "pressure") && words.length === 1) return { cmd: "pressure" };
  if ((cmd0 === "vacuums" || cmd0 === "vacuum") && words.length === 1) return { cmd: "vacuums" };
  if ((cmd0 === "surpluses" || cmd0 === "surplus") && words.length === 1) return { cmd: "surpluses" };
  if ((cmd0 === "bridges") && words.length === 1) return { cmd: "bridges" };

  // Rebuild with corrected command word
  const rest = words.slice(1).join(" ");

  if (cmd0 === "feed") return { cmd: "feed", text: rest };
  if (cmd0 === "score") return { cmd: "score", text: rest };
  if (cmd0 === "correct") return { cmd: "correct", text: rest };
  if (cmd0 === "explore") return { cmd: "explore", word: rest.trim() };
  if (cmd0 === "generate" || cmd0 === "speak" || cmd0 === "say") return { cmd: "generate", seed: rest.trim() };

  // Two-word commands: compare/affinity/describe A and B
  const twoWord = rest.match(/^(\S+)\s+(?:and|vs|,)\s+(\S+)/);
  if (cmd0 === "compare" && twoWord) {
    // OCR-correct the word arguments too
    let wa = twoWord[1], wb = twoWord[2];
    if (engine) {
      if (!engine.nodes[wa]) { const c = engine.correct(wa, 1); if (c.candidates[0]?.score > 0.5) wa = c.candidates[0].word; }
      if (!engine.nodes[wb]) { const c = engine.correct(wb, 1); if (c.candidates[0]?.score > 0.5) wb = c.candidates[0].word; }
    }
    return { cmd: "compare", a: wa, b: wb };
  }
  if (cmd0 === "affinity" && twoWord) {
    let wa = twoWord[1], wb = twoWord[2];
    if (engine) {
      if (!engine.nodes[wa]) { const c = engine.correct(wa, 1); if (c.candidates[0]?.score > 0.5) wa = c.candidates[0].word; }
      if (!engine.nodes[wb]) { const c = engine.correct(wb, 1); if (c.candidates[0]?.score > 0.5) wb = c.candidates[0].word; }
    }
    return { cmd: "affinity", a: wa, b: wb };
  }
  if (cmd0 === "describe" && twoWord) {
    let wa = twoWord[1], wb = twoWord[2];
    if (engine) {
      if (!engine.nodes[wa]) { const c = engine.correct(wa, 1); if (c.candidates[0]?.score > 0.5) wa = c.candidates[0].word; }
      if (!engine.nodes[wb]) { const c = engine.correct(wb, 1); if (c.candidates[0]?.score > 0.5) wb = c.candidates[0].word; }
    }
    return { cmd: "describe", a: wa, b: wb };
  }

  // Teacher commands
  if (cmd0 === "diagnose" && words.length === 1) return { cmd: "diagnose" };
  if (cmd0 === "plan" && words.length === 1) return { cmd: "plan" };
  if (cmd0 === "lesson" || cmd0 === "teach") return { cmd: "lesson", text: rest };
  if (cmd0 === "drill") return { cmd: "drill", text: rest };
  if (cmd0 === "cycle") return { cmd: "cycle", text: rest };

  // Single word = explore (with OCR correction)
  const tk = tokenize(t);
  if (tk.length === 1) {
    let w = tk[0];
    if (engine && !engine.nodes[w]) {
      const c = engine.correct(w, 1);
      if (c.candidates[0]?.score > 0.5) w = c.candidates[0].word;
    }
    return { cmd: "explore", word: w };
  }
  if (tk.length >= 2) return { cmd: "feed", text: t };
  return { cmd: "unknown" };
}

// ═══════════════════════════════════════════════════════════════
// Visual components
// ═══════════════════════════════════════════════════════════════
const DC = { unborn: "#3a3a3a", surface: "#5a6a7a", shallow: "#4a8ab5", forming: "#c4a035", structured: "#48b89a", deep: "#2eff8a" };
const Badge = ({ level }) => <span style={{ background: DC[level] || "#3a3a3a", color: "#0a0e14", padding: "2px 8px", borderRadius: 3, fontSize: ".75em", fontWeight: "bold" }}>{level}</span>;
const Bar = ({ value, color = "#48b89a", label, w = 120 }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: ".8em", marginBottom: 2 }}>
    {label && <span style={{ width: 90, color: "#5a6a7a", textAlign: "right" }}>{label}</span>}
    <div style={{ width: w, height: 8, background: "#1a2030", borderRadius: 4 }}>
      <div style={{ width: `${Math.min(Math.abs(value), 1) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
    <span style={{ color: "#5a6a7a", minWidth: 40 }}>{value.toFixed(3)}</span>
  </div>
);

// ═══════════════════════════════════════════════════════════════
// Response renderers
// ═══════════════════════════════════════════════════════════════
function rExplore(eng, word) {
  const w = word.toLowerCase(), d = eng.depth(w), node = eng.nodes[w];
  if (!node) return <div><Badge level="unborn" /> <b>{w}</b> -- never seen</div>;
  const nb = Object.entries(node.neighbors).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const nx = Object.entries(node.next).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const px = Object.entries(node.prev).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const sim = eng.similar(w, 5);
  return (<div>
    <div style={{ marginBottom: 8 }}><Badge level={d.level} /> <b>{w}</b> <span style={{ color: "#5a6a7a" }}>freq:{node.freq} ev:{d.evidence.toFixed(2)}</span></div>
    {nb.length > 0 && <div style={{ marginBottom: 4 }}><span style={{ color: "#5a6a7a" }}>neighbors:</span> {nb.map(([w, c]) => <span key={w} style={{ marginRight: 8 }}>{w}<sup style={{ color: "#5a6a7a" }}>{c}</sup></span>)}</div>}
    {nx.length > 0 && <div style={{ marginBottom: 4 }}><span style={{ color: "#5a6a7a" }}>next:</span> {nx.map(([w, c]) => <span key={w} style={{ color: "#48b89a", marginRight: 8 }}>{w}<sup style={{ color: "#5a6a7a" }}>{c}</sup></span>)}</div>}
    {px.length > 0 && <div style={{ marginBottom: 4 }}><span style={{ color: "#5a6a7a" }}>prev:</span> {px.map(([w, c]) => <span key={w} style={{ color: "#4a8ab5", marginRight: 8 }}>{w}<sup style={{ color: "#5a6a7a" }}>{c}</sup></span>)}</div>}
    {sim.length > 0 && <div><span style={{ color: "#5a6a7a" }}>similar:</span> {sim.map(s => <span key={s.word} style={{ marginRight: 8 }}>{s.word}<span style={{ color: "#5a6a7a" }}>({s.similarity.toFixed(3)})</span></span>)}</div>}
  </div>);
}

function rScore(r) {
  const cc = r.coherence > .6 ? "#48b89a" : r.coherence > .3 ? "#c4a035" : "#c44035";
  return (<div>
    <div style={{ marginBottom: 8 }}>coherence: <b style={{ color: cc }}>{r.coherence.toFixed(4)}</b> <span style={{ color: "#5a6a7a" }}>mean surprise: {r.meanSurprise.toFixed(4)}</span></div>
    {r.steps.map((s, i) => {
      const sc = s.surprise < .3 ? "#48b89a" : s.surprise < .7 ? "#c4a035" : "#c44035";
      return (<div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
        <span style={{ width: 100, color: s.known ? "#a8b4c4" : "#5a6a7a" }}>{s.afGate > .2 && "\u26A1"}{s.word}</span>
        <div style={{ width: 80, height: 8, background: "#1a2030", borderRadius: 4 }}>
          <div style={{ width: `${Math.min(s.surprise, 1) * 100}%`, height: "100%", background: sc, borderRadius: 4 }} />
        </div>
        <span style={{ color: "#5a6a7a", fontSize: ".75em" }}>{s.surprise.toFixed(3)}</span>
      </div>);
    })}
  </div>);
}

function rCompare(r, a, b) {
  const s = r.signals;
  return (<div>
    <div style={{ marginBottom: 8 }}>similarity: <b style={{ color: "#48b89a" }}>{r.similarity.toFixed(4)}</b> <span style={{ color: "#5a6a7a" }}>depth:{r.depth} weight:{r.totalWeight.toFixed(2)}</span></div>
    {s.neighborOverlap != null && <Bar label="neighbors" value={s.neighborOverlap} />}
    {s.expectsAB != null && <Bar label={`${a}\u2192${b}`} value={s.expectsAB} color="#4a8ab5" />}
    {s.expectsBA != null && <Bar label={`${b}\u2192${a}`} value={s.expectsBA} color="#c4a035" />}
    {s.trajectoryAB != null && <Bar label={`traj ${a}\u2192`} value={s.trajectoryAB} color="#4a8ab5" />}
    {s.trajectoryBA != null && <Bar label={`traj ${b}\u2192`} value={s.trajectoryBA} color="#c4a035" />}
    {s.indirectAB != null && <Bar label="indirect" value={s.indirectAB} color="#7a5ab5" />}
    {s.posSim != null && <Bar label="position" value={s.posSim} color="#5a6a7a" />}
    <Bar label="edit" value={s.editSim} color="#3a3a3a" />
    <Bar label="bigram" value={s.bigramSim} color="#3a3a3a" />
  </div>);
}

function rAffinity(r) {
  if (!r.known) return <div style={{ color: "#5a6a7a" }}>one or both words unknown</div>;
  return (<div>
    <div style={{ marginBottom: 8 }}>mutual: <b style={{ color: "#48b89a" }}>{r.mutual.toFixed(4)}</b> <span style={{ color: "#5a6a7a" }}>asym:{r.asym.toFixed(4)}</span></div>
    <Bar label="orbit" value={r.orbit} />
    <Bar label={`pull ${r.a}\u2192`} value={r.pullAB} color="#4a8ab5" />
    <Bar label={`pull ${r.b}\u2192`} value={r.pullBA} color="#c4a035" />
    <Bar label={`ind ${r.a}\u2192`} value={r.indAB} color="#7a5ab5" />
    <Bar label={`ind ${r.b}\u2192`} value={r.indBA} color="#7a5ab5" />
    <Bar label="charSim" value={r.charSim} color="#3a3a3a" />
    <Bar label="expOvlp" value={r.expOvlp} color="#5a6a7a" />
    <Bar label="posAlign" value={r.posAlign} color="#5a6a7a" />
  </div>);
}

function rPressure(items, label) {
  if (!items.length) return <div style={{ color: "#5a6a7a" }}>no {label}</div>;
  return (<div>{items.map((p, i) => (
    <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
      <span style={{ width: 90 }}>{p.word}</span>
      <Badge level={p.depth} />
      <div style={{ width: 60, height: 8, background: "#1a2030", borderRadius: 4 }}>
        <div style={{ width: `${Math.min(Math.abs(p.pressure) / 10, 1) * 100}%`, height: "100%", background: p.pressure < 0 ? "#c44035" : "#48b89a", borderRadius: 4 }} />
      </div>
      <span style={{ color: "#5a6a7a", fontSize: ".75em" }}>{p.pressure > 0 ? "+" : ""}{p.pressure} cl:{p.closure.toFixed(2)}</span>
    </div>
  ))}</div>);
}

const HELP = [
  ["feed: [text]", "Feed text to the stone"],
  ["[word]", "Explore a word"],
  ["compare A and B", "Compare (asymmetric)"],
  ["affinity A and B", "Pre-contact attraction"],
  ["describe A and B", "Describe relationship from grooves"],
  ["generate [word]", "Generate sentence by walking grooves"],
  ["diagnose", "What does the engine need?"],
  ["plan", "Teaching plan based on pressure"],
  ["lesson [text]", "Directed feed + verify changes"],
  ["drill [sentence]", "Contrastive forward/backward scoring"],
  ["cycle [text]", "Full teaching iteration"],
  ["score: [sentence]", "Walk with surprise"],
  ["correct [text]", "OCR correction"],
  ["pressure", "Graph shape"],
  ["vacuums / surpluses", "Where graph pulls / pushes"],
  ["bridges", "Structural bridges"],
  ["stats", "Vocabulary + depths"],
  ["reset", "Clear engine"],
];

// ═══════════════════════════════════════════════════════════════
// Engine profiles
// ═══════════════════════════════════════════════════════════════
const ENGINE_PROFILES = [
  { id: "clinical",   label: "Clinical Analysis (v2)" },
  { id: "medical",    label: "Medical Diagnosis" },
  { id: "legal",      label: "Legal Review" },
  { id: "financial",  label: "Financial Modeling" },
  { id: "scientific", label: "Scientific Research" },
];

function idbKeyForProfile(profileId) {
  return "engine_" + profileId;
}

// ═══════════════════════════════════════════════════════════════
// SVG chart components (no external libs)
// ═══════════════════════════════════════════════════════════════
const CHART_PAD = { top: 8, right: 12, bottom: 20, left: 36 };
const ACCENT = "#48b89a";
const CHART_BG = "#0e1219";
const GRID_COLOR = "#1a2130";
const LABEL_COLOR = "#5a6a7a";

function LineChart({ data, width, height, label }) {
  if (!data || data.length === 0) {
    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect width={width} height={height} fill={CHART_BG} rx={4} />
        <text x={width / 2} y={height / 2} fill={LABEL_COLOR} fontSize={11} textAnchor="middle" fontFamily="'IBM Plex Mono',monospace">awaiting data...</text>
      </svg>
    );
  }
  const w = width - CHART_PAD.left - CHART_PAD.right;
  const h = height - CHART_PAD.top - CHART_PAD.bottom;
  const mn = Math.min(...data);
  const mx = Math.max(...data);
  const range = mx - mn || 1;
  const pts = data.map((v, i) => {
    const x = CHART_PAD.left + (data.length > 1 ? (i / (data.length - 1)) * w : w / 2);
    const y = CHART_PAD.top + h - ((v - mn) / range) * h;
    return `${x},${y}`;
  });
  const polyline = pts.join(" ");
  const gridLines = 3;
  const grids = [];
  for (let g = 0; g <= gridLines; g++) {
    const gy = CHART_PAD.top + (g / gridLines) * h;
    const val = mx - (g / gridLines) * range;
    grids.push({ y: gy, val });
  }
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <rect width={width} height={height} fill={CHART_BG} rx={4} />
      {grids.map((g, i) => (
        <g key={i}>
          <line x1={CHART_PAD.left} y1={g.y} x2={width - CHART_PAD.right} y2={g.y} stroke={GRID_COLOR} strokeWidth={1} />
          <text x={CHART_PAD.left - 4} y={g.y + 3} fill={LABEL_COLOR} fontSize={8} textAnchor="end" fontFamily="'IBM Plex Mono',monospace">{g.val.toFixed(2)}</text>
        </g>
      ))}
      <polyline points={polyline} fill="none" stroke={ACCENT} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      {data.map((v, i) => {
        const x = CHART_PAD.left + (data.length > 1 ? (i / (data.length - 1)) * w : w / 2);
        const y = CHART_PAD.top + h - ((v - mn) / range) * h;
        return <circle key={i} cx={x} cy={y} r={2} fill={ACCENT} />;
      })}
      {label && <text x={CHART_PAD.left} y={height - 4} fill={LABEL_COLOR} fontSize={8} fontFamily="'IBM Plex Mono',monospace">{label}</text>}
    </svg>
  );
}

function BarChart({ data, width, height }) {
  // data = { surface: N, shallow: N, forming: N, structured: N, deep: N }
  const entries = Object.entries(data || {}).filter(([k]) => k !== "unborn");
  if (!entries.length || entries.every(([, v]) => v === 0)) {
    return (
      <svg width={width} height={height} style={{ display: "block" }}>
        <rect width={width} height={height} fill={CHART_BG} rx={4} />
        <text x={width / 2} y={height / 2} fill={LABEL_COLOR} fontSize={11} textAnchor="middle" fontFamily="'IBM Plex Mono',monospace">awaiting data...</text>
      </svg>
    );
  }
  const w = width - CHART_PAD.left - CHART_PAD.right;
  const h = height - CHART_PAD.top - CHART_PAD.bottom;
  const mx = Math.max(...entries.map(([, v]) => v), 1);
  const barW = Math.max(w / entries.length - 6, 4);
  const barColors = { surface: "#5a6a7a", shallow: "#4a8ab5", forming: "#c4a035", structured: "#48b89a", deep: "#2eff8a" };
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <rect width={width} height={height} fill={CHART_BG} rx={4} />
      {entries.map(([k, v], i) => {
        const bh = (v / mx) * h;
        const x = CHART_PAD.left + (i / entries.length) * w + 3;
        const y = CHART_PAD.top + h - bh;
        return (
          <g key={k}>
            <rect x={x} y={y} width={barW} height={bh} fill={barColors[k] || ACCENT} rx={2} />
            <text x={x + barW / 2} y={CHART_PAD.top + h + 12} fill={LABEL_COLOR} fontSize={7} textAnchor="middle" fontFamily="'IBM Plex Mono',monospace">{k.slice(0, 4)}</text>
            <text x={x + barW / 2} y={y - 3} fill={LABEL_COLOR} fontSize={8} textAnchor="middle" fontFamily="'IBM Plex Mono',monospace">{v}</text>
          </g>
        );
      })}
    </svg>
  );
}

function ChartPanel({ title, children }) {
  return (
    <div style={{
      background: "#0e1219",
      border: "1px solid #1a2130",
      borderRadius: 6,
      padding: "10px 12px",
      flex: 1,
      minHeight: 0,
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{ color: "#5a6a7a", fontSize: ".7em", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>{title}</div>
      <div style={{ flex: 1, minHeight: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
        {children}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════
export default function ShifuChat() {
  const [eng, setEng] = useState(null);
  const [teacher, setTeacher] = useState(null);
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [wc, setWc] = useState(0);
  const [profileId, setProfileId] = useState("clinical");
  const [coherenceHistory, setCoherenceHistory] = useState([]);
  const [entropyHistory, setEntropyHistory] = useState([]);
  const [surpriseHistory, setSurpriseHistory] = useState([]);
  const [teachingHistory, setTeachingHistory] = useState([]);
  const [depthCounts, setDepthCounts] = useState({});
  const chatRef = useRef(null);
  const fileRef = useRef(null);

  // Load engine for current profile
  useEffect(() => {
    (async () => {
      try {
        const key = idbKeyForProfile(profileId);
        const saved = await IDB.get(key);
        if (saved) {
          const e = ShifuEmbryo.deserialize(saved);
          setEng(e); setTeacher(new Teacher(e));
          setWc(Object.keys(e.nodes).length);
          const st = e.stats();
          setDepthCounts(st.depths);
          setMsgs([{ from: "bot", content: <span style={{ color: "#5a6a7a" }}>restored {Object.keys(e.nodes).length} words from {ENGINE_PROFILES.find(p => p.id === profileId)?.label}</span> }]);
        } else {
          const e = new ShifuEmbryo();
          setEng(e); setTeacher(new Teacher(e));
          setWc(0);
          setDepthCounts({});
          setMsgs([{ from: "bot", content: <span style={{ color: "#5a6a7a" }}>new engine initialized: {ENGINE_PROFILES.find(p => p.id === profileId)?.label}. type <span style={{ color: "#48b89a" }}>help</span></span> }]);
        }
        setCoherenceHistory([]);
        setEntropyHistory([]);
        setSurpriseHistory([]); setTeachingHistory([]);
      } catch {
        const fe = new ShifuEmbryo();
        setEng(fe); setTeacher(new Teacher(fe));
        setWc(0);
      }
    })();
  }, [profileId]);

  const save = useCallback(async (e) => {
    try { await IDB.set(idbKeyForProfile(profileId), e.serialize()); } catch {}
  }, [profileId]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTo(0, chatRef.current.scrollHeight);
  }, [msgs]);

  // Update viz after engine changes
  const updateViz = useCallback((engine) => {
    const st = engine.stats();
    setDepthCounts(st.depths);
    setWc(Object.keys(engine.nodes).length);
  }, []);

  const processCommand = async (t) => {
    if (!t.trim() || !eng) return;
    setMsgs(p => [...p, { from: "user", content: t }]);
    const intent = parseIntent(t, eng);
    let res;

    switch (intent.cmd) {
      case "help": res = <div>{HELP.map(([c, d], i) => <div key={i}><span style={{ color: "#48b89a", display: "inline-block", width: 180 }}>{c}</span><span style={{ color: "#5a6a7a" }}>{d}</span></div>)}</div>; break;
      case "stats": { const s = eng.stats(); res = <div><div>v{s.version} | vocab: <b>{s.vocab}</b> | sent: {s.sentences} | tok: {s.tokens}</div><div style={{ marginTop: 4 }}>{Object.entries(s.depths).filter(([, v]) => v > 0).map(([l, c]) => <span key={l} style={{ marginRight: 10 }}><Badge level={l} /> {c}</span>)}</div></div>; break; }
      case "reset": { const f = new ShifuEmbryo(); setEng(f); setTeacher(new Teacher(f)); setWc(0); setDepthCounts({}); setCoherenceHistory([]); setEntropyHistory([]); setSurpriseHistory([]); setTeachingHistory([]); await IDB.del(idbKeyForProfile(profileId)); res = <span style={{ color: "#5a6a7a" }}>engine cleared</span>; break; }
      case "feed": {
        const r = eng.feedText(intent.text);
        updateViz(eng);
        await save(eng);
        // Update entropy — novelty = new words ratio
        const vocab = Object.keys(eng.nodes).length;
        const novelty = vocab > 0 ? r.tokens / vocab : 0;
        setEntropyHistory(prev => [...prev.slice(-49), novelty]);
        // Spontaneous teaching — the engine teaches itself after each feed
        let teachMsg = null;
        if (teacher && Object.keys(eng.nodes).length >= 10) {
          const sp = teacher.spontaneous();
          if (sp.generated?.length > 0 || sp.decayed > 0) {
            teachMsg = (<div style={{ marginTop: 6, fontSize: ".85em", color: "#5a6a7a", borderTop: "1px solid #1a2030", paddingTop: 6 }}>
              <span style={{ color: "#c4a035" }}>spontaneous consolidation:</span>
              {sp.generated?.map((g, i) => <div key={i}>{g.type}: <span style={{ color: "#48b89a" }}>{g.sentence}</span></div>)}
              {sp.decayed > 0 && <div>pruned {sp.decayed} weak edges</div>}
            </div>);
            updateViz(eng);
            await save(eng);
            setTeachingHistory(prev => [...prev.slice(-49), {
              reinforcements: (sp.generated || []).filter(a => a.type === "reinforce").length,
              bridges: (sp.generated || []).filter(a => a.type === "bridge").length,
              decayed: sp.decayed || 0,
              vocab: Object.keys(eng.nodes).length,
            }]);
          }
        }
        res = <div>
          <span>fed {r.sentences} sent, {r.tokens} tok <span style={{ color: "#5a6a7a" }}>| vocab: {Object.keys(eng.nodes).length}</span></span>
          {teachMsg}
        </div>;
        break;
      }
      case "explore": res = rExplore(eng, intent.word); break;
      case "generate": { const g = eng.generate(intent.seed); res = g.text ? <div><span style={{ color: "#48b89a" }}>{g.text}</span> <span style={{ color: "#5a6a7a" }}>({g.words.length} words)</span></div> : <span style={{ color: "#5a6a7a" }}>{g.reason || "no grooves to follow"}</span>; break; }
      case "score": {
        const scoreResult = eng.scoreSentence(intent.text);
        res = rScore(scoreResult);
        // Update coherence history
        setCoherenceHistory(prev => [...prev.slice(-49), scoreResult.coherence]);
        // Update surprise history
        setSurpriseHistory(prev => [...prev.slice(-49), scoreResult.meanSurprise]);
        break;
      }
      case "compare": res = rCompare(eng.compare(intent.a, intent.b), intent.a, intent.b); break;
      case "affinity": res = rAffinity(eng.affinity(intent.a, intent.b)); break;
      case "describe": { const d = eng.describe(intent.a, intent.b); res = (<div>
        <div style={{ marginBottom: 8 }}><b>{d.a}</b> <span style={{ color: "#5a6a7a" }}>&harr;</span> <b>{d.b}</b></div>
        {d.description ? (<div style={{ marginBottom: 8 }}><span style={{ color: "#48b89a", fontSize: "1.05em" }}>{d.description}</span> <span style={{ color: "#5a6a7a" }}>({d.coherence?.toFixed(3)})</span></div>) : (<div style={{ color: "#5a6a7a" }}>no path found</div>)}
        {d.shared?.length > 0 && <div style={{ marginBottom: 4 }}><span style={{ color: "#5a6a7a" }}>shared context:</span> {d.shared.map(s => <span key={s.word} style={{ marginRight: 8 }}>{s.word}<sup style={{ color: "#5a6a7a" }}>{s.weight}</sup></span>)}</div>}
        {d.bridges?.length > 0 && <div style={{ marginBottom: 4 }}><span style={{ color: "#5a6a7a" }}>bridges:</span> {d.bridges.map(b => <span key={b.word} style={{ color: "#4a8ab5", marginRight: 8 }}>{b.word}</span>)}</div>}
        {d.candidates?.length > 1 && <div style={{ marginTop: 6 }}><span style={{ color: "#5a6a7a" }}>alternatives:</span>{d.candidates.slice(1, 4).map((c, i) => <div key={i} style={{ color: "#5a6a7a", fontSize: ".85em", marginLeft: 8 }}>{c.coherence.toFixed(3)} {c.sentence}</div>)}</div>}
      </div>); break; }
      case "correct": { const ws = intent.text.split(/\s+/); res = <div>{ws.map((w, i) => { if (eng.nodes[w.toLowerCase()]) return <span key={i} style={{ marginRight: 6 }}>{w}</span>; const r = eng.correct(w); const top = r.candidates[0]; if (!top) return <span key={i} style={{ color: "#c44035", marginRight: 6 }}>{w}?</span>; return <span key={i} style={{ marginRight: 6 }}><span style={{ textDecoration: "line-through", color: "#5a6a7a" }}>{w}</span> <span style={{ color: "#48b89a" }}>{top.word}</span><sup style={{ color: "#5a6a7a" }}>{r.confidence.toFixed(2)}</sup></span>; })}</div>; break; }
      case "pressure": res = rPressure(eng.pressure().slice(0, 15), "pressure"); break;
      case "vacuums": res = rPressure(eng.vacuums(10), "vacuums"); break;
      case "surpluses": res = rPressure(eng.surpluses(10), "surpluses"); break;
      case "bridges": res = rPressure(eng.bridges(10), "bridges"); break;
      case "diagnose": { if (!teacher) break; const d = teacher.diagnose(); res = (<div>
        <div style={{ marginBottom: 6 }}>vocab: <b>{d.vocab}</b></div>
        {d.needs.map((n, i) => <div key={i} style={{ color: "#c4a035", marginBottom: 4 }}>{n}</div>)}
        {d.starved.length > 0 && <div style={{ marginTop: 6 }}><span style={{ color: "#5a6a7a" }}>starved:</span> {d.starved.slice(0,5).map(s => <span key={s.word} style={{ marginRight: 8, color: "#c44035" }}>{s.word}<sup>{s.deficit}</sup></span>)}</div>}
        {d.underexposed.length > 0 && <div><span style={{ color: "#5a6a7a" }}>underexposed:</span> {d.underexposed.slice(0,5).map(u => <span key={u.word} style={{ marginRight: 8 }}>{u.word}<sup>{u.freq}</sup></span>)}</div>}
      </div>); break; }
      case "plan": { if (!teacher) break; const p = teacher.plan(); res = (<div>
        <div style={{ marginBottom: 8, color: "#5a6a7a" }}>teaching plan ({p.steps.length} steps):</div>
        {p.steps.map((s, i) => <div key={i} style={{ marginBottom: 6 }}><span style={{ color: "#48b89a" }}>#{s.priority}</span> <b>{s.action}</b>: {s.instruction}</div>)}
      </div>); break; }
      case "lesson": { if (!teacher || !intent.text) break; const r = teacher.lesson(intent.text); updateViz(eng); await save(eng); res = (<div>
        <div>lesson #{r.lesson}: {r.fed.sentences} sent, {r.fed.tokens} tok | +{r.newWords} words | pruned {r.decayed} edges</div>
        {r.filledVacuums.length > 0 && <div style={{ color: "#48b89a" }}>filled vacuums: {r.filledVacuums.join(", ")}</div>}
        {r.newVacuums.length > 0 && <div style={{ color: "#c44035" }}>new vacuums: {r.newVacuums.join(", ")}</div>}
      </div>); break; }
      case "drill": { if (!teacher || !intent.text) break; const r = teacher.drill(intent.text); res = (<div>
        <div style={{ marginBottom: 4 }}>forward: <b style={{ color: "#48b89a" }}>{r.forwardCoherence.toFixed(4)}</b> | backward: <b style={{ color: "#c4a035" }}>{r.backwardCoherence.toFixed(4)}</b></div>
        <div>asymmetry: <b>{r.asymmetry.toFixed(4)}</b> {r.learned ? <span style={{ color: "#48b89a" }}>-- direction learned</span> : <span style={{ color: "#c44035" }}>-- direction not yet clear</span>}</div>
        <div style={{ color: "#5a6a7a", fontSize: ".8em", marginTop: 4 }}>reversed: {r.reversed}</div>
      </div>); break; }
      case "cycle": { if (!teacher || !intent.text) break; const r = teacher.cycle(intent.text); updateViz(eng); await save(eng); res = (<div>
        <div style={{ marginBottom: 4 }}>before: {r.before.vocab} words | after: {r.after.vocab} words</div>
        <div>{r.lesson.fed.sentences} sent, +{r.lesson.newWords} words, pruned {r.lesson.decayed}</div>
        {r.drill && <div style={{ marginTop: 4 }}>drill: fwd {r.drill.forwardCoherence.toFixed(3)} | bwd {r.drill.backwardCoherence.toFixed(3)} | asym {r.drill.asymmetry.toFixed(3)}</div>}
        {r.after.needs.map((n, i) => <div key={i} style={{ color: "#c4a035", marginTop: 4, fontSize: ".85em" }}>{n}</div>)}
      </div>); break; }
      default: res = <span style={{ color: "#5a6a7a" }}>type <span style={{ color: "#48b89a" }}>help</span></span>;
    }
    setMsgs(p => [...p, { from: "bot", content: res }]);
  };

  const submit = async (ev) => {
    ev.preventDefault();
    if (!input.trim()) return;
    const t = input.trim();
    setInput("");
    await processCommand(t);
  };

  // Extract text from PDF using pdf.js
  const extractPdfText = async (arrayBuffer) => {
    const pdfjsLib = await import("pdfjs-dist");
    const workerSrc = await import("pdfjs-dist/build/pdf.worker.min.mjs?url");
    pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc.default;
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const pages = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      pages.push(content.items.map(item => item.str).join(" "));
    }
    return pages.join("\n");
  };

  const ingestFile = async (file) => {
    if (!eng) return;
    const sizeTxt = (file.size / 1024).toFixed(1) + "KB";
    setMsgs(p => [...p, { from: "user", content: <span style={{ color: "#4a8ab5" }}>uploaded: {file.name} ({sizeTxt})</span> }]);

    let text = "";
    try {
      if (file.name.toLowerCase().endsWith(".pdf")) {
        const buf = await file.arrayBuffer();
        text = await extractPdfText(buf);
        if (text.trim().length < 20) {
          setMsgs(p => [...p, { from: "bot", content: <span style={{ color: "#c44035" }}>could not extract text from <b>{file.name}</b> (scanned/image PDF)</span> }]);
          return;
        }
      } else {
        text = await file.text();
      }
    } catch (err) {
      setMsgs(p => [...p, { from: "bot", content: <span style={{ color: "#c44035" }}>error reading <b>{file.name}</b>: {err.message}</span> }]);
      return;
    }

    if (text.trim().length === 0) {
      setMsgs(p => [...p, { from: "bot", content: <span style={{ color: "#5a6a7a" }}>empty file: {file.name}</span> }]);
      return;
    }

    const beforeVocab = Object.keys(eng.nodes).length;
    const r = eng.feedText(text);
    const afterVocab = Object.keys(eng.nodes).length;

    // Consolidation: decay weak edges after bulk ingestion
    if (r.tokens > 500 && typeof eng.decay === "function") {
      eng.decay(1);
    }

    updateViz(eng);
    await save(eng);
    const novelty = (afterVocab - beforeVocab) / Math.max(r.tokens, 1);
    setEntropyHistory(prev => [...prev.slice(-49), novelty]);

    // Multiple spontaneous cycles — run until consolidation settles
    const allActions = [];
    let totalDecayed = 0;
    if (teacher) {
      for (let cycle = 0; cycle < 5; cycle++) {
        const sp = teacher.spontaneous();
        if (sp.generated) allActions.push(...sp.generated);
        totalDecayed += sp.decayed || 0;
        // Stop if nothing generated and nothing decayed — settled
        if ((!sp.generated || sp.generated.length === 0) && (sp.decayed || 0) === 0) break;
      }
    }

    const teachMsg = (allActions.length > 0 || totalDecayed > 0) ? (
      <div style={{ marginTop: 6, fontSize: ".85em", color: "#5a6a7a", borderTop: "1px solid #1a2030", paddingTop: 6 }}>
        <span style={{ color: "#c4a035" }}>spontaneous consolidation ({allActions.length} reinforcements):</span>
        {allActions.slice(0, 10).map((g, i) => <div key={i}>{g.type}: <span style={{ color: "#48b89a" }}>{g.sentence}</span></div>)}
        {allActions.length > 10 && <div style={{ color: "#5a6a7a" }}>...and {allActions.length - 10} more</div>}
        {totalDecayed > 0 && <div>pruned {totalDecayed} weak edges across cycles</div>}
      </div>
    ) : null;

    if (allActions.length > 0 || totalDecayed > 0) {
      updateViz(eng);
      await save(eng);
      setTeachingHistory(prev => [...prev.slice(-49), {
        reinforcements: allActions.filter(a => a.type === "reinforce").length,
        bridges: allActions.filter(a => a.type === "bridge").length,
        decayed: totalDecayed,
        vocab: Object.keys(eng.nodes).length,
      }]);
    }

    setMsgs(p => [...p, { from: "bot", content: <div>
      <span>ingested <b>{file.name}</b>: {r.sentences} sentences, {r.tokens} tokens <span style={{ color: "#5a6a7a" }}>| vocab: {Object.keys(eng.nodes).length}</span></span>
      {teachMsg}
    </div> }]);
  };

  const handleFileUpload = async (ev) => {
    const files = ev.target.files;
    if (!files || files.length === 0 || !eng) return;
    // Process all files sequentially
    for (const file of Array.from(files)) {
      await ingestFile(file);
    }
    ev.target.value = "";
  };

  const chartWidth = 280;
  const chartHeight = 100;

  return (
    <div style={{
      background: "#0a0e14",
      color: "#a8b4c4",
      fontFamily: "'IBM Plex Mono',monospace",
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* ── Header ── */}
      <div style={{
        padding: "16px 24px 12px",
        borderBottom: "1px solid #1a2130",
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div>
          <div style={{
            fontFamily: "'Instrument Serif',serif",
            fontSize: "2.2em",
            color: "#a8b4c4",
            lineHeight: 1,
            letterSpacing: "0.04em",
          }}>SHIFU</div>
          <select
            value={profileId}
            onChange={(e) => setProfileId(e.target.value)}
            style={{
              marginTop: 6,
              background: "#0e1219",
              border: "1px solid #1a2130",
              color: "#a8b4c4",
              padding: "5px 10px",
              fontFamily: "'IBM Plex Mono',monospace",
              fontSize: ".78em",
              borderRadius: 4,
              outline: "none",
              cursor: "pointer",
              appearance: "auto",
            }}
          >
            {ENGINE_PROFILES.map(p => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </div>
        <span style={{ color: "#5a6a7a", fontSize: ".78em", paddingBottom: 4 }}>{wc} words | {eng?.sentenceCount || 0} sentences</span>
      </div>

      {/* ── Body: split view ── */}
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>

        {/* ── LEFT: Chat (60%) ── */}
        <div style={{
          width: "60%",
          display: "flex",
          flexDirection: "column",
          borderRight: "1px solid #1a2130",
          minHeight: 0,
        }}>
          {/* Messages */}
          <div ref={chatRef} style={{
            flex: 1,
            overflow: "auto",
            padding: "16px 20px",
          }}>
            {msgs.map((m, i) => (
              <div key={i} style={{
                display: "flex",
                justifyContent: m.from === "user" ? "flex-end" : "flex-start",
                marginBottom: 10,
              }}>
                <div style={{
                  maxWidth: "85%",
                  padding: "10px 14px",
                  borderRadius: 8,
                  background: m.from === "user" ? "#1a2540" : "#0e1219",
                  border: `1px solid ${m.from === "user" ? "#1a3060" : "#1a2130"}`,
                  fontSize: ".88em",
                  lineHeight: 1.5,
                }}>
                  {m.content}
                </div>
              </div>
            ))}
          </div>

          {/* Input bar */}
          <form onSubmit={submit} style={{
            padding: "10px 16px",
            borderTop: "1px solid #1a2130",
            display: "flex",
            gap: 8,
            flexShrink: 0,
          }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Pose a clinical query or upload data stream..."
              style={{
                flex: 1,
                background: "#0e1219",
                border: "1px solid #1a2130",
                color: "#a8b4c4",
                padding: "10px 14px",
                fontFamily: "inherit",
                fontSize: ".85em",
                borderRadius: 6,
                outline: "none",
              }}
              autoFocus
            />
            {/* File upload button */}
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.pdf,.md,.csv,.json,.html,.xml,.doc,.rtf"
              multiple
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              title="Upload files (.txt, .pdf, .md, .csv...)"
              style={{
                background: "#0e1219",
                color: "#5a6a7a",
                border: "1px solid #1a2130",
                padding: "10px 14px",
                fontFamily: "inherit",
                fontSize: ".85em",
                borderRadius: 6,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
              }}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ display: "block" }}>
                <path d="M8 1v10M4 7l4-4 4 4" stroke="#5a6a7a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M2 12v2h12v-2" stroke="#5a6a7a" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <button
              type="submit"
              style={{
                background: "#48b89a",
                color: "#0a0e14",
                border: "none",
                padding: "10px 20px",
                fontFamily: "inherit",
                fontWeight: "bold",
                fontSize: ".85em",
                borderRadius: 6,
                cursor: "pointer",
              }}
            >Send</button>
          </form>
        </div>

        {/* ── RIGHT: Visualizations (40%) ── */}
        <div style={{
          width: "40%",
          display: "flex",
          flexDirection: "column",
          gap: 8,
          padding: "12px 14px",
          overflow: "auto",
          minHeight: 0,
        }}>
          <ChartPanel title="Thought Process Coherence">
            <LineChart
              data={coherenceHistory}
              width={chartWidth}
              height={chartHeight}
              label={coherenceHistory.length > 0 ? `latest: ${coherenceHistory[coherenceHistory.length - 1].toFixed(3)}` : null}
            />
          </ChartPanel>

          <ChartPanel title="Model Confidence">
            <BarChart
              data={depthCounts}
              width={chartWidth}
              height={chartHeight}
            />
          </ChartPanel>

          <ChartPanel title="Information Entropy">
            <LineChart
              data={entropyHistory}
              width={chartWidth}
              height={chartHeight}
              label={entropyHistory.length > 0 ? `latest: ${entropyHistory[entropyHistory.length - 1].toFixed(3)}` : null}
            />
          </ChartPanel>

          <ChartPanel title="Model Surprise Score">
            <LineChart
              data={surpriseHistory}
              width={chartWidth}
              height={chartHeight}
              label={surpriseHistory.length > 0 ? `latest: ${surpriseHistory[surpriseHistory.length - 1].toFixed(3)}` : null}
            />
          </ChartPanel>

          <ChartPanel title="Teaching Activity">
            {teachingHistory.length === 0 ? (
              <div style={{ color: "#5a6a7a", textAlign: "center", padding: 20, fontSize: ".85em" }}>awaiting data...</div>
            ) : (
              <svg width={chartWidth} height={chartHeight} style={{ display: "block" }}>
                {(() => {
                  const data = teachingHistory;
                  const maxR = Math.max(...data.map(d => d.reinforcements), 1);
                  const maxB = Math.max(...data.map(d => d.bridges), 1);
                  const maxD = Math.max(...data.map(d => d.decayed), 1);
                  const maxAll = Math.max(maxR, maxB, maxD);
                  const barW = Math.max(Math.floor(chartWidth / Math.max(data.length, 1)) - 2, 3);
                  return data.map((d, i) => {
                    const x = (i / Math.max(data.length - 1, 1)) * (chartWidth - barW);
                    const rH = (d.reinforcements / maxAll) * (chartHeight - 20);
                    const bH = (d.bridges / maxAll) * (chartHeight - 20);
                    const dH = (d.decayed / maxAll) * (chartHeight - 20);
                    return (
                      <g key={i}>
                        <rect x={x} y={chartHeight - 15 - rH} width={barW} height={rH} fill="#48b89a" opacity={0.8} />
                        <rect x={x} y={chartHeight - 15 - rH - bH} width={barW} height={bH} fill="#4a8ab5" opacity={0.8} />
                        <line x1={x} y1={chartHeight - 15} x2={x + barW} y2={chartHeight - 15 - (dH * 0.3)} stroke="#c44035" strokeWidth={1.5} opacity={0.6} />
                      </g>
                    );
                  });
                })()}
                <text x={0} y={chartHeight - 2} fill="#5a6a7a" fontSize="9" fontFamily="inherit">
                  {teachingHistory.length > 0 ? `r:${teachingHistory[teachingHistory.length-1].reinforcements} b:${teachingHistory[teachingHistory.length-1].bridges} d:${teachingHistory[teachingHistory.length-1].decayed}` : ""}
                </text>
              </svg>
            )}
          </ChartPanel>
        </div>
      </div>
    </div>
  );
}
