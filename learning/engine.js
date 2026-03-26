// SHIFU CORE ENGINE v2.1.0
//
// A word is one neuron. Each exposure grows its connections.
// Co-occurrence = dendrites. Expectations = axons. Resonance = lateral synapses.
//
// v2.0: wave propagation — the network acts as one
// v2.1: inhibition, error correction, settling — the network thinks
// v2.1.1: contextual gating — context selects meaning, same neuron different pattern
//
// Three new mechanisms:
//
// 1. INHIBITION
//    High-frequency words ("the", "and", "with") connect to everything.
//    Without inhibition they drown out meaningful signal at hop 2.
//    Fix: each word's activation energy is scaled by 1/log(freq+1),
//    normalized against the corpus median. Common words propagate LESS.
//    This is the GABAergic interneuron: widespread but suppressive.
//
// 2. ERROR CORRECTION (backward pass)
//    Forward pass: "physician" arrives, surprise is high (unexpected).
//    Then "treats" arrives (makes sense), then "patient" (confirms).
//    The backward pass: after all words are scored, the FINAL field's
//    energy flows backward. If word i was surprising but the final field
//    strongly activates word i, then the initial surprise was premature.
//    correctedSurprise[i] = forwardSurprise[i] * (1 - retroReinforcement)
//    This is the cerebellum: compare prediction to outcome, update error.
//
// 3. SETTLING (iterative convergence)
//    A sentence field computed in one forward pass isn't stable —
//    it's the first ripple. settle() re-propagates: take the field,
//    re-activate each significant node, let the interference pattern
//    update, repeat until the field changes less than epsilon.
//    This is the Hopfield attractor: the network converges to a
//    stable state that IS the settled meaning.

const VERSION = "2.1.1";
const CONFIG = {
  version: VERSION,
  channels: { form: 16, context: 12, history: 8, influence: 8, contrast: 8, expectation: 8 },
  routing: {
    correction: { ocr: .7, form: .3, context: 0, history: 0, influence: 0, contrast: 0, expectation: 0 },
    meaning:    { form: .20, context: .20, history: .10, influence: .15, contrast: .15, expectation: .20, ocr: 0 },
  },
  ocr: {"0,o":.1,"1,l":.2,"1,i":.2,"5,s":.3,"8,b":.3,"6,g":.4,"l,i":.2,"m,n":.4,"u,v":.5,"c,e":.5,"r,n":.3,"d,o":.3,"f,t":.4,"h,b":.4,"a,e":.4,"a,o":.4,"u,n":.4,"e,i":.4,"f,l":.4,"s,e":.5,"b,d":.4},
  thresholds: { confidence: { reject: 0.02, low: 0.05 } },
  soft: { threshold: 0.3, maxCandidates: 8, discount: 0.6 },
  decay: { factor: 0.97, interval: 100, minCount: 0.01 },
  skip: { window: 7, minDist: 2 },
  compact: { maxNxPerWord: 50, maxCoPerWord: 100, maxSnxPerWord: 50, maxResPerWord: 30, minFreq: 1 },
  resonance: { topK: 5, increment: 0.1, boostRate: 0.08, maxDiscount: 0.95 },
  wave: {
    decay: 0.3, maxHop1: 15, maxHop2Nodes: 5, maxHop2Each: 8,
    coWeight: 0.30, nxWeight: 0.25, resWeight: 0.25, snxWeight: 0.20,
  },
  // v2.1: inhibition — suppress high-frequency noise
  inhibition: {
    floor: 0.1,     // minimum inhibition weight (even rare words get slightly suppressed)
    ceiling: 1.0,   // maximum inhibition weight (rare words pass through fully)
  },
  // v2.1: backward error correction
  correction: {
    strength: 0.6,  // how much retroactive context reduces surprise (0=none, 1=full)
  },
  // v2.1: settling — iterative convergence
  settle: {
    maxIter: 3, epsilon: 0.01, reactivateTop: 8, iterDecay: 0.5,
  },
  // v2.1.1: contextual gating — context selects meaning
  gate: {
    dampen: 0.4,     // multiplier for connections NOT in context field (suppress)
    boost: 1.5,      // multiplier for connections aligned with context (amplify)
  },
};
const IDX = { form:[0,16], ctx:[16,28], hist:[28,36], inf:[36,44], con:[44,52], exp:[52,60], all:[0,60] };
const V = new Set("aeiou");
const mn=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
const sd=a=>{const m=mn(a);return Math.sqrt(mn(a.map(v=>(v-m)**2)));};
function cos(a,b,lo,hi){let d=0,na=0,nb=0;for(let i=lo;i<hi;i++){d+=a[i]*b[i];na+=a[i]**2;nb+=b[i]**2;}na=Math.sqrt(na);nb=Math.sqrt(nb);return na<1e-6||nb<1e-6?0:d/(na*nb);}
function cosVec(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]**2;nb+=b[i]**2;}na=Math.sqrt(na);nb=Math.sqrt(nb);return na<1e-6||nb<1e-6?0:d/(na*nb);}
function prng(seed){let s=seed;return()=>{s=(s*16807)%2147483647;return s/2147483647;};}
function shuffle(arr,rand){const a=[...arr];for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function ocrDist(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return ocrDist(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++){const k=[a[i],b[j]].sort().join(",");c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:(CONFIG.ocr[k]??1))));}p=c;}return p[b.length];}
function levDist(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return levDist(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++)c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:1)));p=c;}return p[b.length];}
function topN(obj, n){const e=Object.entries(obj).sort((a,b)=>b[1]-a[1]);if(e.length<=n)return obj;return Object.fromEntries(e.slice(0,n));}
function fieldCosine(a,b){let dot=0,na=0,nb=0;for(const[k,v]of a){na+=v*v;if(b.has(k))dot+=v*b.get(k);}for(const[,v]of b)nb+=v*v;na=Math.sqrt(na);nb=Math.sqrt(nb);return(na<1e-9||nb<1e-9)?0:dot/(na*nb);}

class ShifuEngine {
  constructor(cfg=CONFIG){
    this.config=cfg;this.version=cfg.version;
    this.wf={};this.co={};this.wp={};this.bf={};
    this.fs={};this.ls={};this.eg={};this.ph={};
    this.nv={};this._pn={};
    this.sl={};this.sn={};this.sd={};
    this._globalPos=[];this._globalSentLen=[];this._globalFreqSum=0;this._globalWordCount=0;
    this.nx={};this.px={};this.nx2={};this.snx={};this.res={};
    this.ns=0;this.nt=0;this._cache={};this._idxLen={};this._idxBg={};
    this._lastDecay=0;this._simCache={};
    this._medianFreq=1;this._medianDegree=1;
  }

  _reindex(w){const l=w.length;(this._idxLen[l]??=[]);if(!this._idxLen[l].includes(w))this._idxLen[l].push(w);for(let i=0;i<w.length-1;i++){const bg=w.slice(i,i+2);(this._idxBg[bg]??=[]);if(!this._idxBg[bg].includes(w))this._idxBg[bg].push(w);}}
  _candidates(g,r=2){g=g.toLowerCase();const c=new Set();for(let d=-r;d<=r;d++){const ws=this._idxLen[g.length+d];if(ws)for(const w of ws)c.add(w);}for(let i=0;i<g.length-1;i++){const ws=this._idxBg[g.slice(i,i+2)];if(ws)for(const w of ws)c.add(w);}return[...c];}

  // ─── Inhibition weight ────────────────────────────────────────────
  // Combines frequency-based AND degree-based suppression.
  // Frequency: how often does this word appear? (too common = noise)
  // Degree: how many different words does it co-occur with? (too connected = hub)
  // A word like "the" is both high-frequency AND high-degree → heavily suppressed.
  // A word like "patient" is high-frequency but moderate-degree → moderately suppressed.
  // A word like "thymectomy" is low-frequency and low-degree → passes through.
  _inhibitionWeight(w) {
    const cfg = this.config.inhibition || CONFIG.inhibition;
    const freq = this.wf[w] || 0;
    if (freq === 0) return cfg.ceiling;
    const medF = Math.max(this._medianFreq, 1);

    // Frequency component: 1/log2(freqRatio + 2)
    const freqRatio = freq / medF;
    const freqInh = 1 / Math.log2(freqRatio + 2);

    // Degree component: how many co-occurrence neighbors?
    const degree = this.co[w] ? Object.keys(this.co[w]).length : 0;
    const medDeg = Math.max(this._medianDegree, 1);
    const degRatio = degree / medDeg;
    const degInh = 1 / Math.log2(degRatio + 2);

    // Blend: both must agree for strong suppression
    // If a word is frequent BUT low-degree (e.g. technical term repeated), it passes.
    // If a word is infrequent BUT high-degree (unlikely), it gets some suppression.
    const raw = freqInh * 0.5 + degInh * 0.5;
    return Math.max(cfg.floor, Math.min(cfg.ceiling, raw));
  }

  // Update cached median frequency and degree
  _updateMedianFreq() {
    const freqs = Object.values(this.wf);
    if (freqs.length === 0) { this._medianFreq = 1; this._medianDegree = 1; return; }
    freqs.sort((a, b) => a - b);
    this._medianFreq = freqs[Math.floor(freqs.length / 2)] || 1;

    const degrees = Object.keys(this.wf).map(w => this.co[w] ? Object.keys(this.co[w]).length : 0);
    degrees.sort((a, b) => a - b);
    this._medianDegree = degrees[Math.floor(degrees.length / 2)] || 1;
  }

  // ─── Resonance ────────────────────────────────────────────────────
  _addResonance(a,b,amount){if(a===b)return;this.res[a]??={};this.res[b]??={};this.res[a][b]=(this.res[a][b]||0)+amount;this.res[b][a]=(this.res[b][a]||0)+amount;}
  _resonanceDiscount(target,bridge){const cfg=this.config.resonance||CONFIG.resonance;const baseDis=(this.config.soft||CONFIG.soft).discount;const evidence=this.res[target]?.[bridge]||0;return Math.min(baseDis+evidence*cfg.boostRate,cfg.maxDiscount);}
  resonancePartners(w,k=10){w=w.toLowerCase();const r=this.res[w];if(!r)return[];return Object.entries(r).sort((a,b)=>b[1]-a[1]).slice(0,k).map(([word,evidence])=>({word,evidence,discount:this._resonanceDiscount(w,word)}));}

  // ─── Soft lookups ─────────────────────────────────────────────────
  _similarTo(target,pool=null,k=null){const cfg=this.config.soft||CONFIG.soft;k=k||cfg.maxCandidates;const threshold=cfg.threshold;const cacheKey='~sim~'+target+'~'+(pool?pool.join(","):"all");if(this._simCache[cacheKey])return this._simCache[cacheKey];const tCtx=this.contextVec(target);if(tCtx.every(x=>x===0))return[];const candidates=pool||Object.keys(this.wf);const results=[];for(const w of candidates){if(w===target)continue;const wCtx=this.contextVec(w);if(wCtx.every(x=>x===0))continue;const sim=cosVec(tCtx,wCtx);if(sim>=threshold)results.push({word:w,sim});}results.sort((a,b)=>b.sim-a.sim);const out=results.slice(0,k);this._simCache[cacheKey]=out;return out;}
  softNx(a){a=a.toLowerCase();const hard=this.nx[a];const hasHard=hard&&Object.keys(hard).length>0;const resCfg=this.config.resonance||CONFIG.resonance;const softCfg=this.config.soft||CONFIG.soft;let augmented=hasHard?{...hard}:{};let usedResonance=false;const resPartners=this.res[a];if(resPartners){const sorted=Object.entries(resPartners).filter(([w])=>this.nx[w]&&Object.keys(this.nx[w]).length>0).sort((a,b)=>b[1]-a[1]).slice(0,resCfg.topK);for(const[partner]of sorted){const discount=this._resonanceDiscount(a,partner);for(const[next,count]of Object.entries(this.nx[partner])){if(!(next in augmented)){augmented[next]=count*discount;usedResonance=true;}}}}if(Object.keys(augmented).length>0)return{table:augmented,soft:usedResonance&&!hasHard,resonance:usedResonance};const pool=Object.keys(this.nx);if(pool.length===0)return{table:null,soft:false,resonance:false};const similar=this._similarTo(a,pool);if(similar.length===0)return{table:null,soft:false,resonance:false};const merged={};for(const{word,sim}of similar){const nx=this.nx[word];if(!nx)continue;for(const[next,count]of Object.entries(nx)){merged[next]=(merged[next]||0)+count*sim*softCfg.discount;}}return Object.keys(merged).length>0?{table:merged,soft:true,resonance:false}:{table:null,soft:false,resonance:false};}
  softNx2(a,b){a=a.toLowerCase();b=b.toLowerCase();const hard=this.nx2[a]?.[b];if(hard&&Object.keys(hard).length>0)return{table:hard,soft:false,resonance:false};if(!this.nx2[a]){const resA=this.res[a];if(resA){const partners=Object.entries(resA).filter(([w])=>this.nx2[w]).sort((a,b)=>b[1]-a[1]).slice(0,3);for(const[partner]of partners){const ph=this.nx2[partner]?.[b];if(ph&&Object.keys(ph).length>0){const d=this._resonanceDiscount(a,partner);const dd={};for(const[k,v]of Object.entries(ph))dd[k]=v*d;return{table:dd,soft:true,resonance:true};}}}return{table:null,soft:false,resonance:false};}const softCfg=this.config.soft||CONFIG.soft;const resCfg=this.config.resonance||CONFIG.resonance;const pool=Object.keys(this.nx2[a]);const resB=this.res[b];if(resB){const partners=Object.entries(resB).filter(([w])=>pool.includes(w)).sort((a,b)=>b[1]-a[1]).slice(0,resCfg.topK);if(partners.length>0){const merged={};for(const[partner]of partners){const d=this._resonanceDiscount(b,partner);const cont=this.nx2[a][partner];if(!cont)continue;for(const[next,count]of Object.entries(cont)){merged[next]=(merged[next]||0)+count*d;}}if(Object.keys(merged).length>0)return{table:merged,soft:true,resonance:true};}}if(pool.length===0)return{table:null,soft:false,resonance:false};const similar=this._similarTo(b,pool);if(similar.length===0)return{table:null,soft:false,resonance:false};const merged={};for(const{word,sim}of similar){const cont=this.nx2[a][word];if(!cont)continue;for(const[next,count]of Object.entries(cont)){merged[next]=(merged[next]||0)+count*sim*softCfg.discount;}}return Object.keys(merged).length>0?{table:merged,soft:true,resonance:false}:{table:null,soft:false,resonance:false};}

  // ─── Decay & Compact ──────────────────────────────────────────────
  decay(factor=null){const cfg=this.config.decay||CONFIG.decay;const f=factor||cfg.factor;const minC=cfg.minCount;for(const w of Object.keys(this.co)){for(const nb of Object.keys(this.co[w])){this.co[w][nb]*=f;if(this.co[w][nb]<minC)delete this.co[w][nb];}if(Object.keys(this.co[w]).length===0)delete this.co[w];}for(const w of Object.keys(this.nx)){for(const next of Object.keys(this.nx[w])){this.nx[w][next]*=f;if(this.nx[w][next]<minC)delete this.nx[w][next];}if(Object.keys(this.nx[w]).length===0)delete this.nx[w];}for(const w of Object.keys(this.px)){for(const prev of Object.keys(this.px[w])){this.px[w][prev]*=f;if(this.px[w][prev]<minC)delete this.px[w][prev];}if(Object.keys(this.px[w]).length===0)delete this.px[w];}for(const a of Object.keys(this.nx2)){for(const b of Object.keys(this.nx2[a])){for(const c of Object.keys(this.nx2[a][b])){this.nx2[a][b][c]*=f;if(this.nx2[a][b][c]<minC)delete this.nx2[a][b][c];}if(Object.keys(this.nx2[a][b]).length===0)delete this.nx2[a][b];}if(Object.keys(this.nx2[a]).length===0)delete this.nx2[a];}for(const w of Object.keys(this.snx)){for(const t of Object.keys(this.snx[w])){this.snx[w][t]*=f;if(this.snx[w][t]<minC)delete this.snx[w][t];}if(Object.keys(this.snx[w]).length===0)delete this.snx[w];}const resF=Math.sqrt(f);for(const w of Object.keys(this.res)){for(const t of Object.keys(this.res[w])){this.res[w][t]*=resF;if(this.res[w][t]<minC)delete this.res[w][t];}if(Object.keys(this.res[w]).length===0)delete this.res[w];}this._cache={};this._simCache={};this._lastDecay=this.ns;}
  compact(opts={}){const cfg={...((this.config.compact)||CONFIG.compact),...opts};let pruned={nx:0,co:0,snx:0,nx2:0,res:0,vocab:0};for(const w of Object.keys(this.nx)){const b=Object.keys(this.nx[w]).length;if(b>cfg.maxNxPerWord){this.nx[w]=topN(this.nx[w],cfg.maxNxPerWord);pruned.nx+=b-cfg.maxNxPerWord;}}for(const w of Object.keys(this.co)){const b=Object.keys(this.co[w]).length;if(b>cfg.maxCoPerWord){this.co[w]=topN(this.co[w],cfg.maxCoPerWord);pruned.co+=b-cfg.maxCoPerWord;}}for(const w of Object.keys(this.snx)){const b=Object.keys(this.snx[w]).length;if(b>cfg.maxSnxPerWord){this.snx[w]=topN(this.snx[w],cfg.maxSnxPerWord);pruned.snx+=b-cfg.maxSnxPerWord;}}const maxRes=cfg.maxResPerWord||30;for(const w of Object.keys(this.res)){const b=Object.keys(this.res[w]).length;if(b>maxRes){this.res[w]=topN(this.res[w],maxRes);pruned.res+=b-maxRes;}}for(const a of Object.keys(this.nx2)){for(const b of Object.keys(this.nx2[a])){const n=Object.keys(this.nx2[a][b]).length;if(n>cfg.maxNxPerWord){this.nx2[a][b]=topN(this.nx2[a][b],cfg.maxNxPerWord);pruned.nx2+=n-cfg.maxNxPerWord;}}}if(cfg.minFreq>1){const low=new Set(Object.entries(this.wf).filter(([,c])=>c<cfg.minFreq).map(([w])=>w));for(const w of low){delete this.co[w];delete this.nx[w];delete this.px[w];delete this.snx[w];delete this.nx2[w];delete this.res[w];pruned.vocab++;}}this._cache={};this._simCache={};return pruned;}

  // ─── Feed ─────────────────────────────────────────────────────────
  feed(raw){
    const ws=(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>1);
    if(ws.length<2)return 0;
    this.ns++;const sentLen=ws.length,uniqueInSent=new Set(ws).size;
    this._globalSentLen.push(sentLen);if(this._globalSentLen.length>200)this._globalSentLen=this._globalSentLen.slice(-200);
    const skipCfg=this.config.skip||CONFIG.skip;const resCfg=this.config.resonance||CONFIG.resonance;
    for(let i=0;i<ws.length;i++){
      const w=ws[i],rp=i/Math.max(ws.length-1,1);
      this.nt++;const isNew=!(w in this.wf);this.wf[w]=(this.wf[w]||0)+1;
      this._globalFreqSum++;this._globalWordCount=Object.keys(this.wf).length;
      if(isNew)this._reindex(w);
      (this.wp[w]??=[]).push(rp);if(this.wp[w].length>100)this.wp[w]=this.wp[w].slice(-100);
      this._globalPos.push(rp);if(this._globalPos.length>500)this._globalPos=this._globalPos.slice(-500);
      this.co[w]??={};const nb=new Set();
      for(let j=Math.max(0,i-3);j<Math.min(ws.length,i+4);j++){if(j!==i){this.co[w][ws[j]]=(this.co[w][ws[j]]||0)+1;nb.add(ws[j]);}}
      for(let k=0;k<w.length-1;k++)this.bf[w.slice(k,k+2)]=(this.bf[w.slice(k,k+2)]||0)+1;
      this.fs[w]??=this.ns;
      if(w in this.ls){(this.eg[w]??=[]).push(this.ns-this.ls[w]);if(this.eg[w].length>50)this.eg[w]=this.eg[w].slice(-50);}
      this.ls[w]=this.ns;
      (this.ph[w]??=[]).push(rp);if(this.ph[w].length>50)this.ph[w]=this.ph[w].slice(-50);
      if(this._pn[w]&&nb.size){const prev=this._pn[w],union=new Set([...prev,...nb]);(this.nv[w]??=[]).push(1-[...prev].filter(x=>nb.has(x)).length/Math.max(union.size,1));if(this.nv[w].length>50)this.nv[w]=this.nv[w].slice(-50);}
      this._pn[w]=nb;
      (this.sl[w]??=[]).push(sentLen);if(this.sl[w].length>50)this.sl[w]=this.sl[w].slice(-50);
      const mates=ws.filter(x=>x!==w);
      (this.sn[w]??=[]).push(mates.length?mates.filter(m=>(this.wf[m]||0)<3).length/mates.length:0);if(this.sn[w].length>50)this.sn[w]=this.sn[w].slice(-50);
      (this.sd[w]??=[]).push(uniqueInSent/sentLen);if(this.sd[w].length>50)this.sd[w]=this.sd[w].slice(-50);
      if(i<ws.length-1){const next=ws[i+1];this.nx[w]??={};this.nx[w][next]=(this.nx[w][next]||0)+1;}
      if(i>0){const prev=ws[i-1];this.px[w]??={};this.px[w][prev]=(this.px[w][prev]||0)+1;}
      if(i<ws.length-2){const b=ws[i+1],c=ws[i+2];this.nx2[w]??={};this.nx2[w][b]??={};this.nx2[w][b][c]=(this.nx2[w][b][c]||0)+1;}
      const maxJ=Math.min(ws.length,i+skipCfg.window+1);
      for(let j=i+skipCfg.minDist;j<maxJ;j++){const dist=j-i;this.snx[w]??={};this.snx[w][ws[j]]=(this.snx[w][ws[j]]||0)+1/dist;}
      if(i>0){const prev=ws[i-1];const prevNx=this.nx[prev];if(prevNx){const entries=Object.entries(prevNx).sort((a,b)=>b[1]-a[1]).slice(0,resCfg.topK);for(const[other]of entries){if(other!==w)this._addResonance(w,other,resCfg.increment);}}}
      if(i<ws.length-1){const next=ws[i+1];const nextPx=this.px[next];if(nextPx){const entries=Object.entries(nextPx).sort((a,b)=>b[1]-a[1]).slice(0,resCfg.topK);for(const[other]of entries){if(other!==w)this._addResonance(w,other,resCfg.increment);}}}
    }
    this._cache={};this._simCache={};this._updateMedianFreq();
    const decayCfg=this.config.decay||CONFIG.decay;
    if(decayCfg.interval>0&&(this.ns-this._lastDecay)>=decayCfg.interval){this.decay();}
    return ws.length;
  }
  feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0;for(const x of s)tk+=this.feed(x);return{sentences:s.length,tokens:tk};}
  feedBatch(texts){if(!Array.isArray(texts))return{error:"texts must be an array"};let ts=0,tk=0;for(const t of texts){if(typeof t!=="string")continue;const r=this.feedText(t);ts+=r.sentences;tk+=r.tokens;}return{texts:texts.length,sentences:ts,tokens:tk};}

  // ─── WAVE PROPAGATION — now with inhibition ───────────────────────
  // Shared hop-2 logic: only propagate nodes relevant to origin word
  _hop2Selective(w, field, d2, cfg, excite) {
    const originNbs = new Set(Object.keys(this.co[w] || {}));
    const hop1All = [...field.entries()].filter(([k]) => k !== w).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop2Nodes * 2);
    const hop2Nodes = [];
    for (const [node] of hop1All) {
      if (hop2Nodes.length >= cfg.maxHop2Nodes) break;
      const nodeNbs = Object.keys(this.co[node] || {});
      if (nodeNbs.filter(n => originNbs.has(n)).length > 0 || (this.res[w] && node in this.res[w])) hop2Nodes.push(node);
    }
    for (const node of hop2Nodes) {
      const nCo = this.co[node];
      if (nCo) { const entries = Object.entries(nCo).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop2Each); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [nb, cnt] of entries) excite(nb, d2 * cfg.coWeight * cnt / tot); }
      const nNx = this.nx[node];
      if (nNx) { const entries = Object.entries(nNx).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop2Each); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [next, cnt] of entries) excite(next, d2 * cfg.nxWeight * cnt / tot); }
      const nRes = this.res[node];
      if (nRes) { const entries = Object.entries(nRes).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop2Each); for (const [partner] of entries) { const discount = this._resonanceDiscount(node, partner); excite(partner, d2 * cfg.resWeight * discount); } }
    }
  }
  activate(w) {
    w = w.toLowerCase();
    const cfg = this.config.wave || CONFIG.wave;
    const field = new Map();
    const d1 = cfg.decay;
    const d2 = d1 * d1;

    field.set(w, 1.0);
    // Helper: add energy with inhibition applied to the TARGET
    const excite = (target, energy) => {
      const inh = this._inhibitionWeight(target);
      field.set(target, (field.get(target) || 0) + energy * inh);
    };

    // Hop 1
    const co = this.co[w];
    if (co) { const entries = Object.entries(co).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [nb, cnt] of entries) excite(nb, d1 * cfg.coWeight * cnt / tot); }
    const nx = this.nx[w];
    if (nx) { const entries = Object.entries(nx).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [next, cnt] of entries) excite(next, d1 * cfg.nxWeight * cnt / tot); }
    const res = this.res[w];
    if (res) { const entries = Object.entries(res).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); for (const [partner] of entries) { const discount = this._resonanceDiscount(w, partner); excite(partner, d1 * cfg.resWeight * discount); } }
    const snx = this.snx[w];
    if (snx) { const entries = Object.entries(snx).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [target, wt] of entries) excite(target, d1 * cfg.snxWeight * wt / tot); }

    this._hop2Selective(w, field, d2, cfg, excite);
    return field;
  }

  // ─── CONTEXTUAL ACTIVATION — the interpretation mechanism ─────────
  //
  // activate(word) sends a word's FULL wave — all connections fire equally.
  // activateInContext(word, contextField) GATES the wave through context:
  //
  //   For each target the wave would excite:
  //     - If the target is ALREADY in the context field → boost (aligned)
  //     - If the target is NOT in the context field → dampen (unaligned)
  //
  //   The boost/dampen ratio is controlled by config.gate.
  //
  //   This means: "discharge" after "patient summary" amplifies hospital
  //   connections and suppresses wound-fluid connections. Same neuron,
  //   different activation pattern. Context selects meaning.
  //
  //   When contextField is empty/null, falls back to bare activate().
  activateInContext(w, contextField = null) {
    w = w.toLowerCase();

    // No context → bare activation
    if (!contextField || contextField.size === 0) return this.activate(w);

    const cfg = this.config.wave || CONFIG.wave;
    const gateCfg = this.config.gate || CONFIG.gate;
    const field = new Map();
    const d1 = cfg.decay;
    const d2 = d1 * d1;

    field.set(w, 1.0);

    // Contextual excite: modulate energy by context alignment
    const maxCtx = Math.max(...contextField.values(), 1e-9);
    const excite = (target, energy) => {
      const inh = this._inhibitionWeight(target);
      // Context gate: how much does the context field support this target?
      const ctxEnergy = contextField.get(target) || 0;
      const ctxNorm = ctxEnergy / maxCtx; // 0 = not in field, 1 = top of field
      // Gate: aligned targets get boosted, unaligned get dampened
      // gate = dampen + (boost - dampen) * ctxNorm
      // At ctxNorm=0: gate = dampen (e.g. 0.4 = suppress 60%)
      // At ctxNorm=1: gate = boost (e.g. 1.5 = amplify 50%)
      const gate = gateCfg.dampen + (gateCfg.boost - gateCfg.dampen) * ctxNorm;
      field.set(target, (field.get(target) || 0) + energy * inh * gate);
    };

    // Hop 1 (same structure as activate, but using contextual excite)
    const co = this.co[w];
    if (co) { const entries = Object.entries(co).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [nb, cnt] of entries) excite(nb, d1 * cfg.coWeight * cnt / tot); }
    const nx = this.nx[w];
    if (nx) { const entries = Object.entries(nx).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [next, cnt] of entries) excite(next, d1 * cfg.nxWeight * cnt / tot); }
    const res = this.res[w];
    if (res) { const entries = Object.entries(res).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); for (const [partner] of entries) { const discount = this._resonanceDiscount(w, partner); excite(partner, d1 * cfg.resWeight * discount); } }
    const snx = this.snx[w];
    if (snx) { const entries = Object.entries(snx).sort((a, b) => b[1] - a[1]).slice(0, cfg.maxHop1); const tot = entries.reduce((a, [, c]) => a + c, 0); for (const [target, wt] of entries) excite(target, d1 * cfg.snxWeight * wt / tot); }

    // Hop 2 — selective (same relevance filter as activate)
    this._hop2Selective(w, field, d2, cfg, excite);
    return field;
  }

  // ─── SETTLE — iterative convergence to attractor state ────────────
  // Takes a field (Map<string, number>), re-propagates the top nodes,
  // lets interference accumulate, repeats until stable.
  // Returns { field, iterations, converged }.
  settle(initialField) {
    const cfg = this.config.settle || CONFIG.settle;
    let currentField = new Map(initialField);
    let iterations = 0;
    let converged = false;

    for (let iter = 0; iter < cfg.maxIter; iter++) {
      iterations++;
      const prevField = new Map(currentField);
      const energyScale = Math.pow(cfg.iterDecay, iter + 1);

      // Re-activate top nodes — they send new waves that modify the field
      const topNodes = [...currentField.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, cfg.reactivateTop);

      for (const [node, energy] of topNodes) {
        if (!(node in this.wf)) continue; // skip unknown words
        const wave = this.activate(node);
        for (const [target, waveEnergy] of wave) {
          if (target === node) continue; // don't re-excite self
          currentField.set(target, (currentField.get(target) || 0) + waveEnergy * energyScale * (energy / topNodes[0][1]));
        }
      }

      // Check convergence: how much did the field change?
      const delta = 1 - fieldCosine(prevField, currentField);
      if (delta < cfg.epsilon) { converged = true; break; }
    }

    return { field: currentField, iterations, converged };
  }

  // ─── SENTENCE SCORING — forward + backward + settling ─────────────
  //
  // The forward pass blends THREE signals:
  //   1. Wave reinforcement: does the global field already expect this word?
  //   2. Wave synchrony: does this word's wave agree with the global field?
  //   3. Sequential expectation: did the PREVIOUS word predict THIS one? (nx)
  //
  // Signal 3 is DIRECTIONAL — it's the one that knows "doctor→treats" ≠ "treats→doctor".
  // Signals 1&2 are contextual — they know these words BELONG together but not the ORDER.
  // Together they capture both "right words" and "right order."
  //
  // Backward correction is CONSTRAINED by sequential support:
  //   If nx says "this was unexpected," correction can only partially override.
  //   correctedSurprise = surprise * (1 - retroReinforcement * seqGate * strength)
  //   where seqGate = max(seqExpected, minGate). Words with no sequential support
  //   get minimal correction. Words WITH sequential support get full correction.
  scoreSentence(raw) {
    const ws = (raw.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w => w.length > 1);
    if (ws.length < 2) return { words: ws, steps: [], totalSurprise: 0, meanSurprise: 0, coherence: 0, correctedCoherence: 0, settledCoherence: 0, field: new Map(), settled: false };

    const globalField = new Map();
    const steps = [];
    let totalSurprise = 0;

    // ─── FORWARD PASS ─────────────────────────────────────────────
    for (let i = 0; i < ws.length; i++) {
      const w = ws[i];
      const step = { word: w, position: i, known: w in this.wf };

      // Signal 1: wave reinforcement (contextual, symmetric)
      if (i > 0 && globalField.size > 0) {
        const fieldEnergy = globalField.get(w) || 0;
        const maxEnergy = Math.max(...globalField.values(), 1e-9);
        step.reinforcement = Math.min(fieldEnergy / maxEnergy, 1);
      } else { step.reinforcement = 0; }

      // Signal 2: wave synchrony (contextual, symmetric)
      // USE CONTEXTUAL ACTIVATION — the wave is gated by what the field already knows
      const wave = this.activateInContext(w, globalField);
      step.waveSize = wave.size;
      if (i > 0 && globalField.size > 0) {
        step.synchrony = fieldCosine(wave, globalField);
      } else { step.synchrony = 0; }

      // Signal 3: sequential expectation (DIRECTIONAL — carries word order)
      // Uses RELATIVE PREFERENCE, not absolute probability.
      // At scale, most bigrams occur once, so absolute P is uniformly low.
      // But the RANK of the target among all possible next-words still carries signal.
      // If "treats" is the top prediction after "doctor" (even at count=1),
      // seqExpected is high. If "treats" is #20 of 25 predictions, seqExpected is low.
      if (i > 0) {
        const prev = ws[i - 1];
        const nxResult = this.softNx(prev);
        const nxPrev = nxResult.table;
        if (nxPrev && Object.keys(nxPrev).length > 0) {
          const totalNext = Object.values(nxPrev).reduce((a, b) => a + b, 0);
          const absProb = (nxPrev[w] || 0) / Math.max(totalNext, 1);

          // Rank-based: where does this word sit among all predictions?
          const sorted = Object.entries(nxPrev).sort((a, b) => b[1] - a[1]);
          const rank = sorted.findIndex(([word]) => word === w);
          const nEntries = sorted.length;
          // rankScore: 1.0 for top prediction, 0.0 for not-in-list
          const rankScore = rank >= 0 ? 1 - (rank / Math.max(nEntries, 1)) : 0;

          // Blend: absolute probability (rewards high counts) + rank (rewards relative preference)
          step.seqExpected = absProb * 0.4 + rankScore * 0.6;
        } else {
          step.seqExpected = 0;
        }
        step.seqSurprise = 1 - step.seqExpected;
      } else { step.seqExpected = null; step.seqSurprise = 0; }

      // Blend: wave context (40%) + sequential order (35%) + synchrony (25%)
      const waveW = 0.30, seqW = 0.40, syncW = 0.30;
      const waveSurprise = 1 - step.reinforcement;
      const seqS = i > 0 ? step.seqSurprise : 0;
      const syncS = 1 - step.synchrony;
      step.surprise = i > 0 ? waveSurprise * waveW + seqS * seqW + syncS * syncW : 0;

      totalSurprise += step.surprise;
      step.cumulativeSurprise = totalSurprise;
      steps.push(step);

      for (const [node, energy] of wave) {
        globalField.set(node, (globalField.get(node) || 0) + energy);
      }
    }

    const meanSurprise = steps.length ? totalSurprise / steps.length : 0;
    const forwardCoherence = 1 - Math.min(meanSurprise, 1);

    // ─── BACKWARD PASS (error correction, doubly constrained) ────────
    // Gate 1: seqExpected — was this word sequentially predicted?
    // Gate 2: consistency — were the PRECEDING steps coherent?
    //   If most prior steps were low-surprise, this sentence is flowing well,
    //   so correction is trustworthy. If prior steps were all high-surprise,
    //   the sentence is incoherent and correction should not rescue it.
    // Gate 3: retroReinforcement — does the final field validate this word?
    // All three must agree for strong correction.
    const corrCfg = this.config.correction || CONFIG.correction;
    let correctedTotalSurprise = 0;
    const maxFinalEnergy = Math.max(...globalField.values(), 1e-9);
    const minGate = 0.05;

    for (let si = 0; si < steps.length; si++) {
      const step = steps[si];
      if (step.position === 0) { step.correctedSurprise = 0; correctedTotalSurprise += 0; continue; }

      const finalEnergy = globalField.get(step.word) || 0;
      const retroReinforcement = Math.min(finalEnergy / maxFinalEnergy, 1);
      step.retroReinforcement = retroReinforcement;

      // Consistency: what fraction of PRECEDING steps had surprise < 0.7?
      const priorSteps = steps.slice(1, si); // skip position 0
      const coherentCount = priorSteps.filter(s => s.surprise < 0.7).length;
      const consistency = priorSteps.length > 0 ? coherentCount / priorSteps.length : 0.5;
      step.consistency = consistency;

      // Combined gate: all three signals must agree
      const seqGate = Math.max(step.seqExpected || 0, minGate);
      const combinedGate = seqGate * (0.4 + consistency * 0.6); // consistency modulates seq
      step.correctedSurprise = step.surprise * (1 - retroReinforcement * combinedGate * corrCfg.strength);
      correctedTotalSurprise += step.correctedSurprise;
    }
    const correctedMeanSurprise = steps.length ? correctedTotalSurprise / steps.length : 0;
    const correctedCoherence = 1 - Math.min(correctedMeanSurprise, 1);

    // ─── SETTLING ─────────────────────────────────────────────────
    const settleResult = this.settle(globalField);

    // Settled coherence: corrected coherence is the primary signal.
    // Settling adjusts via field concentration — how focused the attractor is.
    // But concentration alone isn't meaning, so it only nudges the score.
    const settledValues = [...settleResult.field.values()];
    const sortedE = settledValues.sort((a, b) => b - a);
    const totalE = sortedE.reduce((a, b) => a + b, 0);
    const top20pct = sortedE.slice(0, Math.max(1, Math.floor(sortedE.length * 0.2)));
    const concentration = totalE > 0 ? top20pct.reduce((a, b) => a + b, 0) / totalE : 0;
    // Settled = 80% corrected + 20% concentration nudge
    const settledCoherence = correctedCoherence * 0.8 + concentration * 0.2;

    return {
      words: ws,
      steps,
      totalSurprise,
      meanSurprise,
      coherence: forwardCoherence,
      correctedCoherence,
      settledCoherence,
      field: settleResult.field,
      settled: settleResult.converged,
      settleIterations: settleResult.iterations,
    };
  }

  sentenceField(raw) { return this.scoreSentence(raw).field; }
  compareSentences(a, b) { const fA = this.sentenceField(a); const fB = this.sentenceField(b); return { similarity: fieldCosine(fA, fB), sizeA: fA.size, sizeB: fB.size, overlap: [...fA.keys()].filter(k => fB.has(k)).length }; }

  // ─── Vec methods (unchanged) ──────────────────────────────────────
  formVec(w){w=w.toLowerCase();const n=w.length;if(n<1)return new Float64Array(16);const f=new Float64Array(16),ch=Array.from(w).map(c=>V.has(c)?1:-1);if(n>=2)f[0]=ch.reduce((a,c,i)=>i>0&&c!==ch[i-1]?a+1:a,0)/(n-1);let mx=0,r=0;for(const c of ch){if(c<0){r++;mx=Math.max(mx,r);}else r=0;}f[1]=mx/n;mx=0;r=0;for(const c of ch){if(c>0){r++;mx=Math.max(mx,r);}else r=0;}f[2]=mx/n;f[3]=ch[0];f[4]=ch[n-1];if(n>=3){const t=Math.floor(n/3);f[5]=mn(ch.slice(2*t))-mn(ch.slice(0,t));}f[6]=ch.filter(c=>c>0).length/n;if(n>=2){const mu=mn(ch);f[7]=Math.sqrt(mn(ch.map(c=>(c-mu)**2)));}if(n>=2){let cv=0,vc=0,cc=0,vv=0;for(let i=0;i<n-1;i++){const a=V.has(w[i]),b=V.has(w[i+1]);if(!a&&b)cv++;else if(a&&!b)vc++;else if(!a&&!b)cc++;else vv++;}const t=n-1;f[8]=cv/t;f[9]=vc/t;f[10]=cc/t;f[11]=vv/t;}if(n>=3){let cvc=0,vcv=0;const iv=Array.from(w).map(c=>V.has(c));for(let i=0;i<n-2;i++){if(!iv[i]&&iv[i+1]&&!iv[i+2])cvc++;if(iv[i]&&!iv[i+1]&&iv[i+2])vcv++;}f[12]=cvc/(n-2);f[13]=vcv/(n-2);}f[14]=new Set(w).size/n;f[15]=Math.min(n/15,1);return f;}
  contextVec(w){w=w.toLowerCase();const s=new Float64Array(12),co=this.co[w];if(!co)return s;const ent=Object.entries(co).sort((a,b)=>b[1]-a[1]),tot=ent.reduce((a,[,c])=>a+c,0),nN=ent.length;s[0]=Math.min(nN/30,1);s[1]=ent[0]?ent[0][1]/tot:0;s[2]=ent.slice(0,3).reduce((a,[,c])=>a+c,0)/(tot||1);const tf=ent.slice(0,5).map(([nb])=>Math.log2((this.wf[nb]||0)+1));if(tf.length){s[3]=mn(tf)/10;s[4]=tf.length>1?sd(tf)/5:0;}const np=[];for(const[nb]of ent.slice(0,10)){const p=this.wp[nb];if(p)np.push(...p.slice(-10));}if(np.length){s[5]=mn(np);s[6]=np.length>1?sd(np):0;}const mp=this.wp[w];if(mp?.length){s[7]=mn(mp);s[8]=mp.length>1?sd(mp):0;}if(nN>=2){const tn=ent[0][0],tc=this.co[tn]||{};const mk=new Set(Object.keys(co)),tk=new Set(Object.keys(tc)),un=new Set([...mk,...tk]);s[9]=1-[...mk].filter(x=>tk.has(x)).length/Math.max(un.size,1);}s[10]=Math.min(Math.log2((this.wf[w]||0)+1)/10,1);s[11]=tot/Math.max(this.wf[w]||1,1)/10;return s;}
  historyVec(w){w=w.toLowerCase();const s=new Float64Array(8),cnt=this.wf[w]||0;if(!cnt)return s;s[0]=Math.min(Math.log2(cnt+1)/10,1);s[1]=Math.min((this.ns-(this.fs[w]||this.ns)+1)/(this.ns||1),1);s[2]=Math.max(0,1-(this.ns-(this.ls[w]||0))/(this.ns||1));const g=this.eg[w]||[];if(g.length>=2)s[3]=1-Math.min(sd(g)/(mn(g)+1e-3),2)/2;else if(g.length===1)s[3]=.5;const ph=this.ph[w]||[];if(ph.length>=2)s[4]=1-Math.min(sd(ph)*3,1);if(ph.length)s[5]=mn(ph);const v=this.nv[w]||[];if(v.length)s[6]=mn(v);s[7]=Math.min(cnt/10,1);return s;}
  influenceVec(w){w=w.toLowerCase();const s=new Float64Array(8),co=this.co[w];if(!co)return s;const sl=this.sl[w]||[];if(sl.length)s[0]=Math.min(mn(sl)/15,1);if(sl.length>=2)s[1]=Math.min(sd(sl)/5,1);const sn=this.sn[w]||[];if(sn.length)s[2]=mn(sn);const sDiv=this.sd[w]||[];if(sDiv.length)s[3]=mn(sDiv);const nbs=Object.keys(co);if(nbs.length>=3){let tri=0,pairs=0;const sample=nbs.slice(0,10);for(let i=0;i<sample.length;i++)for(let j=i+1;j<sample.length;j++){pairs++;if(this.co[sample[i]]&&sample[j] in this.co[sample[i]])tri++;}s[4]=pairs?1-tri/pairs:0;}const pos=this.wp[w]||[];if(pos.length)s[5]=Math.abs(mn(pos)-0.5)*2;if(nbs.length){const nf=nbs.slice(0,10).map(n=>Math.log2((this.wf[n]||0)+1));s[6]=Math.min(sd(nf)/3,1);}if(sl.length>=4){const h=Math.floor(sl.length/2);const e=mn(sl.slice(0,h)),l=mn(sl.slice(h));s[7]=e>0?1-Math.min(Math.abs(l-e)/e,1):0;}return s;}
  contrastVec(w){w=w.toLowerCase();const s=new Float64Array(8),cnt=this.wf[w]||0;if(!cnt||!this._globalWordCount)return s;const avgFreq=this._globalFreqSum/Math.max(this._globalWordCount,1);s[0]=avgFreq>0?Math.min(Math.abs(cnt-avgFreq)/avgFreq,2)/2:0;const wp=this.wp[w]||[],gp=this._globalPos;if(wp.length&&gp.length)s[1]=Math.min(Math.abs(mn(wp)-mn(gp))*3,1);if(wp.length>=2&&gp.length>=2)s[2]=Math.min(Math.abs(sd(wp)-sd(gp))*5,1);const sl=this.sl[w]||[],gsl=this._globalSentLen;if(sl.length&&gsl.length)s[3]=Math.min(Math.abs(mn(sl)-mn(gsl))/Math.max(mn(gsl),1),1);const nbs=Object.keys(this.co[w]||{}).length;const avgNbs=Object.values(this.co).reduce((a,c)=>a+Object.keys(c).length,0)/Math.max(this._globalWordCount,1);s[4]=avgNbs>0?Math.min(Math.abs(nbs-avgNbs)/avgNbs,2)/2:0;const co=this.co[w];if(co){const myNbs=new Set(Object.keys(co));const topWord=Object.entries(this.wf).sort((a,b)=>b[1]-a[1])[0];if(topWord){const topNbs=new Set(Object.keys(this.co[topWord[0]]||{}));const union=new Set([...myNbs,...topNbs]);s[5]=union.size?1-[...myNbs].filter(x=>topNbs.has(x)).length/union.size:0;}}const maxFreq=Math.max(...Object.values(this.wf),1);s[6]=1-Math.min(cnt/maxFreq,1);if(wp.length>=6){const h=Math.floor(wp.length/2);const eSD=sd(wp.slice(0,h)),lSD=sd(wp.slice(h));s[7]=eSD>0?Math.max(0,1-lSD/eSD):0;}return s;}
  expectationVec(w){w=w.toLowerCase();const s=new Float64Array(8);const nx=this.nx[w],px=this.px[w];if(!nx&&!px)return s;if(nx){const ent=Object.entries(nx);const tot=ent.reduce((a,[,c])=>a+c,0);const top=ent.sort((a,b)=>b[1]-a[1])[0];s[0]=top?top[1]/tot:0;}if(nx)s[1]=Math.min(Object.keys(nx).length/15,1);if(px){const ent=Object.entries(px);const tot=ent.reduce((a,[,c])=>a+c,0);const top=ent.sort((a,b)=>b[1]-a[1])[0];s[2]=top?top[1]/tot:0;}if(px)s[3]=Math.min(Object.keys(px).length/15,1);s[4]=Math.abs((s[0]||0)-(s[2]||0));const fwdN=nx?Object.keys(nx).length:0;const bwdN=px?Object.keys(px).length:0;s[5]=Math.min((fwdN+bwdN)/20,1);if(nx&&px){const fwd=new Set(Object.keys(nx)),bwd=new Set(Object.keys(px));const union=new Set([...fwd,...bwd]);const inter=[...fwd].filter(x=>bwd.has(x)).length;s[6]=union.size?1-inter/union.size:0;}if(nx){const ent=Object.entries(nx).sort((a,b)=>b[1]-a[1]);const tot=ent.reduce((a,[,c])=>a+c,0);const top3=ent.slice(0,3).reduce((a,[,c])=>a+c,0);s[7]=tot?top3/tot:0;}return s;}
  vec(w){const k=w.toLowerCase().trim();if(this._cache[k])return this._cache[k];const v=[...this.formVec(k),...this.contextVec(k),...this.historyVec(k),...this.influenceVec(k),...this.contrastVec(k),...this.expectationVec(k)];this._cache[k]=v;return v;}
  compare(a,b,p="meaning",mask=null){const va=this.vec(a),vb=this.vec(b);const sc={form:cos(va,vb,...IDX.form),context:cos(va,vb,...IDX.ctx),history:cos(va,vb,...IDX.hist),influence:cos(va,vb,...IDX.inf),contrast:cos(va,vb,...IDX.con),expectation:cos(va,vb,...IDX.exp),ocr:Math.max(0,1-ocrDist(a,b)/Math.max(a.length,b.length,1))};sc.full=cos(va,vb,...IDX.all);const al=a.toLowerCase(),bl=b.toLowerCase();const nxAR=this.softNx(al);const nxBR=this.softNx(bl);const nxA=nxAR.table,nxB=nxBR.table;sc.expectsAB=nxA?((nxA[bl]||0)/Math.max(Object.values(nxA).reduce((a,b)=>a+b,0),1)):0;sc.expectsBA=nxB?((nxB[al]||0)/Math.max(Object.values(nxB).reduce((a,b)=>a+b,0),1)):0;sc.directional=Math.abs(sc.expectsAB-sc.expectsBA);sc.softExpectation=nxAR.soft||nxBR.soft;const n2AB=this.softNx2(al,bl);const n2BA=this.softNx2(bl,al);sc.trajectoryAB=n2AB.table?Math.min(Object.keys(n2AB.table).length/5,1):0;sc.trajectoryBA=n2BA.table?Math.min(Object.keys(n2BA.table).length/5,1):0;sc.softTrajectory=n2AB.soft||n2BA.soft;const snxA=this.snx[al],snxB=this.snx[bl];const stA=snxA?Object.values(snxA).reduce((a,b)=>a+b,0):0;const stB=snxB?Object.values(snxB).reduce((a,b)=>a+b,0):0;sc.longRangeAB=snxA?(snxA[bl]||0)/Math.max(stA,1):0;sc.longRangeBA=snxB?(snxB[al]||0)/Math.max(stB,1):0;sc.longRangeAsymmetry=Math.abs(sc.longRangeAB-sc.longRangeBA);sc.resonance=this.res[al]?.[bl]||0;sc.resonanceDiscount=this._resonanceDiscount(al,bl);sc.usedResonance=nxAR.resonance||nxBR.resonance||n2AB.resonance||n2BA.resonance;const waveA=this.activate(al);const waveB=this.activate(bl);sc.waveOverlap=fieldCosine(waveA,waveB);const rt=this.config.routing[p]||{form:.14,context:.14,history:.14,influence:.14,contrast:.14,expectation:.14,ocr:.14};let wt={...rt};if(mask){for(const ch of["form","context","history","influence","contrast","expectation","ocr"])if(!mask[ch])wt[ch]=0;const s=Object.values(wt).reduce((a,b)=>a+b,0);if(s>0)for(const ch in wt)wt[ch]/=s;}const routed=wt.form*sc.form+wt.context*sc.context+wt.history*sc.history+(wt.influence||0)*sc.influence+(wt.contrast||0)*sc.contrast+(wt.expectation||0)*sc.expectation+wt.ocr*sc.ocr;return{routed,...sc,weights:wt};}
  correct(g,k=5){const cands=this._candidates(g);const out=cands.map(w=>({word:w,...this.compare(g,w,"correction")}));out.sort((a,b)=>b.routed-a.routed||(this.wf[b.word]||0)-(this.wf[a.word]||0));const top=out.slice(0,k);const conf=top.length>=2?top[0].routed-top[1].routed:top.length?1:0;return{candidates:top,confidence:conf,reject:conf<this.config.thresholds.confidence.reject,lowConfidence:conf<this.config.thresholds.confidence.low};}
  similar(w,k=10){const key=w.toLowerCase(),out=[];for(const c of Object.keys(this.wf)){if(c===key)continue;out.push({word:c,...this.compare(w,c,"meaning")});}return out.sort((a,b)=>b.routed-a.routed).slice(0,k);}
  stats(){const v=Object.keys(this.wf);return{version:this.version,sentences:this.ns,tokens:this.nt,vocabulary:v.length,bigrams:Object.keys(this.bf).length,cooccurrences:Object.values(this.co).reduce((a,v)=>a+Object.keys(v).length,0),mature:v.filter(w=>this.wf[w]>=10).length,transitions:Object.values(this.nx).reduce((a,v)=>a+Object.keys(v).length,0),trajectories:Object.values(this.nx2).reduce((a,v)=>a+Object.values(v).reduce((b,c)=>b+Object.keys(c).length,0),0),skipGrams:Object.values(this.snx).reduce((a,v)=>a+Object.keys(v).length,0),resonancePairs:Object.values(this.res).reduce((a,v)=>a+Object.keys(v).length,0)/2,medianFreq:this._medianFreq,lastDecay:this._lastDecay};}
  serialize(){const tr=(o,n)=>Object.fromEntries(Object.entries(o).map(([k,v])=>[k,Array.isArray(v)?v.slice(-n):v]));return JSON.stringify({version:this.version,config:this.config,wf:this.wf,co:this.co,wp:tr(this.wp,50),bf:this.bf,fs:this.fs,ls:this.ls,eg:tr(this.eg,30),ph:tr(this.ph,30),nv:tr(this.nv,20),sl:tr(this.sl,30),sn:tr(this.sn,30),sd:tr(this.sd,30),nx:this.nx,px:this.px,nx2:this.nx2,snx:this.snx,res:this.res,gp:this._globalPos.slice(-200),gsl:this._globalSentLen.slice(-200),gfs:this._globalFreqSum,gwc:this._globalWordCount,pn:Object.fromEntries(Object.entries(this._pn).map(([k,v])=>[k,[...v]])),ns:this.ns,nt:this.nt,lastDecay:this._lastDecay});}
  static deserialize(json){const d=JSON.parse(json);const e=new ShifuEngine(d.config||CONFIG);e.wf=d.wf||{};e.co=d.co||{};e.wp=d.wp||{};e.bf=d.bf||{};e.fs=d.fs||{};e.ls=d.ls||{};e.eg=d.eg||{};e.ph=d.ph||{};e.nv=d.nv||{};e.sl=d.sl||{};e.sn=d.sn||{};e.sd=d.sd||{};e.nx=d.nx||{};e.px=d.px||{};e.nx2=d.nx2||{};e.snx=d.snx||{};e.res=d.res||{};e._globalPos=d.gp||[];e._globalSentLen=d.gsl||[];e._globalFreqSum=d.gfs||0;e._globalWordCount=d.gwc||0;if(d.pn)e._pn=Object.fromEntries(Object.entries(d.pn).map(([k,v])=>[k,new Set(v)]));e.ns=d.ns||0;e.nt=d.nt||0;e._lastDecay=d.lastDecay||0;for(const w of Object.keys(e.wf))e._reindex(w);e._updateMedianFreq();return e;}
}
module.exports={ShifuEngine,CONFIG,VERSION,ocrDist,levDist,cos,cosVec,fieldCosine,mn,sd,prng,shuffle,IDX};
