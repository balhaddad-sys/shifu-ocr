// SHIFU CORE ENGINE v1.4.1
// A stone etched by exposure.
//
// The baby state is Form — 16 dimensions of structure before any data.
// The stone before etching. Every word has shape before it has meaning.
//
// feed() is the perturbation. Text hits the stone. Each sentence etches:
// co-occurrence neighborhoods deepen, positional patterns wear grooves,
// sequential expectations form channels, the self-model baseline shifts.
//
// The stone doesn't compute language. Language is the pattern of etchings
// left by exposure. The 7 channels aren't modules — they're faces of the
// same stone showing different marks from the same perturbation:
//
//   Form (16D)        — the stone before etching
//   Context (12D)     — which atoms got displaced together
//   History (8D)      — how deep the groove, how recently cut
//   Influence (8D)    — how the etching changed the surrounding surface
//   Contrast (8D)     — how this groove differs from the stone's average
//   Expectation (8D)  — which direction the groove runs
//   Affinity          — which unetched regions will accept the chisel easiest
//
// Convergence at 99.93% is the stone finding steady state after perturbation.
// Same chisel strokes in different order produce the same etchings —
// because the stone's grain determines where marks can form.

const VERSION = "1.4.1";
const CONFIG = {
  version: VERSION,
  channels: { form: 16, context: 12, history: 8, influence: 8, contrast: 8, expectation: 8 },
  routing: {
    correction: { ocr: .7, form: .3, context: 0, history: 0, influence: 0, contrast: 0, expectation: 0 },
    meaning:    { form: .20, context: .20, history: .10, influence: .15, contrast: .15, expectation: .20, ocr: 0 },
  },
  ocr: {"0,o":.1,"1,l":.2,"1,i":.2,"5,s":.3,"8,b":.3,"6,g":.4,"l,i":.2,"m,n":.4,"u,v":.5,"c,e":.5,"r,n":.3,"d,o":.3,"f,t":.4,"h,b":.4,"a,e":.4,"a,o":.4,"u,n":.4,"e,i":.4,"f,l":.4,"s,e":.5,"b,d":.4},
  thresholds: { confidence: { reject: 0.02, low: 0.05 } },
};
const IDX = { form:[0,16], ctx:[16,28], hist:[28,36], inf:[36,44], con:[44,52], exp:[52,60], all:[0,60] };
const V = new Set("aeiou");
const mn=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0;
const sd=a=>{const m=mn(a);return Math.sqrt(mn(a.map(v=>(v-m)**2)));};
function cos(a,b,lo,hi){let d=0,na=0,nb=0;for(let i=lo;i<hi;i++){d+=a[i]*b[i];na+=a[i]**2;nb+=b[i]**2;}na=Math.sqrt(na);nb=Math.sqrt(nb);return na<1e-6||nb<1e-6?0:d/(na*nb);}
function prng(seed){let s=seed;return()=>{s=(s*16807)%2147483647;return s/2147483647;};}
function shuffle(arr,rand){const a=[...arr];for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function ocrDist(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return ocrDist(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++){const k=[a[i],b[j]].sort().join(",");c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:(CONFIG.ocr[k]??1))));}p=c;}return p[b.length];}
function levDist(a,b){a=a.toLowerCase();b=b.toLowerCase();if(a.length<b.length)return levDist(b,a);if(!b.length)return a.length;let p=Array.from({length:b.length+1},(_,i)=>i);for(let i=0;i<a.length;i++){const c=[i+1];for(let j=0;j<b.length;j++)c.push(Math.min(p[j+1]+1,c[j]+1,p[j]+(a[i]===b[j]?0:1)));p=c;}return p[b.length];}

class ShifuEngine {
  // The stone. Born with grain (config) but no etchings.
  // Every property below is either part of the grain (config, thresholds)
  // or a surface that will be etched by feed() (everything else).
  constructor(cfg=CONFIG){
    this.config=cfg;this.version=cfg.version;
    // ── Surface marks (etched by exposure) ──
    this.wf={};   // word frequency — how deep each groove
    this.co={};   // co-occurrence — which atoms displaced together
    this.wp={};   // word positions — where on the surface each mark falls
    this.bf={};   // bigram frequency — sub-groove patterns
    this.fs={};   // first seen — when each groove was first cut
    this.ls={};   // last seen — when each groove was last deepened
    this.eg={};   // encounter gaps — rhythm of re-etching
    this.ph={};   // position history — how the groove's location drifts
    this.nv={};   // neighbor volatility — how the surrounding surface changes
    this._pn={};  // previous neighbors — last observed neighborhood
    this.sl={};   // sentence lengths — how wide each chisel stroke was
    this.sn={};   // sentence novelty — how much new material each stroke carried
    this.sd={};   // sentence diversity — how varied each stroke was
    // ── Self-model: the stone's knowledge of its own texture ──
    this._globalPos=[];this._globalSentLen=[];this._globalFreqSum=0;this._globalWordCount=0;
    // ── Directional grooves: which way each channel runs ──
    this.nx={};   // next-word expectations — the groove's forward direction
    this.px={};   // prev-word expectations — the groove's backward direction
    this.nx2={};  // second-order trajectory — where the groove goes after the next turn
    // ── Bookkeeping ──
    this.ns=0;this.nt=0;this._cache={};this._idxLen={};this._idxBg={};
  }

  _reindex(w){const l=w.length;(this._idxLen[l]??=[]);if(!this._idxLen[l].includes(w))this._idxLen[l].push(w);for(let i=0;i<w.length-1;i++){const bg=w.slice(i,i+2);(this._idxBg[bg]??=[]);if(!this._idxBg[bg].includes(w))this._idxBg[bg].push(w);}}
  _candidates(g,r=2){g=g.toLowerCase();const c=new Set();for(let d=-r;d<=r;d++){const ws=this._idxLen[g.length+d];if(ws)for(const w of ws)c.add(w);}for(let i=0;i<g.length-1;i++){const ws=this._idxBg[g.slice(i,i+2)];if(ws)for(const w of ws)c.add(w);}return[...c];}

  // ─── The perturbation: text hits the stone ──────────────────────────
  // A sentence is a chisel stroke. Each word in the sentence:
  //   - deepens its own groove (wf)
  //   - displaces its neighbors (co)
  //   - records where on the surface it fell (wp, ph)
  //   - notes when it was last struck (fs, ls, eg)
  //   - observes how its neighborhood shifted (nv)
  //   - measures the width and novelty of this stroke (sl, sn, sd)
  //   - extends its directional channel (nx, px, nx2)
  // After the stroke, the cache is cleared — the stone's surface has changed,
  // so all previously computed readings are stale.
  feed(raw){
    const ws=(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>=1);
    if(ws.length<2)return{tokens:0,newWords:0,newConnections:0,novelty:0};
    const beforeVocab=Object.keys(this.wf).length;
    const beforeConnections=Object.values(this.co).reduce((a,c)=>a+Object.keys(c).length,0);
    this.ns++;
    const sentLen=ws.length,uniqueInSent=new Set(ws).size;
    this._globalSentLen.push(sentLen);if(this._globalSentLen.length>200)this._globalSentLen=this._globalSentLen.slice(-200);

    for(let i=0;i<ws.length;i++){
      const w=ws[i],rp=i/Math.max(ws.length-1,1);
      this.nt++;
      const isNew=!(w in this.wf);
      this.wf[w]=(this.wf[w]||0)+1;
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

      // Directional grooves: which way does the channel run from here?
      if(i<ws.length-1){const next=ws[i+1];this.nx[w]??={};this.nx[w][next]=(this.nx[w][next]||0)+1;}
      if(i>0){const prev=ws[i-1];this.px[w]??={};this.px[w][prev]=(this.px[w][prev]||0)+1;}
      // Second-order: given (w → next), where does the groove go after the turn?
      if(i<ws.length-2){const b=ws[i+1],c=ws[i+2];this.nx2[w]??={};this.nx2[w][b]??={};this.nx2[w][b][c]=(this.nx2[w][b][c]||0)+1;}
    }
    this._cache={};
    const afterVocab=Object.keys(this.wf).length;
    const afterConnections=Object.values(this.co).reduce((a,c)=>a+Object.keys(c).length,0);
    const newWords=afterVocab-beforeVocab;
    const newConnections=afterConnections-beforeConnections;
    const novelty=newWords/Math.max(ws.length,1);
    return{tokens:ws.length,newWords,newConnections,novelty};
  }

  feedText(t){const s=t.split(/[.!?\n]+/).map(s=>s.trim()).filter(s=>s.length>5);let tk=0,nw=0,nc=0;for(const x of s){const r=this.feed(x);tk+=r.tokens;nw+=r.newWords;nc+=r.newConnections;}return{sentences:s.length,tokens:tk,newWords:nw,newConnections:nc,novelty:nw/Math.max(tk,1)};}

  // ─── The stone before etching ──────────────────────────────────────
  formVec(w){w=w.toLowerCase();const n=w.length;if(n<1)return new Float64Array(16);const f=new Float64Array(16),ch=Array.from(w).map(c=>V.has(c)?1:-1);if(n>=2)f[0]=ch.reduce((a,c,i)=>i>0&&c!==ch[i-1]?a+1:a,0)/(n-1);let mx=0,r=0;for(const c of ch){if(c<0){r++;mx=Math.max(mx,r);}else r=0;}f[1]=mx/n;mx=0;r=0;for(const c of ch){if(c>0){r++;mx=Math.max(mx,r);}else r=0;}f[2]=mx/n;f[3]=ch[0];f[4]=ch[n-1];if(n>=3){const t=Math.floor(n/3);f[5]=mn(ch.slice(2*t))-mn(ch.slice(0,t));}f[6]=ch.filter(c=>c>0).length/n;if(n>=2){const mu=mn(ch);f[7]=Math.sqrt(mn(ch.map(c=>(c-mu)**2)));}if(n>=2){let cv=0,vc=0,cc=0,vv=0;for(let i=0;i<n-1;i++){const a=V.has(w[i]),b=V.has(w[i+1]);if(!a&&b)cv++;else if(a&&!b)vc++;else if(!a&&!b)cc++;else vv++;}const t=n-1;f[8]=cv/t;f[9]=vc/t;f[10]=cc/t;f[11]=vv/t;}if(n>=3){let cvc=0,vcv=0;const iv=Array.from(w).map(c=>V.has(c));for(let i=0;i<n-2;i++){if(!iv[i]&&iv[i+1]&&!iv[i+2])cvc++;if(iv[i]&&!iv[i+1]&&iv[i+2])vcv++;}f[12]=cvc/(n-2);f[13]=vcv/(n-2);}f[14]=new Set(w).size/n;f[15]=Math.min(n/15,1);return f;}
  // ─── Which atoms displaced together ────────────────────────────────
  contextVec(w){w=w.toLowerCase();const s=new Float64Array(12),co=this.co[w];if(!co)return s;const ent=Object.entries(co).sort((a,b)=>b[1]-a[1]),tot=ent.reduce((a,[,c])=>a+c,0),nN=ent.length;s[0]=Math.min(nN/30,1);s[1]=ent[0]?ent[0][1]/tot:0;s[2]=ent.slice(0,3).reduce((a,[,c])=>a+c,0)/(tot||1);const tf=ent.slice(0,5).map(([nb])=>Math.log2((this.wf[nb]||0)+1));if(tf.length){s[3]=mn(tf)/10;s[4]=tf.length>1?sd(tf)/5:0;}const np=[];for(const[nb]of ent.slice(0,10)){const p=this.wp[nb];if(p)np.push(...p.slice(-10));}if(np.length){s[5]=mn(np);s[6]=np.length>1?sd(np):0;}const mp=this.wp[w];if(mp?.length){s[7]=mn(mp);s[8]=mp.length>1?sd(mp):0;}if(nN>=2){const tn=ent[0][0],tc=this.co[tn]||{};const mk=new Set(Object.keys(co)),tk=new Set(Object.keys(tc)),un=new Set([...mk,...tk]);s[9]=1-[...mk].filter(x=>tk.has(x)).length/Math.max(un.size,1);}s[10]=Math.min(Math.log2((this.wf[w]||0)+1)/10,1);s[11]=tot/Math.max(this.wf[w]||1,1)/10;return s;}
  // ─── How deep the groove, how recently cut ─────────────────────────
  historyVec(w){w=w.toLowerCase();const s=new Float64Array(8),cnt=this.wf[w]||0;if(!cnt)return s;s[0]=Math.min(Math.log2(cnt+1)/10,1);s[1]=Math.min((this.ns-(this.fs[w]||this.ns)+1)/(this.ns||1),1);s[2]=Math.max(0,1-(this.ns-(this.ls[w]||0))/(this.ns||1));const g=this.eg[w]||[];if(g.length>=2)s[3]=1-Math.min(sd(g)/(mn(g)+1e-3),2)/2;else if(g.length===1)s[3]=.5;const ph=this.ph[w]||[];if(ph.length>=2)s[4]=1-Math.min(sd(ph)*3,1);if(ph.length)s[5]=mn(ph);const v=this.nv[w]||[];if(v.length)s[6]=mn(v);s[7]=Math.min(cnt/10,1);return s;}
  // ─── How the etching changed the surrounding surface ───────────────
  influenceVec(w){w=w.toLowerCase();const s=new Float64Array(8),co=this.co[w];if(!co)return s;const sl=this.sl[w]||[];if(sl.length)s[0]=Math.min(mn(sl)/15,1);if(sl.length>=2)s[1]=Math.min(sd(sl)/5,1);const sn=this.sn[w]||[];if(sn.length)s[2]=mn(sn);const sDiv=this.sd[w]||[];if(sDiv.length)s[3]=mn(sDiv);const nbs=Object.keys(co);if(nbs.length>=3){let tri=0,pairs=0;const sample=nbs.slice(0,10);for(let i=0;i<sample.length;i++)for(let j=i+1;j<sample.length;j++){pairs++;if(this.co[sample[i]]&&sample[j] in this.co[sample[i]])tri++;}s[4]=pairs?1-tri/pairs:0;}const pos=this.wp[w]||[];if(pos.length)s[5]=Math.abs(mn(pos)-0.5)*2;if(nbs.length){const nf=nbs.slice(0,10).map(n=>Math.log2((this.wf[n]||0)+1));s[6]=Math.min(sd(nf)/3,1);}if(sl.length>=4){const h=Math.floor(sl.length/2);const e=mn(sl.slice(0,h)),l=mn(sl.slice(h));s[7]=e>0?1-Math.min(Math.abs(l-e)/e,1):0;}return s;}
  // ─── How this groove differs from the stone's average ──────────────
  contrastVec(w){w=w.toLowerCase();const s=new Float64Array(8),cnt=this.wf[w]||0;if(!cnt||!this._globalWordCount)return s;const avgFreq=this._globalFreqSum/Math.max(this._globalWordCount,1);s[0]=avgFreq>0?Math.min(Math.abs(cnt-avgFreq)/avgFreq,2)/2:0;const wp=this.wp[w]||[],gp=this._globalPos;if(wp.length&&gp.length)s[1]=Math.min(Math.abs(mn(wp)-mn(gp))*3,1);if(wp.length>=2&&gp.length>=2)s[2]=Math.min(Math.abs(sd(wp)-sd(gp))*5,1);const sl=this.sl[w]||[],gsl=this._globalSentLen;if(sl.length&&gsl.length)s[3]=Math.min(Math.abs(mn(sl)-mn(gsl))/Math.max(mn(gsl),1),1);const nbs=Object.keys(this.co[w]||{}).length;const avgNbs=Object.values(this.co).reduce((a,c)=>a+Object.keys(c).length,0)/Math.max(this._globalWordCount,1);s[4]=avgNbs>0?Math.min(Math.abs(nbs-avgNbs)/avgNbs,2)/2:0;const co=this.co[w];if(co){const myNbs=new Set(Object.keys(co));const topWord=Object.entries(this.wf).sort((a,b)=>b[1]-a[1])[0];if(topWord){const topNbs=new Set(Object.keys(this.co[topWord[0]]||{}));const union=new Set([...myNbs,...topNbs]);s[5]=union.size?1-[...myNbs].filter(x=>topNbs.has(x)).length/union.size:0;}}const maxFreq=Math.max(...Object.values(this.wf),1);s[6]=1-Math.min(cnt/maxFreq,1);if(wp.length>=6){const h=Math.floor(wp.length/2);const eSD=sd(wp.slice(0,h)),lSD=sd(wp.slice(h));s[7]=eSD>0?Math.max(0,1-lSD/eSD):0;}return s;}

  // ─── Which direction the groove runs ───────────────────────────────
  // The brain reads "doctor" and expects {treats, prescribed, examined, ordered...}
  // That set IS the directional structure. No parsing. No roles.
  // "Doctor → treats" is expected. "Treats → doctor" is not.
  expectationVec(w){
    w=w.toLowerCase();const s=new Float64Array(8);
    const nx=this.nx[w],px=this.px[w];
    if(!nx&&!px)return s;

    // 0: forward predictability — how concentrated are the next-word expectations?
    // High = the word strongly predicts what follows (like "very" → adjective)
    // Low = many different things follow (like "the" → anything)
    if(nx){
      const ent=Object.entries(nx);const tot=ent.reduce((a,[,c])=>a+c,0);
      const top=ent.sort((a,b)=>b[1]-a[1])[0];
      s[0]=top?top[1]/tot:0; // concentration of top next-word
    }

    // 1: forward diversity — how many different words follow this one?
    if(nx)s[1]=Math.min(Object.keys(nx).length/15,1);

    // 2: backward predictability — how concentrated is what precedes?
    if(px){
      const ent=Object.entries(px);const tot=ent.reduce((a,[,c])=>a+c,0);
      const top=ent.sort((a,b)=>b[1]-a[1])[0];
      s[2]=top?top[1]/tot:0;
    }

    // 3: backward diversity
    if(px)s[3]=Math.min(Object.keys(px).length/15,1);

    // 4: directional asymmetry — does the word predict forward more than backward?
    // Agent-like words predict what follows. Object-like words are predicted by what precedes.
    s[4]=Math.abs((s[0]||0)-(s[2]||0)); // high = strong directional bias

    // 5: expectation breadth — total unique transitions (fwd + bwd)
    const fwdN=nx?Object.keys(nx).length:0;
    const bwdN=px?Object.keys(px).length:0;
    s[5]=Math.min((fwdN+bwdN)/20,1);

    // 6: forward-backward overlap — do the same words appear before AND after?
    // If yes, the word sits in a symmetric position. If no, it's directional.
    if(nx&&px){
      const fwd=new Set(Object.keys(nx)),bwd=new Set(Object.keys(px));
      const union=new Set([...fwd,...bwd]);
      const inter=[...fwd].filter(x=>bwd.has(x)).length;
      s[6]=union.size?1-inter/union.size:0; // high = directional (different words before/after)
    }

    // 7: expectation stability — do the forward expectations stabilize?
    // Use the concentration of the top-3 as a proxy
    if(nx){
      const ent=Object.entries(nx).sort((a,b)=>b[1]-a[1]);
      const tot=ent.reduce((a,[,c])=>a+c,0);
      const top3=ent.slice(0,3).reduce((a,[,c])=>a+c,0);
      s[7]=tot?top3/tot:0; // high = expectations have settled into a pattern
    }

    return s;
  }

  // ─── The full surface of the stone: all 60 dimensions ───────────────
  // Reading the stone at a word means reading all six faces at once.
  // The result is cached until the next perturbation (feed) invalidates it.
  vec(w){const k=w.toLowerCase().trim();if(this._cache[k])return this._cache[k];const v=[...this.formVec(k),...this.contextVec(k),...this.historyVec(k),...this.influenceVec(k),...this.contrastVec(k),...this.expectationVec(k)];this._cache[k]=v;return v;}

  // ─── Comparing two etchings: ASYMMETRIC ────────────────────────────
  // Two grooves on the same stone. How similar? How directional?
  // compare(a,b) ≠ compare(b,a) because grooves run in directions.
  // "doctor→treats" has a channel; "treats→doctor" may not.
  // The stone remembers which way the chisel moved.
  // The groove from A to B ≠ the groove from B to A.
  // Direction matters because history matters.
  compare(a,b,p="meaning",mask=null){
    const va=this.vec(a),vb=this.vec(b);
    const sc={form:cos(va,vb,...IDX.form),context:cos(va,vb,...IDX.ctx),history:cos(va,vb,...IDX.hist),influence:cos(va,vb,...IDX.inf),contrast:cos(va,vb,...IDX.con),expectation:cos(va,vb,...IDX.exp),ocr:Math.max(0,1-ocrDist(a,b)/Math.max(a.length,b.length,1))};
    sc.full=cos(va,vb,...IDX.all);

    // Directional signal: is B in A's expected-next set?
    const al=a.toLowerCase(),bl=b.toLowerCase();
    const nxA=this.nx[al],nxB=this.nx[bl];
    const fwdAB=nxA?((nxA[bl]||0)/Math.max(Object.values(nxA).reduce((a,b)=>a+b,0),1)):0;
    const fwdBA=nxB?((nxB[al]||0)/Math.max(Object.values(nxB).reduce((a,b)=>a+b,0),1)):0;
    sc.expectsAB=fwdAB; // A expects B to follow
    sc.expectsBA=fwdBA; // B expects A to follow
    sc.directional=Math.abs(fwdAB-fwdBA); // asymmetry

    // Second-order: given A→B, what does the engine expect next?
    // This is the trajectory signal. "doctor treats → patient" vs "patient treats → ???"
    const nx2AB=this.nx2[al]?.[bl];
    const nx2BA=this.nx2[bl]?.[al];
    sc.trajectoryAB=nx2AB?Math.min(Object.keys(nx2AB).length/5,1):0; // how rich is A→B→?
    sc.trajectoryBA=nx2BA?Math.min(Object.keys(nx2BA).length/5,1):0; // how rich is B→A→?

    const rt=this.config.routing[p]||{form:.14,context:.14,history:.14,influence:.14,contrast:.14,expectation:.14,ocr:.14};
    let wt={...rt};
    if(mask){for(const ch of["form","context","history","influence","contrast","expectation","ocr"])if(!mask[ch])wt[ch]=0;const s=Object.values(wt).reduce((a,b)=>a+b,0);if(s>0)for(const ch in wt)wt[ch]/=s;}
    const routed=wt.form*sc.form+wt.context*sc.context+wt.history*sc.history+(wt.influence||0)*sc.influence+(wt.contrast||0)*sc.contrast+(wt.expectation||0)*sc.expectation+wt.ocr*sc.ocr;
    return{routed,...sc,weights:wt};
  }

  correct(g,k=5){const cands=this._candidates(g);const out=cands.map(w=>({word:w,...this.compare(g,w,"correction")}));out.sort((a,b)=>b.routed-a.routed||(this.wf[b.word]||0)-(this.wf[a.word]||0));const top=out.slice(0,k);const conf=top.length>=2?top[0].routed-top[1].routed:top.length?1:0;return{candidates:top,confidence:conf,reject:conf<this.config.thresholds.confidence.reject,lowConfidence:conf<this.config.thresholds.confidence.low};}
  similar(w,k=10){const key=w.toLowerCase(),out=[];for(const c of Object.keys(this.wf)){if(c===key)continue;out.push({word:c,...this.compare(w,c,"meaning")});}return out.sort((a,b)=>b.routed-a.routed).slice(0,k);}

  // ─── Which unetched regions will accept the chisel easiest ─────────
  // Before two grooves meet, how strongly does the stone's grain pull
  // them toward each other? This is prediction — which words WILL
  // relate, based on the existing pattern of etchings.
  //
  // Five signals, weighted by what actually discriminates:
  //   Shared orbit (35%) — their neighborhoods overlap (relational)
  //   Trajectory pull (20%) — one groove runs toward the other (directional)
  //   Indirect paths (20%) — a groove runs through an intermediary (structural)
  //   Expectation overlap (15%) — they predict similar futures
  //   Form resonance (5%) — their shapes match (baby-state, weakest signal)
  //   Contrast alignment (5%) — they deviate from baseline similarly
  //
  // ASYMMETRIC: the stone's grain may pull a toward b more than b toward a.
  // Pre-contact attraction. Before two words meet in a sentence,
  // how strongly do they attract? This is the stone predicting
  // where the next mark will land.
  //
  // Five signals:
  //   1. Form resonance — do the words look structurally similar?
  //   2. Shared orbit — do they share co-occurrence neighbors?
  //   3. Trajectory pull — does one appear in the other's nx/px tables?
  //   4. Contrast alignment — do they deviate from baseline in similar ways?
  //   5. Expectation overlap — do they predict similar next-words?
  //
  // High affinity = these words are likely to bind meaningfully.
  // Low affinity = these words live in different regions of the engine's space.
  // Asymmetric: affinity(a,b) may differ from affinity(b,a) because
  // trajectory pull is directional.

  affinity(a, b) {
    a = a.toLowerCase(); b = b.toLowerCase();
    const result = { a, b };

    // 1. Form resonance (symmetric) — word shape similarity
    const fa = this.formVec(a), fb = this.formVec(b);
    result.formResonance = cos(fa, fb, 0, 16);

    // 2. Shared orbit (symmetric) — weighted Jaccard of co-occurrence neighborhoods
    // Weight by distinctiveness: rare shared neighbors count more than common ones
    const coA = this.co[a], coB = this.co[b];
    if (coA && coB) {
      const nbA = Object.keys(coA), nbB = new Set(Object.keys(coB));
      const shared = nbA.filter(x => nbB.has(x));
      if (shared.length && nbA.length) {
        // Weight each shared neighbor by 1/log(freq) — rare neighbors count more
        const wt = w => 1 / Math.max(Math.log2((this.wf[w] || 0) + 1), 1);
        const sharedWeight = shared.reduce((s, w) => s + wt(w), 0);
        const totalWeight = nbA.reduce((s, w) => s + wt(w), 0) + [...nbB].reduce((s, w) => s + wt(w), 0) - sharedWeight;
        result.sharedOrbit = totalWeight > 0 ? sharedWeight / totalWeight : 0;
      } else { result.sharedOrbit = 0; }
    } else { result.sharedOrbit = 0; }

    // 3. Trajectory pull (ASYMMETRIC)
    // Does a predict b? Does b predict a? How strong is the pull?
    const nxA = this.nx[a], nxB = this.nx[b];
    const totA = nxA ? Object.values(nxA).reduce((s, v) => s + v, 0) : 0;
    const totB = nxB ? Object.values(nxB).reduce((s, v) => s + v, 0) : 0;
    result.pullAB = totA ? (nxA[b] || 0) / totA : 0;  // a pulls b forward
    result.pullBA = totB ? (nxB[a] || 0) / totB : 0;  // b pulls a forward
    // Also check second-order: does a→?→b or b→?→a exist?
    let indirectAB = 0, indirectBA = 0;
    if (nxA) { for (const mid of Object.keys(nxA)) { if (this.nx[mid]?.[b]) indirectAB++; } }
    if (nxB) { for (const mid of Object.keys(nxB)) { if (this.nx[mid]?.[a]) indirectBA++; } }
    result.indirectAB = Math.min(indirectAB / 5, 1);
    result.indirectBA = Math.min(indirectBA / 5, 1);

    // 4. Contrast alignment (symmetric) — do they deviate from baseline similarly?
    const ca = this.contrastVec(a), cb = this.contrastVec(b);
    result.contrastAlignment = cos(ca, cb, 0, 8);

    // 5. Expectation overlap (symmetric) — do they predict similar futures?
    const ea = this.expectationVec(a), eb = this.expectationVec(b);
    result.expectationOverlap = cos(ea, eb, 0, 8);

    // Combined affinity score (directional: a toward b)
    // Relational signals dominate: shared experience and trajectory matter most.
    // Form resonance is weakest — shape similarity is not affinity.
    result.affinityAB = (
      result.formResonance * 0.05 +
      result.sharedOrbit * 0.35 +
      result.pullAB * 0.20 +
      result.indirectAB * 0.20 +
      result.contrastAlignment * 0.05 +
      result.expectationOverlap * 0.15
    );
    result.affinityBA = (
      result.formResonance * 0.05 +
      result.sharedOrbit * 0.35 +
      result.pullBA * 0.20 +
      result.indirectBA * 0.20 +
      result.contrastAlignment * 0.05 +
      result.expectationOverlap * 0.15
    );

    result.mutual = (result.affinityAB + result.affinityBA) / 2;
    result.asymmetry = Math.abs(result.affinityAB - result.affinityBA);

    return result;
  }

  // ─── Walking the stone: reading etchings in sequence ────────────────
  // Run your finger along the stone's surface. At each groove:
  //   - Does this groove follow from the last? (sequential surprise)
  //   - Does the stone's grain expect it here? (trajectory surprise)
  //   - Does the accumulated pattern of all preceding grooves
  //     create a field that accepts this one? (field surprise)
  //
  // The field is the key. Each groove deposits its neighborhood into
  // a running context. Early grooves reshape what later grooves can be.
  // This is the dynamic state transformation — not in any table,
  // but built live from the intersection of all preceding neighborhoods.
  //
  // Affinity gate: if two adjacent grooves share many neighbors,
  // the stone's grain was already aligned — reduce surprise up to 30%.
  // High-affinity arrivals slide in. Low-affinity arrivals scrape.
  scoreSentence(raw){
    const ws=(raw.toLowerCase().match(/[a-z0-9]+/g)||[]).filter(w=>w.length>=1);
    if(ws.length<2)return{words:ws,steps:[],totalSurprise:0,meanSurprise:0,coherence:0};

    const steps=[];
    let totalSurprise=0;
    const contextField=new Map(); // word → accumulated expectation weight from all prior words

    for(let i=0;i<ws.length;i++){
      const w=ws[i];
      const step={word:w,position:i,known:w in this.wf};

      // 1. Sequential surprise (nx / nx2)
      if(i>0){
        const prev=ws[i-1];
        const nxPrev=this.nx[prev];
        const totalNext=nxPrev?Object.values(nxPrev).reduce((a,b)=>a+b,0):0;
        step.seqExpected=nxPrev?((nxPrev[w]||0)/Math.max(totalNext,1)):0;
        step.seqSurprise=1-step.seqExpected;
      }else{step.seqExpected=null;step.seqSurprise=0;}

      if(i>=2){
        const pp=ws[i-2],prev=ws[i-1];
        const nx2pp=this.nx2[pp]?.[prev];
        const totalNx2=nx2pp?Object.values(nx2pp).reduce((a,b)=>a+b,0):0;
        step.trajExpected=nx2pp?((nx2pp[w]||0)/Math.max(totalNx2,1)):0;
        step.trajSurprise=1-step.trajExpected;
      }else{step.trajExpected=null;step.trajSurprise=0;}

      // 2. Context field surprise — does the accumulated field expect this word?
      // The field is the union of all preceding words' co-occurrence neighborhoods.
      // If multiple preceding words all co-occur with w, field surprise is low.
      // If none do, it's high. This is the dynamic reshaping — early words create the field.
      if(i>0&&contextField.size>0){
        const fieldWeight=contextField.get(w)||0;
        const maxField=Math.max(...contextField.values(),1);
        step.fieldExpected=fieldWeight/maxField; // 0=no prior word expected this, 1=strongly expected
        step.fieldSurprise=1-step.fieldExpected;
      }else{step.fieldExpected=null;step.fieldSurprise=0;}

      // 3. Novelty — does this word expand the field or reinforce it?
      const co=this.co[w];
      if(co){
        const newNbs=Object.keys(co).filter(n=>!contextField.has(n)).length;
        step.novelty=Object.keys(co).length?newNbs/Object.keys(co).length:0;
      }else{step.novelty=1;} // unknown word = maximally novel

      // Combined surprise: weighted blend of all three signals
      const seqW=0.35, trajW=0.30, fieldW=0.35;
      const seqS=step.seqSurprise;
      const trajS=i>=2?step.trajSurprise:seqS; // fall back to seq if no traj
      const fieldS=i>0?step.fieldSurprise:0;
      const baseSurprise=seqS*seqW+trajS*trajW+fieldS*fieldW;

      // Affinity gate: quick shared-orbit Jaccard with previous word
      // High affinity = the word was "expected" in a broader sense → reduce surprise up to 30%
      if(i>0){
        const prevCo=this.co[ws[i-1]], wCo=this.co[w];
        if(prevCo&&wCo){
          const pN=Object.keys(prevCo),wN=new Set(Object.keys(wCo));
          const shared=pN.filter(x=>wN.has(x)).length;
          const union=new Set([...pN,...wN]).size;
          step.affinityGate=union?shared/union:0;
        }else step.affinityGate=0;
      }else step.affinityGate=0;
      step.surprise=baseSurprise*(1-step.affinityGate*0.3);

      totalSurprise+=step.surprise;
      step.cumulativeSurprise=totalSurprise;
      steps.push(step);

      // UPDATE THE FIELD: this word's co-occurrence neighbors become expectations
      // for future words. This is the dynamic reshaping — each word actively
      // reconfigures what later words are allowed to mean.
      if(co){
        for(const [nb,cnt] of Object.entries(co)){
          contextField.set(nb,(contextField.get(nb)||0)+cnt);
        }
      }
      // The word itself is now "in context" — boost its own field weight
      contextField.set(w,(contextField.get(w)||0)+10);
    }

    const meanSurprise=steps.length?totalSurprise/steps.length:0;
    const coherence=1-Math.min(meanSurprise,1);

    return{words:ws,steps,totalSurprise,meanSurprise,coherence};
  }
  stats(){const v=Object.keys(this.wf);return{version:this.version,sentences:this.ns,tokens:this.nt,vocabulary:v.length,bigrams:Object.keys(this.bf).length,cooccurrences:Object.values(this.co).reduce((a,v)=>a+Object.keys(v).length,0),mature:v.filter(w=>this.wf[w]>=10).length};}
  serialize(){const tr=(o,n)=>Object.fromEntries(Object.entries(o).map(([k,v])=>[k,Array.isArray(v)?v.slice(-n):v]));return JSON.stringify({version:this.version,config:this.config,wf:this.wf,co:this.co,wp:tr(this.wp,50),bf:this.bf,fs:this.fs,ls:this.ls,eg:tr(this.eg,30),ph:tr(this.ph,30),nv:tr(this.nv,20),sl:tr(this.sl,30),sn:tr(this.sn,30),sd:tr(this.sd,30),nx:this.nx,px:this.px,nx2:this.nx2,gp:this._globalPos.slice(-200),gsl:this._globalSentLen.slice(-200),gfs:this._globalFreqSum,gwc:this._globalWordCount,pn:Object.fromEntries(Object.entries(this._pn).map(([k,v])=>[k,[...v]])),ns:this.ns,nt:this.nt});}
  static deserialize(json){const d=JSON.parse(json);const e=new ShifuEngine(d.config||CONFIG);e.wf=d.wf||{};e.co=d.co||{};e.wp=d.wp||{};e.bf=d.bf||{};e.fs=d.fs||{};e.ls=d.ls||{};e.eg=d.eg||{};e.ph=d.ph||{};e.nv=d.nv||{};e.sl=d.sl||{};e.sn=d.sn||{};e.sd=d.sd||{};e.nx=d.nx||{};e.px=d.px||{};e.nx2=d.nx2||{};e._globalPos=d.gp||[];e._globalSentLen=d.gsl||[];e._globalFreqSum=d.gfs||0;e._globalWordCount=d.gwc||0;if(d.pn)e._pn=Object.fromEntries(Object.entries(d.pn).map(([k,v])=>[k,new Set(v)]));e.ns=d.ns||0;e.nt=d.nt||0;for(const w of Object.keys(e.wf))e._reindex(w);return e;}
}

module.exports={ShifuEngine,CONFIG,VERSION,ocrDist,levDist,cos,mn,sd,prng,shuffle,IDX};
