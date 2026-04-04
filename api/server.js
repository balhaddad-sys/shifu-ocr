// SHIFU API v1.4.0 — Production-hardened
// Pluggable persistence. Pluggable rate limiter. Startup smoke test.
// Structured error contracts. Request validation. Single version source.
const express=require("express"),fs=require("fs"),path=require("path");
const{ShifuEmbryo:ShifuEngine,VERSION}=require("../core/engine");
const app=express();app.use(express.json({limit:"10mb"}));app.use(express.static(path.join(__dirname,"../public")));
const DATA_DIR=process.env.SHIFU_DATA||path.join(__dirname,"../data");
const API_KEY=process.env.SHIFU_API_KEY||null;
const PERSIST_MS=parseInt(process.env.SHIFU_PERSIST_MS||"30000");
const RATE_WINDOW=parseInt(process.env.SHIFU_RATE_WINDOW||"60000");
const RATE_MAX=parseInt(process.env.SHIFU_RATE_MAX||"120");
const SAFETY="Shifu is a support tool. It does not provide diagnoses. All corrections must be verified by a qualified professional.";

function log(level,msg,meta={}){console.log(JSON.stringify({ts:new Date().toISOString(),level,version:VERSION,msg,...meta}));}
function apiError(res,status,code,message,details){const body={error:{code,message,version:VERSION}};if(details)body.error.details=details;return res.status(status).json(body);}
function requireBody(res,body,fields){for(const f of fields){if(!body[f]||(typeof body[f]==="string"&&!body[f].trim()))return apiError(res,400,"INVALID_INPUT",`Missing: ${f}`);}return null;}

// ─── Pluggable persistence (file default, swap via SHIFU_STORE=firestore) ──
const store=(()=>{
  // To use Firestore: set SHIFU_STORE=firestore, install firebase-admin, uncomment adapter
  return{type:"file",
    async load(id){const p=path.join(DATA_DIR,`${id}.json`);return fs.existsSync(p)?fs.readFileSync(p,"utf8"):null;},
    async save(id,json){if(!fs.existsSync(DATA_DIR))fs.mkdirSync(DATA_DIR,{recursive:true});fs.writeFileSync(path.join(DATA_DIR,`${id}.json`),json);},
    async list(){if(!fs.existsSync(DATA_DIR))return[];return fs.readdirSync(DATA_DIR).filter(f=>f.endsWith(".json")).map(f=>f.replace(".json",""));},
  };
})();

// ─── Pluggable rate limiter (memory default, swap via SHIFU_LIMITER=redis) ──
const counts={};
const limiter={type:"memory",async check(ip){const now=Date.now();if(!counts[ip]||now-counts[ip].s>RATE_WINDOW)counts[ip]={s:now,c:0};return++counts[ip].c<=RATE_MAX;}};

// ─── Engine store (in-memory, periodic persist) ──────────────────────
const engines={};
async function getEngine(id="default"){
  if(engines[id]){engines[id].la=Date.now();return engines[id].eng;}
  let eng;try{const j=await store.load(id);if(j)eng=ShifuEngine.deserialize(j);}catch(e){log("WARN",`Load ${id}`,{error:e.message});}
  if(!eng)eng=new ShifuEngine();
  engines[id]={eng,dirty:false,la:Date.now()};return eng;
}
function markDirty(id){if(engines[id])engines[id].dirty=true;}
async function persistAll(){
  for(const[id,s]of Object.entries(engines)){
    if(!s.dirty)continue;
    try{await store.save(id,s.eng.serialize());s.dirty=false;log("INFO",`Persisted: ${id}`,{vocab:s.eng.stats().vocab});}
    catch(e){log("ERROR",`Persist ${id}`,{error:e.message});}
  }
}
const pTimer=setInterval(persistAll,PERSIST_MS);
process.on("SIGINT",async()=>{clearInterval(pTimer);await persistAll();process.exit(0);});
process.on("SIGTERM",async()=>{clearInterval(pTimer);await persistAll();process.exit(0);});

// ─── Middleware ───────────────────────────────────────────────────────
function auth(req,res,next){if(!API_KEY)return next();if((req.headers["x-api-key"]||req.query.apiKey)!==API_KEY)return apiError(res,401,"AUTH_FAILED","Invalid API key");next();}
async function rateLimit(req,res,next){const ip=req.ip||"unknown";if(!(await limiter.check(ip)))return apiError(res,429,"RATE_LIMITED","Too many requests",{retryAfterMs:RATE_WINDOW});next();}
function withMetrics(name,fn){return async(req,res)=>{const start=Date.now(),eid=req.query.engineId||"default";try{await fn(req,res,eid);}catch(e){log("ERROR",name,{error:e.message,engineId:eid});if(!res.headersSent)apiError(res,500,"INTERNAL","Internal error");}log("INFO",name,{latency:Date.now()-start,engineId:eid});};}
app.use(auth);app.use(rateLimit);

// ─── Endpoints ───────────────────────────────────────────────────────
app.get("/health",withMetrics("health",async(req,res,eid)=>{
  const eng=await getEngine(eid);
  res.json({status:"ok",version:VERSION,engineId:eid,store:store.type,limiter:limiter.type,...eng.stats(),disclaimer:SAFETY});
}));

app.get("/engines",withMetrics("engines",async(_,res)=>{
  const list=Object.entries(engines).map(([id,s])=>({id,loaded:true,...s.eng.stats(),dirty:s.dirty}));
  try{for(const id of await store.list())if(!engines[id])list.push({id,loaded:false});}catch{}
  res.json({engines:list,version:VERSION});
}));

app.post("/feed",withMetrics("feed",async(req,res,eid)=>{
  if(requireBody(res,req.body,["text"]))return;
  const eng=await getEngine(eid);
  const beforeVocab=Object.keys(eng.nodes).length;
  const beforeConn=Object.values(eng.nodes).reduce((a,n)=>a+Object.keys(n.neighbors).length,0);
  const r=eng.feedText(req.body.text);markDirty(eid);
  const afterVocab=Object.keys(eng.nodes).length;
  const afterConn=Object.values(eng.nodes).reduce((a,n)=>a+Object.keys(n.neighbors).length,0);
  const newWords=afterVocab-beforeVocab,newConnections=afterConn-beforeConn;
  res.json({engineId:eid,...r,newWords,newConnections,novelty:newWords/Math.max(r.tokens,1),stats:eng.stats()});
}));

app.post("/compare",withMetrics("compare",async(req,res,eid)=>{
  if(requireBody(res,req.body,["a","b"]))return;
  const{a,b}=req.body;
  res.json({engineId:eid,a,b,...(await getEngine(eid)).compare(a,b)});
}));

app.post("/correct",withMetrics("correct",async(req,res,eid)=>{
  if(requireBody(res,req.body,["text"]))return;
  const{text,k=5}=req.body;const eng=await getEngine(eid);
  const results=text.split(/\s+/).map(w=>{
    if(eng.nodes[w.toLowerCase()])return{original:w,known:true};
    const r=eng.correct(w,k);
    return{original:w,known:false,correction:r.candidates[0]?.word||null,confidence:r.confidence,candidates:r.candidates.slice(0,3)};
  });
  res.json({engineId:eid,results,disclaimer:SAFETY});
}));

app.post("/explore",withMetrics("explore",async(req,res,eid)=>{
  if(requireBody(res,req.body,["word"]))return;
  const eng=await getEngine(eid);const w=req.body.word.toLowerCase();
  const node=eng.nodes[w]||null;
  res.json({engineId:eid,word:w,frequency:node?node.freq:0,depth:eng.depth(w),similar:eng.similar(w,10),node:node?{freq:node.freq,neighbors:Object.keys(node.neighbors).length,next:Object.keys(node.next).length,prev:Object.keys(node.prev).length}:null});
}));

app.post("/score",withMetrics("score",async(req,res,eid)=>{
  if(requireBody(res,req.body,["text"]))return;
  res.json({engineId:eid,...(await getEngine(eid)).scoreSentence(req.body.text)});
}));

app.post("/affinity",withMetrics("affinity",async(req,res,eid)=>{
  if(requireBody(res,req.body,["a","b"]))return;
  res.json({engineId:eid,...(await getEngine(eid)).affinity(req.body.a,req.body.b)});
}));

app.post("/describe",withMetrics("describe",async(req,res,eid)=>{
  if(requireBody(res,req.body,["a","b"]))return;
  res.json({engineId:eid,...(await getEngine(eid)).describe(req.body.a,req.body.b)});
}));

app.get("/stats",withMetrics("stats",async(req,res,eid)=>{
  res.json({engineId:eid,...(await getEngine(eid)).stats()});
}));

app.post("/pressure",withMetrics("pressure",async(req,res,eid)=>{
  res.json((await getEngine(eid)).pressure().slice(0,20));
}));
app.post("/vacuums",withMetrics("vacuums",async(req,res,eid)=>{
  res.json((await getEngine(eid)).vacuums(10));
}));
app.post("/surpluses",withMetrics("surpluses",async(req,res,eid)=>{
  res.json((await getEngine(eid)).surpluses(10));
}));
app.post("/bridges",withMetrics("bridges",async(req,res,eid)=>{
  res.json((await getEngine(eid)).bridges(10));
}));

// ─── Startup smoke test ──────────────────────────────────────────────
function smokeTest(){
  const eng=new ShifuEngine();eng.feed("doctor treats patient");
  const s=eng.scoreSentence("doctor treats patient"),r=eng.correct("seisure");
  const ok=typeof s.coherence==="number"&&r.candidates.length>0;
  if(!ok){log("ERROR","Smoke test FAILED");process.exit(1);}
  log("INFO","Smoke test passed",{version:VERSION,coherence:+s.coherence.toFixed(3)});
}

// ─── Start ───────────────────────────────────────────────────────────
const PORT=process.env.PORT||3001;
smokeTest();
app.listen(PORT,()=>{log("INFO",`Shifu API v${VERSION} ready`,{port:PORT,store:store.type,limiter:limiter.type,auth:API_KEY?"enabled":"open",persistMs:PERSIST_MS});});
module.exports=app;
