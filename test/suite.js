const{ShifuEmbryo,editDistance,ocrDistance,sharedBigrams}=require("../core/engine");
let pass=0,fail=0;
function assert(name,cond){if(cond){pass++;console.log(`  ✓ ${name}`);}else{fail++;console.log(`  ✗ ${name}`);}}
function S(name){console.log(`\n━━━ ${name} ━━━`);}

S("UTILITY FUNCTIONS");
assert("editDistance basic",editDistance("doctor","docter")===1);
assert("editDistance identical",editDistance("abc","abc")===0);
assert("ocrDistance rn→m cheaper",ocrDistance("rnonitoring","monitoring")<editDistance("rnonitoring","monitoring"));
assert("sharedBigrams identical",sharedBigrams("doctor","doctor")===1);
assert("sharedBigrams none",sharedBigrams("abc","xyz")===0);

S("EMBRYO — BORN EMPTY");
const e=new ShifuEmbryo();
assert("starts with no nodes",Object.keys(e.nodes).length===0);
assert("sentence count 0",e.sentenceCount===0);
assert("depth of unknown is unborn",e.depth("anything").level==="unborn");
assert("compare unknown words returns surface",e.compare("abc","xyz").depth==="surface");

S("FEED — THE PERTURBATION");
const corpus=[
  "Doctor treats patient with medication and therapy",
  "Doctor prescribed levetiracetam for seizure prophylaxis",
  "Doctor examined the patient and noted bilateral weakness",
  "Doctor ordered CT head and MRI brain for the stroke patient",
  "Patient presents with acute stroke and left sided weakness",
  "Patient has history of hypertension and diabetes mellitus",
  "Patient was admitted to the stroke unit for monitoring",
  "Aspirin was prescribed for the patient with stroke",
  "Sear the steak in a hot cast iron skillet with olive oil",
  "Deglaze the pan with red wine and reduce by half",
  "Simmer the tomato sauce with fresh basil and oregano",
];
for(const s of corpus)e.feed(s);
assert("nodes created",Object.keys(e.nodes).length>0);
assert("doctor exists",!!e.nodes["doctor"]);
assert("doctor freq > 0",e.nodes["doctor"].freq>0);
assert("doctor has neighbors",Object.keys(e.nodes["doctor"].neighbors).length>0);
assert("doctor has next",Object.keys(e.nodes["doctor"].next).length>0);
assert("sentence count matches",e.sentenceCount===corpus.length);
console.log(`  Vocab: ${Object.keys(e.nodes).length}, Sentences: ${e.sentenceCount}`);

S("DEPTH — EVIDENCE MEASUREMENT");
assert("doctor has depth > surface",e.depth("doctor").level!=="surface"&&e.depth("doctor").level!=="unborn");
assert("oregano is shallow",e.depth("oregano").level==="surface"||e.depth("oregano").level==="shallow");
assert("unseen word is unborn",e.depth("xyzzy").level==="unborn");
console.log(`  doctor: ${e.depth("doctor").level} (${e.depth("doctor").evidence.toFixed(2)})`);
console.log(`  oregano: ${e.depth("oregano").level} (${e.depth("oregano").evidence.toFixed(2)})`);

S("COMPARE — WEIGHTS GROW WITH EVIDENCE");
const cmp1=e.compare("doctor","patient");
assert("doctor↔patient has similarity",cmp1.similarity>0);
assert("doctor↔patient has neighbor signal",cmp1.signals.neighborOverlap!==undefined);
assert("doctor↔patient deeper than surface",cmp1.depth!=="surface");
const cmp2=e.compare("doctor","oregano");
assert("doctor↔oregano lower than doctor↔patient",cmp2.similarity<cmp1.similarity||cmp2.totalWeight<cmp1.totalWeight);
console.log(`  doctor↔patient: ${cmp1.similarity.toFixed(4)} (${cmp1.depth}, weight ${cmp1.totalWeight.toFixed(2)})`);
console.log(`  doctor↔oregano: ${cmp2.similarity.toFixed(4)} (${cmp2.depth}, weight ${cmp2.totalWeight.toFixed(2)})`);

S("ASYMMETRIC COMPARE — DIRECTION MATTERS");
const dt=e.compare("doctor","treats"),td=e.compare("treats","doctor");
assert("expectsAB exists",dt.signals.expectsAB!==undefined);
assert("doctor→treats ≠ treats→doctor",dt.signals.expectsAB!==dt.signals.expectsBA);
console.log(`  doctor→treats: ${(dt.signals.expectsAB*100).toFixed(1)}%, treats→doctor: ${(dt.signals.expectsBA*100).toFixed(1)}%`);

S("TRAJECTORY — SECOND ORDER");
assert("doctor→treats→? exists",e.nodes["doctor"]?.next2?.["treats"]!==undefined);
const nx2=e.nodes["doctor"]?.next2?.["treats"];
console.log(`  doctor→treats→?: ${JSON.stringify(nx2)}`);
assert("patient in doctor→treats→?",nx2&&"patient" in nx2);

S("AFFINITY — PRE-CONTACT ATTRACTION");
// Feed more to strengthen orbits
e.feed("doctor treats patient with careful attention to symptoms");
e.feed("patient was seen by the doctor for follow up assessment");
e.feed("the doctor and patient discussed treatment options together");
const af1=e.affinity("doctor","patient");
const af2=e.affinity("doctor","skillet");
assert("affinity returns object",typeof af1==="object"&&af1.known===true);
assert("doctor↔patient orbit > 0",af1.orbit>0);
assert("doctor↔patient > doctor↔skillet",af1.mutual>af2.mutual);
console.log(`  doctor↔patient mutual: ${af1.mutual.toFixed(4)}`);
console.log(`  doctor↔skillet mutual: ${af2.mutual.toFixed(4)}`);

S("SCORE SENTENCE — DYNAMIC FIELD + AFFINITY GATE");
const s1=e.scoreSentence("doctor treats patient");
const s2=e.scoreSentence("patient treats doctor");
assert("experienced > reversed coherence",s1.coherence>s2.coherence);
console.log(`  "doctor treats patient"  coherence: ${s1.coherence.toFixed(4)}`);
console.log(`  "patient treats doctor"  coherence: ${s2.coherence.toFixed(4)}`);
const s3=e.scoreSentence("patient presents with acute stroke");
const s4=e.scoreSentence("stroke acute with presents patient");
assert("natural > scrambled",s3.coherence>s4.coherence);
console.log(`  "patient presents with acute stroke"  coherence: ${s3.coherence.toFixed(4)}`);
console.log(`  "stroke acute with presents patient"  coherence: ${s4.coherence.toFixed(4)}`);
assert("affinity gate present",s1.steps.some(st=>st.afGate>0));

S("PRESSURE — WHERE THE GRAPH PULLS AND PUSHES");
const pres=e.pressure();
assert("pressure returns array",Array.isArray(pres));
assert("pressure has variation",pres.length>0&&pres[0].pressure!==pres[pres.length-1].pressure);
const vacs=e.vacuums(5);
const surps=e.surpluses(5);
console.log(`  Vacuums: ${vacs.length>0?vacs.map(v=>`${v.word}(${v.pressure})`).join(", "):"none yet (corpus too small)"}`);
console.log(`  Surpluses: ${surps.map(v=>`${v.word}(${v.pressure})`).join(", ")}`);
assert("surpluses exist",surps.length>0);
// Feed more data — vacuums should emerge as some words get predicted but stay thin
e.feed("Doctor prescribed levetiracetam for seizure and monitored the patient closely");
e.feed("Doctor prescribed aspirin for stroke prevention and the patient improved");
e.feed("Doctor prescribed gabapentin for neuropathic pain in the bilateral limbs");
e.feed("Doctor prescribed ceftriaxone for pneumonia and reviewed blood cultures");
e.feed("Doctor prescribed metformin for diabetes mellitus type two management");
const pres2=e.pressure();
const vacs2=e.vacuums(5);
console.log(`  After more data — vacuums: ${vacs2.length>0?vacs2.map(v=>`${v.word}(${v.pressure})`).join(", "):"still none"}`);
// Words like "monitored", "improved" may now have vacuum — they're predicted by "doctor prescribed X for Y and Z" but are themselves thin
const brs=e.bridges(5);
console.log(`  Bridges: ${brs.map(b=>`${b.word}(closure=${b.closure.toFixed(2)})`).join(", ")}`);

S("CORRECTION — THE GENOME");
const c1=e.correct("seisure");
assert("seisure corrects",c1.candidates.length>0);
assert("seizure in candidates",c1.candidates.some(c=>c.word==="seizure"));
const c2=e.correct("pateint");
assert("pateint corrects to patient",c2.candidates[0]?.word==="patient");
console.log(`  seisure → ${c1.candidates[0]?.word} (${c1.confidence.toFixed(3)})`);
console.log(`  pateint → ${c2.candidates[0]?.word} (${c2.confidence.toFixed(3)})`);

S("SERIALIZATION — ROUND TRIP");
const json=e.serialize();
const e2=ShifuEmbryo.deserialize(json);
assert("vocab preserved",Object.keys(e2.nodes).length===Object.keys(e.nodes).length);
assert("sentence count preserved",e2.sentenceCount===e.sentenceCount);
assert("doctor freq preserved",e2.nodes["doctor"].freq===e.nodes["doctor"].freq);
assert("nx preserved",JSON.stringify(e2.nodes["doctor"].next)===JSON.stringify(e.nodes["doctor"].next));
assert("nx2 preserved",JSON.stringify(e2.nodes["doctor"].next2)===JSON.stringify(e.nodes["doctor"].next2));
const sOrig=e.scoreSentence("doctor treats patient");
const sDeser=e2.scoreSentence("doctor treats patient");
assert("deserialized scoring works",Math.abs(sDeser.coherence-sOrig.coherence)<0.01);

S("STATS");
const st=e.stats();
console.log(`  ${JSON.stringify(st)}`);
assert("stats has depths",st.depths!==undefined);
assert("stats has version",st.version==="2.0.0");

console.log(`\n╔═══════════════════════════════════════╗`);
console.log(`║  ${pass}/${pass+fail} passed${fail?`, ${fail} FAILED`:""}${"".padEnd(25-(pass+fail).toString().length*2)}║`);
console.log(`╚═══════════════════════════════════════╝`);
if(fail)process.exit(1);
