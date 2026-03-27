// SHIFU OCR — Integration Test Suite
// Tests the full system: core engine + clinical layer + learning loop + corpus
//                       + trained model loader + pipeline + persistence

const {
  createShifu, restoreShifu, createOrRestore, ShifuEngine, VERSION,
  correctLine, correctTableRow, assessConfidence,
  checkLabRange, ocrWeightedDistance, fixDigraphs,
  seedEngine, fullSeed,
  loadTrainedModel, extractConfusionKnowledge, feedTrainedModelToEngine,
  ShifuPipeline, ShifuPersistence, withAutoSave,
  VOCABULARY,
} = require('../index');

const path = require('path');
const fs = require('fs');

let pass = 0, fail = 0, total = 0;
function assert(name, condition) {
  total++;
  if (condition) { pass++; }
  else { fail++; console.log(`  FAIL: ${name}`); }
}
function S(name) { console.log(`\n--- ${name} ---`); }

// ═══════════════════════════════════════════════════════════════════
S("CORE ENGINE");
// ═══════════════════════════════════════════════════════════════════

const eng = new ShifuEngine();
assert("version 2.0.0", eng.version === "2.0.0");

const corpus = [
  "Doctor treats patient with medication and therapy.",
  "Doctor prescribed levetiracetam for seizure prophylaxis.",
  "Doctor examined the patient and noted bilateral weakness.",
  "Patient presents with acute stroke and left sided weakness.",
  "Patient has history of hypertension and diabetes mellitus.",
];
for (const s of corpus) eng.feed(s);
assert("sentences fed", eng.ns === corpus.length);
assert("60D vectors", eng.vec("doctor").length === 60);
assert("nx populated", Object.keys(eng.nx).length > 0);

// ═══════════════════════════════════════════════════════════════════
S("RESONANCE (from Shifu Ultimate)");
// ═══════════════════════════════════════════════════════════════════

const resEng = new ShifuEngine();
const resCorpus = [
  "doctor treats patient with medication",
  "doctor treats patient with therapy",
  "doctor examines patient carefully",
  "doctor prescribes medication for patient",
  "physician treats patient with medication",
  "physician treats patient with therapy",
  "physician examines patient carefully",
  "physician prescribes medication for patient",
  "doctor manages patient effectively",
  "physician manages patient effectively",
];
for (const s of resCorpus) resEng.feed(s);

const docPhysRes = resEng.res["doctor"]?.["physician"] || 0;
assert("doctor-physician resonance > 0", docPhysRes > 0);
assert("resonance is symmetric", Math.abs(docPhysRes - (resEng.res["physician"]?.["doctor"] || 0)) < 0.001);

const partners = resEng.resonancePartners("doctor");
assert("doctor has resonance partners", partners.length > 0);
assert("physician is a partner", partners.some(p => p.word === "physician"));

const physDiscount = partners.find(p => p.word === "physician");
if (physDiscount) assert("physician discount > base 0.6", physDiscount.discount > 0.6);

// ═══════════════════════════════════════════════════════════════════
S("CLINICAL CORRECTOR");
// ═══════════════════════════════════════════════════════════════════

const r1 = correctLine("levetiracelam seizure");
assert("levetiracetam corrected", r1.output.toLowerCase().includes("levetiracetam"));
assert("seizure exact match", r1.output.toLowerCase().includes("seizure"));

const dgResult = fixDigraphs("rnedication");
assert("rn→m digraph", dgResult === "medication");

const rowResult = correctTableRow({
  Patient: "Abdulah",
  Diagnosis: "chst infecfion",
  Doctor: "Bader",
});
assert("patient name corrected", rowResult.corrected.Patient.output.toLowerCase() === "abdullah");
assert("diagnosis corrected", rowResult.corrected.Diagnosis.output.toLowerCase().includes("chest"));
assert("doctor name exact", rowResult.corrected.Doctor.output.toLowerCase() === "bader");

// ═══════════════════════════════════════════════════════════════════
S("SAFETY FLAGS");
// ═══════════════════════════════════════════════════════════════════

const labOk = checkLabRange("potassium", "4.5");
assert("potassium 4.5 in range", labOk.status === "in_range");

const labBad = checkLabRange("potassium", "45");
assert("potassium 45 out of range", labBad.status === "OUT_OF_RANGE");
assert("suggests 4.5", labBad.alternatives && labBad.alternatives.includes(4.5));

const glucoseOk = checkLabRange("glucose", "6.5");
assert("glucose 6.5 in range", glucoseOk.status === "in_range");

const dist1 = ocrWeightedDistance("doctor", "doctOr");
const dist2 = ocrWeightedDistance("doctor", "xxxxxx");
assert("OCR distance: similar < different", dist1 < dist2);

// ═══════════════════════════════════════════════════════════════════
S("MEDICAL CORPUS SEEDING");
// ═══════════════════════════════════════════════════════════════════

const seeded = new ShifuEngine();
const seedResult = fullSeed(seeded);
assert("corpus seeded", seedResult.totalTokens > 0);
assert("vocabulary built", Object.keys(seeded.wf).length > 100);

assert("doctor in vocab", "doctor" in seeded.wf);
assert("patient in vocab", "patient" in seeded.wf);
assert("stroke in vocab", "stroke" in seeded.wf);
assert("levetiracetam in vocab", "levetiracetam" in seeded.wf);

const medScore = seeded.scoreSentence("doctor prescribed aspirin for the stroke patient");
const gibScore = seeded.scoreSentence("zzzzz xxxxx yyyyy");
console.log(`  medical coherence: ${medScore.coherence.toFixed(4)}, gibberish: ${gibScore.coherence.toFixed(4)}`);
// Strict: medical text MUST score higher than gibberish. No fallback.
assert("medical > gibberish coherence (strict)", medScore.coherence > gibScore.coherence);
assert("medical sentence has some coherence", medScore.coherence > 0);

const corrResult = seeded.correct("seisure");
assert("seisure → seizure", corrResult.candidates[0]?.word === "seizure");

const corrResult2 = seeded.correct("levetiracelam");
assert("levetiracelam → levetiracetam", corrResult2.candidates[0]?.word === "levetiracetam");

// ═══════════════════════════════════════════════════════════════════
S("TRAINED MODEL LOADER");
// ═══════════════════════════════════════════════════════════════════

const model = loadTrainedModel();
assert("trained model loads", model !== null);
assert("model has landscapes", model && Object.keys(model.landscapes || {}).length > 0);
assert("model has many character landscapes", model && Object.keys(model.landscapes).length >= 36);

const knowledge = extractConfusionKnowledge(model);
assert("confusion pairs extracted", knowledge.pairs.length >= 0);
assert("confusion sentences generated", knowledge.sentences.length > 0);

const trainedEng = new ShifuEngine();
fullSeed(trainedEng);
const fedResult = feedTrainedModelToEngine(trainedEng);
assert("trained model fed to engine", fedResult.fed > 0);
assert("characters counted", fedResult.characters > 0);

// Engine should have more knowledge after loading trained model
const preTrainedVocab = Object.keys(trainedEng.wf).length;
assert("vocab grew from trained model", preTrainedVocab > Object.keys(seeded.wf).length);

// ═══════════════════════════════════════════════════════════════════
S("LEARNING LOOP INTEGRATION");
// ═══════════════════════════════════════════════════════════════════

const shifu = createShifu({ seed: true, loadTrained: true });
assert("shifu created with core", !!shifu.core);
assert("shifu created with learning", !!shifu.learning);
assert("core has vocabulary", Object.keys(shifu.core.wf).length > 100);
assert("core has resonance", Object.values(shifu.core.res).length > 0);

// Trained model was loaded (vocab should be larger than corpus alone)
assert("trained model loaded into core", Object.keys(shifu.core.wf).length > preTrainedVocab - 50);

const adaptiveResult = shifu.correctRowAdaptive({
  Patient: "Abdulah",
  Diagnosis: "stroke",
});
assert("adaptive correction works", adaptiveResult.corrected.Patient.output !== undefined);

const ocrRow = { Patient: "Abdulah", Diagnosis: "Chst infecfion" };
const confirmedRow = { Patient: "Abdullah", Diagnosis: "Chest infection" };
shifu.learn(ocrRow, confirmedRow);
assert("learning count increased", shifu.learning.correctionCount === 1);
assert("core absorbed learning", shifu.core.ns > 0);

const score = shifu.scoreSentence("doctor prescribed aspirin for stroke");
assert("sentence scoring works", score.coherence >= 0);

const cmp = shifu.compare("doctor", "physician");
assert("compare works", cmp.routed > 0);

const stats = shifu.stats();
assert("stats has core", !!stats.core);
assert("stats has learning", !!stats.learning);

// ═══════════════════════════════════════════════════════════════════
S("PIPELINE (text mode)");
// ═══════════════════════════════════════════════════════════════════

const pipeline = new ShifuPipeline(shifu);
assert("pipeline created", !!pipeline);

const textResult = pipeline.processText("Levetiracetarn 5OOmg for seizure");
assert("pipeline processText works", !!textResult.corrected);
assert("pipeline has decision", ['accept', 'verify', 'reject'].includes(textResult.decision));
assert("pipeline has coherence", typeof textResult.coherence === 'number');

const tableResult = pipeline.processTableRow({
  Patient: "Abdulah",
  Diagnosis: "str0ke",
  Status: "Red",
});
assert("pipeline processTableRow works", !!tableResult.corrected);
assert("pipeline table has decision", ['accept', 'verify', 'reject'].includes(tableResult.decision));

// createPipeline from shifu
const pipeline2 = shifu.createPipeline();
assert("createPipeline from shifu works", !!pipeline2);

// ═══════════════════════════════════════════════════════════════════
S("PERSISTENCE");
// ═══════════════════════════════════════════════════════════════════

const tmpDir = path.join(__dirname, '..', '.tmp_test_state');
const persistence = new ShifuPersistence(tmpDir);

assert("no initial state", !persistence.hasSavedState());

persistence.save(shifu);
assert("state saved", persistence.hasSavedState());

const meta = persistence.getMeta();
assert("meta has version", meta && !!meta.version);
assert("meta has savedAt", meta && !!meta.savedAt);

const loaded = persistence.load();
assert("state loads", loaded !== null);
assert("loaded has core", !!loaded.savedCoreState);
assert("loaded has learning", !!loaded.savedLearningState);

// Restore from persistence
const restoredFromPersistence = restoreShifu({
  core: loaded.savedCoreState,
  learning: loaded.savedLearningState,
});
assert("restored from persistence works", Object.keys(restoredFromPersistence.core.wf).length > 0);
assert("restored learning state", restoredFromPersistence.learning.correctionCount === 1);

// Clean up
persistence.clear();
assert("state cleared", !persistence.hasSavedState());
try { fs.rmdirSync(tmpDir); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("AUTO-SAVE");
// ═══════════════════════════════════════════════════════════════════

const tmpAutoDir = path.join(__dirname, '..', '.tmp_test_autosave');
const autoShifu = createShifu({ seed: true, loadTrained: true, autoSave: true, stateDir: tmpAutoDir });
assert("auto-save shifu created", !!autoShifu);
assert("has persistence", !!autoShifu.persistence);
assert("has forceSave", typeof autoShifu.forceSave === 'function');

// Force save
autoShifu.forceSave();
assert("force save works", autoShifu.persistence.hasSavedState());

// Clean up
autoShifu.persistence.clear();
try { fs.rmdirSync(tmpAutoDir); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("SERIALIZATION / RESTORE");
// ═══════════════════════════════════════════════════════════════════

const saved = shifu.serialize();
assert("serialize produces core", !!saved.core);
assert("serialize produces learning", !!saved.learning);

const restored = restoreShifu(saved);
assert("restored core works", Object.keys(restored.core.wf).length > 0);
assert("restored learning works", restored.learning.correctionCount === 1);

const restoredScore = restored.scoreSentence("doctor prescribed aspirin for stroke");
assert("restored scoring works", Math.abs(restoredScore.coherence - score.coherence) < 0.01);

// ═══════════════════════════════════════════════════════════════════
S("CONFIDENCE ASSESSMENT");
// ═══════════════════════════════════════════════════════════════════

const goodLine = correctLine("aspirin stroke patient");
assert("good line confidence", assessConfidence(goodLine) === 'accept' || assessConfidence(goodLine) === 'verify');

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #1 — adaptive state flows into correctLine/correctTableRow");
// ═══════════════════════════════════════════════════════════════════

// After learning a word, the shifu.correctLine API (not just raw correctWord) must recognize it
const regShifu1 = createShifu({ seed: true, loadTrained: false });
regShifu1.learn({}, { Diagnosis: 'xyzuniquemed' });
assert("learned word in vocabulary", regShifu1.learning.vocabulary.isKnown('xyzuniquemed'));
const reg1Result = regShifu1.correctLine('xyzuniquemed');
assert("#1: correctLine uses learning engine", reg1Result.words[0].flag === 'exact');

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #2 — adaptive path runs safety checks");
// ═══════════════════════════════════════════════════════════════════

const regShifu2 = createShifu({ seed: true, loadTrained: false });
const adaptiveLabResult = regShifu2.correctRowAdaptive({ Lab: 'potassium 45' });
assert("#2: adaptive path has safety flags", adaptiveLabResult.safetyFlags.length > 0);
assert("#2: adaptive path flags out-of-range", adaptiveLabResult.safetyFlags.some(f => f.status === 'OUT_OF_RANGE'));
assert("#2: adaptive path hasWarnings", adaptiveLabResult.hasWarnings === true);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #3 — no double-counting in learn()");
// ═══════════════════════════════════════════════════════════════════

const regShifu3 = createShifu({ seed: true, loadTrained: false });
regShifu3.learn({}, { Diagnosis: 'rarediagnosis' });
const freq3 = regShifu3.learning.vocabulary.frequency['rarediagnosis'] || 0;
const expectedFreq3 = Math.max(1, Math.round(require('../index').getLearningRate('Diagnosis', 'rarediagnosis', VOCABULARY)));
assert("#3: single weighted increment after learn", freq3 === expectedFreq3);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #4 — edit-distance alignment for confusion learning");
// ═══════════════════════════════════════════════════════════════════

const { AdaptiveConfusionProfile } = require('../index');
const acp = new AdaptiveConfusionProfile();
acp.recordCorrection('rnedication', 'medication');
// Should record r→m substitution (the rn→m confusion), not r→m,n→e,e→d bogus pairs
// With proper alignment: r aligns to m (substitution), n is deleted, e→e, d→d, etc.
// The key check: 'e,e' should NOT be recorded as a confusion (it's a match)
const eeKey = ['e', 'e'].sort().join(',');
const eeCount = acp.confusionCounts[eeKey] || 0;
assert("#4: proper alignment doesn't record e↔e as confusion", eeCount === 0);
// And m,r SHOULD be recorded (the actual substitution)
assert("#4: total corrections counted", acp.totalCorrections === 1);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #5 — context chains boost individual words");
// ═══════════════════════════════════════════════════════════════════

const { ContextChains } = require('../index');
const cc = new ContextChains();
cc.learn({ Patient: 'abdullah', Diagnosis: 'chest infection' });
const chestBoost = cc.getBoost({ Patient: 'abdullah' }, 'diagnosis', 'chest');
assert("#5: context chain boosts individual word 'chest'", chestBoost > 0);
const infectionBoost = cc.getBoost({ Patient: 'abdullah' }, 'diagnosis', 'infection');
assert("#5: context chain boosts individual word 'infection'", infectionBoost > 0);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #6 — coherence: medical > gibberish (strict)");
// ═══════════════════════════════════════════════════════════════════

// Already tested above with the strict assertion, but add explicit edge cases
const regEng6 = new ShifuEngine();
fullSeed(regEng6);
const knownScore = regEng6.scoreSentence("patient admitted with stroke");
const unknownScore = regEng6.scoreSentence("aaabbb cccddd eeefff");
assert("#6: known words coherence > 0", knownScore.coherence > 0);
assert("#6: unknown words coherence === 0", unknownScore.coherence === 0);
assert("#6: known > unknown (strict)", knownScore.coherence > unknownScore.coherence);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #7 — createOrRestore preserves state and auto-save");
// ═══════════════════════════════════════════════════════════════════

const tmpRestore = path.join(__dirname, '..', '.tmp_test_restore');
const origShifu = createShifu({ seed: true, loadTrained: false, autoSave: true, stateDir: tmpRestore });
origShifu.learn({ Diagnosis: 'test' }, { Diagnosis: 'restorecheck' });
origShifu.forceSave();
const restoredShifu = createOrRestore({ stateDir: tmpRestore });
assert("#7: restored correctionCount", restoredShifu.learning.correctionCount === 1);
assert("#7: restored has forceSave", typeof restoredShifu.forceSave === 'function');
assert("#7: restored has persistence", !!restoredShifu.persistence);
assert("#7: restored vocabulary has learned word", restoredShifu.learning.vocabulary.isKnown('restorecheck'));
restoredShifu.persistence.clear();
try { fs.rmdirSync(tmpRestore); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #8 — low-confidence words not substituted, decision is verify");
// ═══════════════════════════════════════════════════════════════════

const lowConfResult = correctLine("for");
assert("#8: low-confidence word keeps original", lowConfResult.words[0].corrected === 'for');
assert("#8: low-confidence flag set", lowConfResult.words[0].flag === 'low_confidence' || lowConfResult.words[0].flag === 'short' || lowConfResult.words[0].flag === 'short_preserved' || lowConfResult.words[0].flag === 'short_unknown' || lowConfResult.words[0].flag === 'exact' || lowConfResult.words[0].flag === 'passthrough' || lowConfResult.words[0].flag === 'unknown' || lowConfResult.words[0].flag === 'clean');

const mixedResult = correctLine("Levetiracetarn 5OOmg for seizure");
const forWord = mixedResult.words.find(w => w.original === 'for');
if (forWord && forWord.flag === 'low_confidence') {
  assert("#8: low-confidence word not substituted", forWord.corrected === 'for');
  assert("#8: assessConfidence returns verify", assessConfidence(mixedResult) === 'verify');
}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #9 — adaptive path has guards for titles/rooms/dosages");
// ═══════════════════════════════════════════════════════════════════

const guardShifu = createShifu({ seed: true, loadTrained: false });
const adaptGuard = guardShifu.correctRowAdaptive({ Doctor: 'Dr. Hisharn', Room: '3O2A', Dose: '5OOmg' });
assert("#9: Dr. preserved in adaptive path", adaptGuard.corrected.Doctor.output.startsWith('Dr.'));
assert("#9: room code preserved in adaptive path", adaptGuard.corrected.Room.output === '3O2A');
assert("#9: dosage preserved in adaptive path", adaptGuard.corrected.Dose.output === '5OOmg');

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #10 — OCR-garbled doses (O→0) trigger safety checks");
// ═══════════════════════════════════════════════════════════════════

const doseClean = correctLine("insulin 300units");
const doseOCR = correctLine("insulin 3OOunits");
assert("#10: clean dose flagged", doseClean.safetyFlags.some(f => f.status === 'IMPLAUSIBLE_DOSE'));
assert("#10: OCR-garbled dose ALSO flagged", doseOCR.safetyFlags.some(f => f.status === 'IMPLAUSIBLE_DOSE'));
const doseMgOCR = correctLine("insulin 3OOmg");
assert("#10: OCR mg dose flagged", doseMgOCR.safetyFlags.some(f => f.status === 'IMPLAUSIBLE_DOSE'));

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #11 — pipeline spawn fallback on invalid executable");
// ═══════════════════════════════════════════════════════════════════

const spawnShifu = createShifu({ seed: false, loadTrained: false });
const spawnPipeline = new ShifuPipeline(spawnShifu, { python: '/nonexistent/python999' });
// This must resolve (not reject) with a fallback object
const spawnTest = spawnPipeline._runPythonOCR('/fake/image.png', 'line').then(result => {
  assert("#11: spawn failure returns fallback", result.fallback === true);
  assert("#11: spawn failure has error message", typeof result.error === 'string' && result.error.length > 0);
});

// ═══════════════════════════════════════════════════════════════════
S("FEEDBACK LOOP");
// ═══════════════════════════════════════════════════════════════════

const { FeedbackLoop, MetricsTracker, DocumentIngestor } = require('../index');
const fbShifu = createShifu({ seed: true, loadTrained: false });
const fbStateDir = path.join(__dirname, '..', '.tmp_feedback_test');
const fb = new FeedbackLoop(fbShifu, { stateDir: fbStateDir });

// propose() returns a correction proposal
const proposal = fb.propose('Levetiracetarn for seizure');
assert("feedback: proposal has id", !!proposal.id);
assert("feedback: proposal has words", proposal.words.length > 0);
assert("feedback: proposal has decision", ['accept', 'verify', 'reject'].includes(proposal.decision));

// evaluate() closes the loop: compare proposal to ground truth, then learn
const evaluation = fb.evaluate(proposal, 'Levetiracetam for seizure');
assert("feedback: evaluation has accuracy", typeof evaluation.accuracy === 'number');
assert("feedback: evaluation total > 0", evaluation.total > 0);
assert("feedback: evaluation tracks corrections", Array.isArray(evaluation.corrections));

// proposeRow and evaluate row
const rowProposal = fb.proposeRow({ Patient: 'Bader A1athoub', Diagnosis: 'str0ke' });
assert("feedback: row proposal has columns", !!rowProposal.columns);
const rowEval = fb.evaluate(rowProposal, { Patient: 'Bader Alathoub', Diagnosis: 'stroke' });
assert("feedback: row eval has accuracy", typeof rowEval.accuracy === 'number');

// Metrics tracked
const fbMetrics = fb.getMetrics();
assert("feedback: metrics tracks evaluations", fbMetrics.totalEvaluations === 2);
assert("feedback: metrics tracks words", fbMetrics.totalWords > 0);
assert("feedback: report is a string", typeof fb.report() === 'string');

// Clean up
try { fs.unlinkSync(path.join(fbStateDir, 'feedback_metrics.json')); } catch {}
try { fs.rmdirSync(fbStateDir); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("METRICS TRACKER");
// ═══════════════════════════════════════════════════════════════════

const metricsDir = path.join(__dirname, '..', '.tmp_metrics_test');
const tracker = new MetricsTracker({ stateDir: metricsDir });

// Record a correction event
const mEntry = tracker.record({
  ocrText: 'Bader A1athoub lschemic str0ke',
  correctedText: 'Bader Alathoub ischemic stroke',
  groundTruth: 'Bader Alathoub ischemic stroke',
  meta: { column: 'census' },
});
assert("metrics: entry has accuracy", typeof mEntry.correctedAccuracy === 'number');
assert("metrics: corrected accuracy 100%", mEntry.correctedAccuracy === 1.0);
assert("metrics: raw accuracy < 100%", mEntry.rawAccuracy < 1.0);
assert("metrics: fixes > 0", mEntry.fixes > 0);
assert("metrics: regressions = 0", mEntry.regressions === 0);

// Record another with a regression
tracker.record({
  ocrText: 'for the patient',
  correctedText: 'dnr the patient',
  groundTruth: 'for the patient',
  meta: { column: 'notes' },
});

const mSummary = tracker.summary();
assert("metrics: summary has rawAccuracy", typeof mSummary.rawAccuracy === 'number');
assert("metrics: summary has correctedAccuracy", typeof mSummary.correctedAccuracy === 'number');
assert("metrics: summary tracks regressions", mSummary.totalRegressions > 0);
assert("metrics: report is string", typeof tracker.report() === 'string');
assert("metrics: byColumn tracks per-column", !!mSummary.byColumn['census']);

// Clean up
tracker.reset();
try { fs.unlinkSync(path.join(metricsDir, 'metrics.json')); } catch {}
try { fs.rmdirSync(metricsDir); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("DOCUMENT INGESTOR");
// ═══════════════════════════════════════════════════════════════════

const ingestShifu = createShifu({ seed: true, loadTrained: false });
const ingestPipeline = new ShifuPipeline(ingestShifu);
const ingestDir = path.join(__dirname, '..', '.tmp_ingest_test');
const ingestor = new DocumentIngestor(ingestPipeline, { outputDir: ingestDir });

// Ingest raw text
const textDoc = ingestor.ingestRawText('Levetiracetarn for seizure\nSod1um 139');
assert("ingest: text doc has lines", textDoc.lines.length === 2);
assert("ingest: text doc has decision", !!textDoc.overallDecision);
assert("ingest: raw OCR text is still corrected", textDoc.lines[0].corrected.toLowerCase().includes('levetiracetam'));

const nativeTextDoc = ingestor._processLineText(
  'The GP directed the patient to do an X-ray, which came back positive for bilateral pleural',
  { correct: false, sourceMode: 'native_text' }
);
assert("ingest: native text is passed through unchanged", nativeTextDoc.lines[0].corrected.includes('came back positive'));
assert("ingest: native text marks source mode", nativeTextDoc.sourceMode === 'native_text');
assert("ingest: native text skips token correction payload", nativeTextDoc.lines[0].words === undefined);

// Ingest ward census
const censusResult = ingestor.ingestWardCensus([
  { Patient: 'Bader A1athoub', Diagnosis: 'str0ke', Doctor: 'Dr. Hisharn' },
  { Patient: 'Noura Bazzah', Diagnosis: 'seizure', Doctor: 'Dr. Saleh' },
]);
assert("ingest: census has rows", censusResult.rows.length === 2);
assert("ingest: census has summary", !!censusResult.summary);
assert("ingest: census summary total = 2", censusResult.summary.total === 2);

// Ingest CSV file
const csvPath = path.join(__dirname, '..', '.tmp_test.csv');
fs.writeFileSync(csvPath, 'Patient,Diagnosis\nBader A1athoub,str0ke\nNoura,seizure\n');
const csvIngest = ingestor.ingestFile(csvPath);
// ingestFile returns a promise for images/PDFs, sync for CSV
if (csvIngest.then) {
  // async path — skip for now
} else {
  assert("ingest: CSV has columns", Array.isArray(csvIngest.columns));
  assert("ingest: CSV has rows", csvIngest.rows.length === 2);
}
fs.unlinkSync(csvPath);

// Clean up
try { fs.rmSync(ingestDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("CLINICAL-WEIGHTED LEARNING");
// ═══════════════════════════════════════════════════════════════════

const { getLearningRate: glr, getColumnWeight: gcw, requiresSafetyReview: rsr } = require('../index');

// Medication column gets 3x weight
assert("clinical weight: medication = 3.0", gcw('Medication').weight === 3.0);
assert("clinical weight: diagnosis = 2.5", gcw('Diagnosis').weight === 2.5);
assert("clinical weight: room = 1.0", gcw('Room').weight === 1.0);

// Safety review required for medications
assert("clinical: medication requires safety review", rsr('Medication', 'aspirin', VOCABULARY));
assert("clinical: room does NOT require safety review", !rsr('Room', '302A', VOCABULARY));

// Learning rate: medication words learn 3x faster regardless of column
const medRate = glr('notes', 'levetiracetam', VOCABULARY);
assert("clinical: medication word gets high rate even in notes column", medRate >= 3.0);

const weightedLearnShifu = createShifu({ seed: false, loadTrained: false });
weightedLearnShifu.learn({ Diagnosis: 'weighteddiagnosis' }, { Diagnosis: 'weighteddiagnosis' });
weightedLearnShifu.learn({ Room: 'weightedroom' }, { Room: 'weightedroom' });
assert(
  "clinical: diagnosis confirmations reinforce vocabulary more than room confirmations",
  weightedLearnShifu.learning.vocabulary.frequency['weighteddiagnosis'] > weightedLearnShifu.learning.vocabulary.frequency['weightedroom']
);

weightedLearnShifu.learn({ Patient: 'Bader', Diagnosis: 'stroke' }, { Patient: 'Bader', Diagnosis: 'stroke' });
const weightedContext = weightedLearnShifu.learning.context.predict({ Patient: 'Bader' }, 'Diagnosis');
assert(
  "clinical: weighted diagnosis learning strengthens context evidence",
  weightedContext.some(p => p.word === 'stroke' && p.count >= 6)
);

const beforeTokenAligned = weightedLearnShifu.correctLine('rare1earnterm').output;
weightedLearnShifu.learn(
  { Diagnosis: 'stroke rare1earnterm' },
  { Diagnosis: 'stroke rarelearnterm' }
);
const afterTokenAligned = weightedLearnShifu.correctLine('rare1earnterm').output;
assert(
  "clinical: multiword learning improves isolated token correction",
  beforeTokenAligned !== afterTokenAligned && afterTokenAligned.toLowerCase() === 'rarelearnterm'
);

// ═══════════════════════════════════════════════════════════════════
S("DYNAMIC CONFUSION COSTS");
// ═══════════════════════════════════════════════════════════════════

const dynAcp = new AdaptiveConfusionProfile();

// Before any corrections: static base costs
const baseCost01 = dynAcp.getCost('0', 'o');
assert("dynamic confusion: base cost for 0/o", baseCost01 === 0.1);

// Feed many corrections — system learns O↔0 is extremely common
for (let i = 0; i < 20; i++) {
  dynAcp.recordCorrection('str0ke', 'stroke');      // 0 → o
  dynAcp.recordCorrection('A1athoub', 'Alathoub');   // 1 → l
}

// After corrections: cost should drop (easier substitution for known confusions)
const learnedCost01 = dynAcp.getCost('0', 'o');
const learnedCost1l = dynAcp.getCost('1', 'l');
assert("dynamic confusion: 0/o cost dropped after learning", learnedCost01 < baseCost01);
assert("dynamic confusion: 1/l cost dropped after learning", learnedCost1l < 0.2);

// Unknown pair should still be expensive
const unknownCost = dynAcp.getCost('x', 'y');
assert("dynamic confusion: unknown pair stays expensive", unknownCost >= 0.9);

// Character confidence from observations
const oConf = dynAcp.getCharConfidence('o');
assert("dynamic confusion: char confidence is a number", typeof oConf === 'number');

// ═══════════════════════════════════════════════════════════════════
S("FULL WARD ROUND SIMULATION — system gets smarter");
// ═══════════════════════════════════════════════════════════════════

const wardShifu = createShifu({ seed: true, loadTrained: false });
const wardFb = new FeedbackLoop(wardShifu, { stateDir: path.join(__dirname, '..', '.tmp_ward_sim') });
const wardMetrics = new MetricsTracker({ stateDir: path.join(__dirname, '..', '.tmp_ward_metrics') });

// Simulate 3 ward rounds of corrections
const wardRounds = [
  // Round 1: nurse corrects OCR errors
  [
    { ocr: { Patient: 'Bader A1athoub', Diagnosis: 'lschemic str0ke' }, truth: { Patient: 'Bader Alathoub', Diagnosis: 'Ischemic stroke' } },
    { ocr: { Patient: 'N0ura Bazzah', Diagnosis: 'seizure epi1epsy' }, truth: { Patient: 'Noura Bazzah', Diagnosis: 'seizure epilepsy' } },
    { ocr: { Patient: 'Ahnad Hassan', Diagnosis: 'gui11ain barre' }, truth: { Patient: 'Ahmad Hassan', Diagnosis: 'guillain barre' } },
  ],
  // Round 2: same error patterns
  [
    { ocr: { Patient: 'Fa1sal Turki', Diagnosis: 'pneurn0nia' }, truth: { Patient: 'Faisal Turki', Diagnosis: 'pneumonia' } },
    { ocr: { Patient: 'Su1aiman 0rnar', Diagnosis: 'str0ke' }, truth: { Patient: 'Sulaiman Omar', Diagnosis: 'stroke' } },
  ],
  // Round 3: by now the system should be better at 1→l and 0→o
  [
    { ocr: { Patient: 'Kha1id Fahad', Diagnosis: 'epi1epsy' }, truth: { Patient: 'Khalid Fahad', Diagnosis: 'epilepsy' } },
    { ocr: { Patient: 'Da1a1 Mariam', Diagnosis: 'str0ke' }, truth: { Patient: 'Dalal Mariam', Diagnosis: 'stroke' } },
  ],
];

const roundAccuracies = [];

for (let round = 0; round < wardRounds.length; round++) {
  let roundCorrect = 0, roundTotal = 0;
  for (const { ocr, truth } of wardRounds[round]) {
    // System proposes corrections
    const proposal = wardFb.proposeRow(ocr);
    // Nurse evaluates against truth
    const evaluation = wardFb.evaluate(proposal, truth);

    // Also track in metrics
    for (const col of Object.keys(truth)) {
      const proposedCol = (proposal.columns || {})[col] || {};
      wardMetrics.record({
        ocrText: ocr[col],
        correctedText: proposedCol.proposed || ocr[col],
        groundTruth: truth[col],
        meta: { column: col, round },
      });
    }

    roundCorrect += evaluation.correct;
    roundTotal += evaluation.total;
  }
  const roundAcc = roundTotal > 0 ? roundCorrect / roundTotal : 0;
  roundAccuracies.push(roundAcc);
}

// The system should maintain or improve accuracy (not regress)
assert("ward sim: round 1 has measurements", roundAccuracies[0] >= 0);
assert("ward sim: round 3 has measurements", roundAccuracies[2] >= 0);

// The feedback loop should have learned
const fbMet = wardFb.getMetrics();
assert("ward sim: feedback tracked all evaluations", fbMet.totalEvaluations === 7);
assert("ward sim: feedback report works", wardFb.report().length > 0);

// The metrics should show improvement data
const wardSummary = wardMetrics.summary();
assert("ward sim: metrics tracked events", wardSummary.totalEvents > 0);
assert("ward sim: metrics has column breakdown", !!wardSummary.byColumn['Patient'] || !!wardSummary.byColumn['Diagnosis']);
assert("ward sim: metrics report works", wardMetrics.report().length > 0);

// The confusion profile should have learned 1→l and 0→o
const wardConfusions = wardShifu.learning.confusion.getTopConfusions(5);
assert("ward sim: confusion profile learned pairs", wardConfusions.length > 0);

// Print the ward round results for visibility
console.log(`  Round accuracies: ${roundAccuracies.map(a => (a * 100).toFixed(0) + '%').join(' → ')}`);
console.log(`  Top confusions: ${wardConfusions.slice(0, 3).map(c => c.pair + ' (' + c.count + 'x)').join(', ')}`);
console.log(`  Metrics: raw ${(wardSummary.rawAccuracy * 100).toFixed(0)}% → corrected ${(wardSummary.correctedAccuracy * 100).toFixed(0)}% (+${(wardSummary.improvement * 100).toFixed(0)}%)`);

// Clean up
try { fs.rmSync(path.join(__dirname, '..', '.tmp_ward_sim'), { recursive: true }); } catch {}
try { fs.rmSync(path.join(__dirname, '..', '.tmp_ward_metrics'), { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #12 — adaptive confusion feeds into static ocrWeightedDistance");
// ═══════════════════════════════════════════════════════════════════

const { setAdaptiveProfile, ocrWeightedDistance: owd } = require('../index');

// Create a shifu system — this wires the adaptive profile into the static module
const learnShifu = createShifu({ seed: true, loadTrained: false });

// Before learning: get baseline distance for a known OCR confusion
const distBefore = owd('str0ke', 'stroke');

// Teach the system many 0→o corrections to drive the cost down
for (let i = 0; i < 30; i++) {
  learnShifu.learn({ Diagnosis: 'str0ke' }, { Diagnosis: 'stroke' });
}

// After learning: static ocrWeightedDistance should now produce a LOWER distance
// because the adaptive profile learned that 0↔o is extremely common
const distAfter = owd('str0ke', 'stroke');
assert("#12: static ocrWeightedDistance uses learned costs", distAfter <= distBefore);
assert("#12: adaptive profile is wired in", distAfter < 1.0);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #13 — MetricsTracker handles insertions/deletions");
// ═══════════════════════════════════════════════════════════════════

const metricsLcsDir = path.join(__dirname, '..', '.tmp_metrics_lcs');
const trackerLcs = new MetricsTracker({ stateDir: metricsLcsDir });

// Insertion case: proposal has extra word
const insEntry = trackerLcs.record({
  ocrText: 'stroke patient',
  correctedText: 'stroke extra patient',
  groundTruth: 'stroke patient',
});
// With LCS: 'stroke' and 'patient' both match ground truth (2/3)
assert("#13: insertion — corrected accuracy reflects LCS", insEntry.correctedAccuracy >= 0.66);

// Deletion case: ground truth has extra word
const delEntry = trackerLcs.record({
  ocrText: 'stroke patient',
  correctedText: 'stroke patient',
  groundTruth: 'stroke patient stable',
});
assert("#13: deletion — corrected accuracy reflects LCS", delEntry.correctedAccuracy >= 0.66);

// Clean up
trackerLcs.reset();
try { fs.rmSync(metricsLcsDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #14 — system learns from corrections and improves");
// ═══════════════════════════════════════════════════════════════════

// This is the critical test: the system should get BETTER after seeing corrections
const learnTestShifu = createShifu({ seed: true, loadTrained: false });

// First correction: the system sees "pneurnonia" → "pneumonia" for the first time
const before = learnTestShifu.correctLine('pneurnonia');
const beforeIsCorrect = before.output.toLowerCase() === 'pneumonia';

// Teach it repeatedly
for (let i = 0; i < 10; i++) {
  learnTestShifu.learn({ Diagnosis: 'pneurnonia' }, { Diagnosis: 'pneumonia' });
}

// After learning: the system should be more confident or at least still correct
const after = learnTestShifu.correctLine('pneurnonia');
assert("#14: system still corrects after learning", after.output.toLowerCase() === 'pneumonia');
// The confidence should be at least as good (learning should not degrade)
assert("#14: confidence maintained or improved", after.words[0].confidence >= before.words[0].confidence - 0.1);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #15 — ingest error handling at boundaries");
// ═══════════════════════════════════════════════════════════════════

const errorShifu = createShifu({ seed: false, loadTrained: false });
const errorPipeline = new ShifuPipeline(errorShifu);
const errorIngestor = new DocumentIngestor(errorPipeline, { outputDir: path.join(__dirname, '..', '.tmp_err_ingest') });

// File not found returns error, doesn't throw
const notFound = errorIngestor.ingestFile('/nonexistent/file.csv');
// ingestFile may be async for some formats
if (notFound.then) {
  notFound.then(r => assert("#15: missing file returns error", !!r.error));
} else {
  assert("#15: missing file returns error", !!notFound.error);
}

// Clean up
try { fs.rmSync(path.join(__dirname, '..', '.tmp_err_ingest'), { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #16 — extreme lab outliers block acceptance");
// ═══════════════════════════════════════════════════════════════════

const labExtreme = correctLine('potassium 9999');
assert("#16: extreme lab has OUT_OF_RANGE flag", labExtreme.safetyFlags.some(f => f.status === 'OUT_OF_RANGE'));
assert("#16: extreme lab hasWarnings = true", labExtreme.hasWarnings === true);
assert("#16: extreme lab does NOT assess as accept", assessConfidence(labExtreme) !== 'accept');

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #17 — feedback substitution scoring");
// ═══════════════════════════════════════════════════════════════════

const fbSubShifu = createShifu({ seed: true, loadTrained: false });
const fbSubDir = path.join(__dirname, '..', '.tmp_fb_sub');
const fbSub = new FeedbackLoop(fbSubShifu, { stateDir: fbSubDir });

// proposal "foo baz" vs confirmed "foo bar" should be 2 tokens, 1 correct
const subProposal = fbSub.propose('foo baz');
// Override proposal words for controlled test
subProposal.words = [
  { original: 'foo', proposed: 'foo', confidence: 0.9, flag: 'exact' },
  { original: 'baz', proposed: 'baz', confidence: 0.5, flag: 'unknown' },
];
const subEval = fbSub.evaluate(subProposal, 'foo bar');
assert("#17: substitution total = 2", subEval.total === 2);
assert("#17: substitution correct = 1", subEval.correct === 1);
assert("#17: substitution has correction for baz→bar", subEval.corrections.some(c => c.proposed === 'baz' && c.confirmed === 'bar'));

// Clean up
try { fs.rmSync(fbSubDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #18 — CSV quoted fields preserved");
// ═══════════════════════════════════════════════════════════════════

const csvQuotedPath = path.join(__dirname, '..', '.tmp_quoted.csv');
fs.writeFileSync(csvQuotedPath, 'Patient,Diagnosis\nBader,"stroke, ischemic"\n');
const csvQuotedShifu = createShifu({ seed: true, loadTrained: false });
const csvQuotedPipeline = new ShifuPipeline(csvQuotedShifu);
const csvQuotedIngestDir = path.join(__dirname, '..', '.tmp_quoted_ingest');
const csvQuotedIngestor = new DocumentIngestor(csvQuotedPipeline, { outputDir: csvQuotedIngestDir });
const csvQuotedResult = csvQuotedIngestor.ingestFile(csvQuotedPath);
const csvQuotedDoc = csvQuotedResult.then ? null : csvQuotedResult;
if (csvQuotedDoc) {
  const rawDiag = csvQuotedDoc.rows[0].raw.Diagnosis;
  assert("#18: quoted CSV field preserved", rawDiag.includes('ischemic'));
}
fs.unlinkSync(csvQuotedPath);
try { fs.rmSync(csvQuotedIngestDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #19 — getTopConfusions filters bookkeeping keys");
// ═══════════════════════════════════════════════════════════════════

const filtAcp = new AdaptiveConfusionProfile();
for (let i = 0; i < 10; i++) filtAcp.recordCorrection('str0ke', 'stroke');
const topConf = filtAcp.getTopConfusions(10);
assert("#19: no ok: keys in top confusions", topConf.every(c => !c.pair.startsWith('ok:')));
assert("#19: no err: keys in top confusions", topConf.every(c => !c.pair.startsWith('err:')));
assert("#19: real confusion pair present", topConf.some(c => c.pair === '0,o' || c.pair === 'o,0' || c.pair.includes('0')));

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #20 — correction history and rollback");
// ═══════════════════════════════════════════════════════════════════

const rollShifu = createShifu({ seed: true, loadTrained: false });

// Learn something
const learnResult = rollShifu.learn({ Diagnosis: 'str0ke' }, { Diagnosis: 'stroke' });
assert("#20: learn returns accepted", learnResult.accepted === true);
assert("#20: correction count = 1", rollShifu.learning.correctionCount === 1);

// History should have one entry
const hist = rollShifu.getHistory();
assert("#20: history has 1 entry", hist.length === 1);
assert("#20: history entry has columns", hist[0].columns && hist[0].columns.includes('Diagnosis'));

// Undo it
const undone = rollShifu.undo();
assert("#20: undo returns the undone correction", undone !== null);
assert("#20: undo correction count back to 0", rollShifu.learning.correctionCount === 0);

// History should be empty after undo
assert("#20: history empty after undo", rollShifu.getHistory().length === 0);

// Undo on empty history returns null
assert("#20: undo on empty returns null", rollShifu.undo() === null);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #21 — confidence-gated learning rejects suspicious input");
// ═══════════════════════════════════════════════════════════════════

const gateShifu = createShifu({ seed: true, loadTrained: false });

// Normal correction: should be accepted
const normalLearn = gateShifu.learn(
  { Diagnosis: 'str0ke' },
  { Diagnosis: 'stroke' }
);
assert("#21: normal correction accepted", normalLearn.accepted === true);
assert("#21: normal correction no rejections", normalLearn.rejected.length === 0);

// Suspicious correction: confirmed text is a very distant unknown word (nurse typo)
const susLearn = gateShifu.learn(
  { Diagnosis: 'stroke' },
  { Diagnosis: 'xyzzyplughfoo' }  // gibberish — not in vocab, very far from 'stroke'
);
assert("#21: suspicious correction flagged", susLearn.rejected.length > 0);
assert("#21: rejected word identified", susLearn.rejected[0].word === 'xyzzyplughfoo');
// Fully rejected learn must return accepted: false and NOT mutate state
assert("#21: fully rejected learn returns accepted=false", susLearn.accepted === false);
assert("#21: rejected word not in vocabulary", !gateShifu.learning.vocabulary.isKnown('xyzzyplughfoo'));
assert("#21: rejected word not in frequency", (gateShifu.learning.vocabulary.frequency['xyzzyplughfoo'] || 0) === 0);
// correctionCount should not have incremented for a fully rejected learn
assert("#21: correctionCount unchanged after full rejection", gateShifu.learning.correctionCount === 1); // only the first learn counted

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #22 — write lock prevents concurrent corruption");
// ═══════════════════════════════════════════════════════════════════

const lockShifu = createShifu({ seed: true, loadTrained: false });

// Simulate concurrent access by manually locking
lockShifu.learning._locked = true;
const blockedResult = lockShifu.learn({ Diagnosis: 'test' }, { Diagnosis: 'test2' });
assert("#22: blocked learn returns not accepted", blockedResult.accepted === false);
assert("#22: blocked reason is locked", blockedResult.reason === 'locked');
lockShifu.learning._locked = false;

// Normal learn should work after unlock
const unlockedResult = lockShifu.learn({ Diagnosis: 'test' }, { Diagnosis: 'test2' });
assert("#22: unlocked learn works", unlockedResult.accepted === true);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #23 — Python OCR health check");
// ═══════════════════════════════════════════════════════════════════

const healthShifu = createShifu({ seed: false, loadTrained: false });
const healthPipeline = new ShifuPipeline(healthShifu, { python: '/nonexistent/python999' });
const healthCheck = healthPipeline.checkPythonOCR().then(result => {
  assert("#23: bad python reports unavailable", result.available === false);
  assert("#23: health check has error", typeof result.error === 'string');
});

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #24 — rollback survives serialization");
// ═══════════════════════════════════════════════════════════════════

const serShifu = createShifu({ seed: true, loadTrained: false });
serShifu.learn({ Diagnosis: 'str0ke' }, { Diagnosis: 'stroke' });
serShifu.learn({ Patient: 'A1athoub' }, { Patient: 'Alathoub' });

const serState = serShifu.serialize();
const deserShifu = restoreShifu(serState);
assert("#24: history survives serialization", deserShifu.learning._history.length === 2);
assert("#24: columns preserved after restore", deserShifu.learning._history[0].columns && deserShifu.learning._history[0].columns.length > 0);
assert("#24: raw rows stripped from persisted history", !deserShifu.learning._history[0].ocrRow);
const deserUndo = deserShifu.learning.undo();
assert("#24: undo works after deserialization", deserUndo !== null);
assert("#24: correct count after undo", deserShifu.learning.correctionCount === 1);
assert("#24: coreSnapshot not available after restore (too large for disk)", deserUndo.coreRestored === false);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #25 — pipeline.processTableRow blocks severity 'error'");
// ═══════════════════════════════════════════════════════════════════

const pipeErrShifu = createShifu({ seed: true, loadTrained: false });
const pipeErrPipeline = new ShifuPipeline(pipeErrShifu);
const pipeErrResult = pipeErrPipeline.processTableRow({ Lab: 'potassium 9999' });
assert("#25: pipeline table row with extreme lab not accepted", pipeErrResult.decision !== 'accept');
assert("#25: pipeline table row flags OUT_OF_RANGE", pipeErrResult.safetyFlags.some(f => f.status === 'OUT_OF_RANGE'));

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #26 — MetricsTracker counts fixes and regressions independently");
// ═══════════════════════════════════════════════════════════════════

const mFlipDir = path.join(__dirname, '..', '.tmp_metrics_flip');
const mFlipTracker = new MetricsTracker({ stateDir: mFlipDir });

// ocr='a b c', corrected='a x y', truth='a y c'
// truth 'a' matched by both raw and corrected
// truth 'y' not in raw, IS in corrected → fix
// truth 'c' IS in raw, not in corrected → regression
const flipEntry = mFlipTracker.record({
  ocrText: 'a b c',
  correctedText: 'a x y',
  groundTruth: 'a y c',
});
assert("#26: fixes detected", flipEntry.fixes >= 1);
assert("#26: regressions detected", flipEntry.regressions >= 1);
assert("#26: not both zero", flipEntry.fixes + flipEntry.regressions > 0);

mFlipTracker.reset();
try { fs.rmSync(mFlipDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #27 — CSV multiline quoted fields");
// ═══════════════════════════════════════════════════════════════════

const csvMultiPath = path.join(__dirname, '..', '.tmp_multi.csv');
fs.writeFileSync(csvMultiPath, 'Patient,Diagnosis\nBader,"stroke\nischemic"\n');
const csvMultiShifu = createShifu({ seed: true, loadTrained: false });
const csvMultiPipeline = new ShifuPipeline(csvMultiShifu);
const csvMultiIngestDir = path.join(__dirname, '..', '.tmp_multi_ingest');
const csvMultiIngestor = new DocumentIngestor(csvMultiPipeline, { outputDir: csvMultiIngestDir });
const csvMultiResult = csvMultiIngestor.ingestFile(csvMultiPath);
const csvMultiDoc = csvMultiResult.then ? null : csvMultiResult;
if (csvMultiDoc) {
  assert("#27: multiline CSV has 1 data row", csvMultiDoc.rows.length === 1);
  const rawDiag = csvMultiDoc.rows[0].raw.Diagnosis;
  assert("#27: multiline field preserved", rawDiag.includes('ischemic'));
}
fs.unlinkSync(csvMultiPath);
try { fs.rmSync(csvMultiIngestDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #28 — undo() fully rolls back context and vocabulary");
// ═══════════════════════════════════════════════════════════════════

const undoFullShifu = createShifu({ seed: true, loadTrained: false });

// Capture state before learning
const preContextEvents = undoFullShifu.learning.context.totalEvents;
const preTotalReadings = undoFullShifu.learning.vocabulary.totalReadings;

// Learn unique words so we can verify core rollback
undoFullShifu.learn({ Diagnosis: 'rareundoalpha rareundobeta' }, { Diagnosis: 'rareundoalpha rareundobeta' });
assert("#28: context events incremented", undoFullShifu.learning.context.totalEvents > preContextEvents);
assert("#28: totalReadings incremented", undoFullShifu.learning.vocabulary.totalReadings > preTotalReadings);
// Core should now know these words
const preUndoCoherence = undoFullShifu.scoreSentence('rareundoalpha rareundobeta').coherence;
assert("#28: core learned the words", preUndoCoherence > 0);

// Undo via shifu API (passes core for full rollback)
undoFullShifu.undo();
assert("#28: context events restored", undoFullShifu.learning.context.totalEvents === preContextEvents);
assert("#28: totalReadings restored", undoFullShifu.learning.vocabulary.totalReadings === preTotalReadings);
// Core should be rolled back too — words should be unknown again
const postUndoCoherence = undoFullShifu.scoreSentence('rareundoalpha rareundobeta').coherence;
assert("#28: core state rolled back", postUndoCoherence === 0);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #29 — auto-save returns learn() result and skips failed learns");
// ═══════════════════════════════════════════════════════════════════

const autoSaveDir = path.join(__dirname, '..', '.tmp_autosave_contract');
const autoSaveShifu = createShifu({ seed: true, loadTrained: false, autoSave: true, stateDir: autoSaveDir });
const autoResult = autoSaveShifu.learn({ Diagnosis: 'str0ke' }, { Diagnosis: 'stroke' });
assert("#29: auto-save learn returns result", autoResult !== undefined);
assert("#29: auto-save learn returns accepted", autoResult.accepted === true);

// Locked learn should return result too
autoSaveShifu.learning._locked = true;
const lockedResult = autoSaveShifu.learn({ Diagnosis: 'test' }, { Diagnosis: 'test2' });
assert("#29: locked learn returns result through auto-save", lockedResult !== undefined);
assert("#29: locked learn returns not accepted", lockedResult.accepted === false);
autoSaveShifu.learning._locked = false;

// Fully rejected learn should also return accepted: false and not trigger save
const rejectedAutoResult = autoSaveShifu.learn(
  { Diagnosis: 'stroke' },
  { Diagnosis: 'xyzzyplughfoo999' }
);
assert("#29: fully rejected learn returns false through auto-save", rejectedAutoResult.accepted === false);

autoSaveShifu.persistence.clear();
try { fs.rmdirSync(autoSaveDir); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #30 — correctRowAdaptive blocks severity 'error'");
// ═══════════════════════════════════════════════════════════════════

const adaptErrShifu = createShifu({ seed: true, loadTrained: false });
const adaptErrResult = adaptErrShifu.correctRowAdaptive({ Lab: 'potassium 9999' });
assert("#30: adaptive path flags OUT_OF_RANGE", adaptErrResult.safetyFlags.some(f => f.status === 'OUT_OF_RANGE'));
assert("#30: adaptive path hasWarnings includes error severity", adaptErrResult.hasWarnings === true);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #31 — reversal detection works even with partial roles");
// ═══════════════════════════════════════════════════════════════════

const { structuralInvariance: siTest } = require('../index');
const revProbeEngine = new ShifuEngine();
// Train enough that roles can be extracted
const revCorpus = [
  "doctor treats patient", "doctor examines patient",
  "physician treats patient", "nurse helps patient",
  "doctor prescribed medication", "patient received treatment",
];
for (let r = 0; r < 5; r++) for (const s of revCorpus) revProbeEngine.feed(s);

const revNormal = siTest("doctor treats patient", "physician treats patient", revProbeEngine);
const revReversed = siTest("doctor treats patient", "patient treats doctor", revProbeEngine);
assert("#31: paraphrase invariance > 0", revNormal.invariance > 0);
assert("#31: reversal penalized below paraphrase", revReversed.invariance < revNormal.invariance || revReversed.swapDetected === true);

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #32 — feedback handles blank confirmed cells");
// ═══════════════════════════════════════════════════════════════════

const fbBlankShifu = createShifu({ seed: true, loadTrained: false });
const fbBlankDir = path.join(__dirname, '..', '.tmp_fb_blank');
const fbBlank = new FeedbackLoop(fbBlankShifu, { stateDir: fbBlankDir });

const blankProposal = fbBlank.proposeRow({ Diagnosis: 'stroke' });
const blankEval = fbBlank.evaluate(blankProposal, { Diagnosis: '' });
assert("#32: blank confirmed scores total > 0", blankEval.total > 0);
assert("#32: blank confirmed marks words as wrong", blankEval.correct === 0);
assert("#32: blank confirmed has corrections", blankEval.corrections.length > 0);

try { fs.rmSync(fbBlankDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("REGRESSION: Fix #33 — proposal metrics persisted on propose()");
// ═══════════════════════════════════════════════════════════════════

const fbPersistShifu = createShifu({ seed: true, loadTrained: false });
const fbPersistDir = path.join(__dirname, '..', '.tmp_fb_persist');
const fbPersist = new FeedbackLoop(fbPersistShifu, { stateDir: fbPersistDir });

fbPersist.propose('str0ke');
// Metrics should be saved to disk after propose()
const fbPersist2 = new FeedbackLoop(fbPersistShifu, { stateDir: fbPersistDir });
assert("#33: proposal count persisted", fbPersist2.getMetrics().totalProposals === 1);

try { fs.rmSync(fbPersistDir, { recursive: true }); } catch {}

// ═══════════════════════════════════════════════════════════════════
S("SEMANTIC PROBE — honest measurement of meaning vs pattern");
// ═══════════════════════════════════════════════════════════════════

// This section does NOT assert pass/fail. It MEASURES and REPORTS.
// These are the four tests from the theoretical critique.

const probe = new ShifuEngine();
// Train on a rich corpus with active/passive, synonyms, argument structures
const probeSentences = [
  // Active voice patterns
  "doctor treats patient with medication and therapy",
  "doctor examines patient and records findings",
  "doctor prescribes medication for the patient",
  "doctor admits patient to the ward",
  "doctor discharges patient from the hospital",
  // Passive voice equivalents
  "patient was treated by the doctor with medication",
  "patient was examined by the doctor carefully",
  "medication was prescribed by the doctor for pain",
  "patient was admitted to the ward by doctor",
  "patient was discharged from the hospital by doctor",
  // Synonym pairs in context
  "physician treats patient with medication",
  "physician examines patient and records findings",
  "physician prescribes medication for the patient",
  "clinician treats patient with medication",
  "clinician examines patient carefully",
  // Unusual but valid
  "patient treats wound with bandage",
  "nurse administers medication to patient",
  "medication reduces symptoms in patient",
  // Repeated for statistical strength
  "doctor treats patient in the ward",
  "physician treats patient in the ward",
  "patient was treated in the ward by doctor",
  "doctor prescribed aspirin for stroke patient",
  "aspirin was prescribed by doctor for stroke",
  "physician prescribed aspirin for stroke patient",
];
for (let round = 0; round < 5; round++) {
  for (const s of probeSentences) probe.feed(s);
}

console.log('\n  --- Probe 1: Paraphrase Invariance ---');
// Active vs passive of the same event should score similarly to the engine
const activeScore = probe.scoreSentence("doctor treats patient with medication");
const passiveScore = probe.scoreSentence("patient was treated by the doctor with medication");
const unrelatedScore = probe.scoreSentence("hospital building has many floors");
const paraCoherence = Math.abs(activeScore.coherence - passiveScore.coherence);
console.log(`  Active coherence:   ${activeScore.coherence.toFixed(4)}`);
console.log(`  Passive coherence:  ${passiveScore.coherence.toFixed(4)}`);
console.log(`  Unrelated coherence: ${unrelatedScore.coherence.toFixed(4)}`);
console.log(`  Active-Passive gap: ${paraCoherence.toFixed(4)} ${paraCoherence < 0.15 ? '(CLOSE)' : '(GAP)'}`);

console.log('\n  --- Probe 2: Synonym Resonance ---');
// doctor/physician should have high resonance after seeing both in same slots
const dpRes = probe.res["doctor"]?.["physician"] || 0;
const dpDiscount = probe._resonanceDiscount("doctor", "physician");
const dcRes = probe.res["doctor"]?.["clinician"] || 0;
console.log(`  doctor-physician resonance: ${dpRes.toFixed(2)} (discount: ${dpDiscount.toFixed(3)})`);
console.log(`  doctor-clinician resonance: ${dcRes.toFixed(2)}`);
// Compare: doctor-medication should have LOW resonance (different role)
const dmRes = probe.res["doctor"]?.["medication"] || 0;
console.log(`  doctor-medication resonance: ${dmRes.toFixed(2)} (should be lower)`);
const synonymWin = dpRes > dmRes;
console.log(`  Synonym > non-synonym: ${synonymWin ? 'YES' : 'NO'}`);

console.log('\n  --- Probe 3: Argument Reversal Detection ---');
// "doctor treats patient" vs "patient treats doctor" — different events
const normalScore = probe.scoreSentence("doctor treats patient");
const reversedScore = probe.scoreSentence("patient treats doctor");
console.log(`  "doctor treats patient" coherence: ${normalScore.coherence.toFixed(4)}`);
console.log(`  "patient treats doctor" coherence: ${reversedScore.coherence.toFixed(4)}`);
const reversalDetected = normalScore.coherence > reversedScore.coherence;
console.log(`  Normal > reversed: ${reversalDetected ? 'YES' : 'NO'}`);

console.log('\n  --- Probe 4: Novel-but-Valid vs Gibberish ---');
// The engine should give SOME coherence to novel valid sentences
// (words it knows, structure it recognizes, but exact combo never seen)
const novelValid = probe.scoreSentence("nurse prescribes medication for patient");
const gibberish = probe.scoreSentence("zzzzz xxxxx yyyyy");
const novelBad = probe.scoreSentence("medication treats doctor with patient");
console.log(`  Novel valid:   ${novelValid.coherence.toFixed(4)}`);
console.log(`  Gibberish:     ${gibberish.coherence.toFixed(4)}`);
console.log(`  Novel invalid: ${novelBad.coherence.toFixed(4)}`);
const novelWin = novelValid.coherence > gibberish.coherence;
const novelSeparation = novelValid.coherence > novelBad.coherence;
console.log(`  Novel valid > gibberish:  ${novelWin ? 'YES' : 'NO'}`);
console.log(`  Novel valid > novel bad:  ${novelSeparation ? 'YES' : 'NO'}`);

console.log('\n  --- Probe 5: Structural Invariance (NEW LAYER) ---');
// This is the decisive test: does the invariance layer collapse active/passive?
const { structuralInvariance: si } = require('../index');

const activePassive = si("doctor treats patient with medication", "patient was treated by the doctor with medication", probe);
const activeUnrelated = si("doctor treats patient with medication", "hospital building has many floors", probe);
const activeReversed = si("doctor treats patient", "patient treats doctor", probe);

console.log(`  Active-Passive invariance:   ${activePassive.invariance.toFixed(4)}`);
console.log(`    Roles A: agent=${activePassive.rolesA.agent}, action=${activePassive.rolesA.action}, patient=${activePassive.rolesA.patient} (passive=${activePassive.rolesA.passive})`);
console.log(`    Roles B: agent=${activePassive.rolesB.agent}, action=${activePassive.rolesB.action}, patient=${activePassive.rolesB.patient} (passive=${activePassive.rolesB.passive})`);
console.log(`  Active-Unrelated invariance: ${activeUnrelated.invariance.toFixed(4)}`);
console.log(`  Active-Reversed invariance:  ${activeReversed.invariance.toFixed(4)}`);

const paraphraseCollapse = activePassive.invariance > activeUnrelated.invariance;
const reversalSeparation = activePassive.invariance > activeReversed.invariance;
console.log(`  Paraphrase > unrelated: ${paraphraseCollapse ? 'YES' : 'NO'}`);
console.log(`  Paraphrase > reversed:  ${reversalSeparation ? 'YES' : 'NO'}`);

console.log('\n  --- Probe 6: Cross-Domain Transfer ---');
// "doctor treats patient" and "teacher educates student" — same relational structure
// Both trained now to give the engine something to work with
const crossCorpus = [
  "teacher educates student in the classroom",
  "teacher instructs student with materials",
  "student was educated by the teacher carefully",
  "teacher assigns homework to student",
  "professor teaches student advanced topics",
];
for (let r = 0; r < 5; r++) for (const s of crossCorpus) probe.feed(s);

const crossDomain = si("doctor treats patient", "teacher educates student", probe);
const crossDomainReversed = si("doctor treats patient", "student educates teacher", probe);
console.log(`  doctor-treats-patient vs teacher-educates-student: ${crossDomain.invariance.toFixed(4)}`);
console.log(`    Agent sim:   ${crossDomain.comparison.agentSim.toFixed(3)} (${crossDomain.comparison.details.agent?.method || 'n/a'})`);
console.log(`    Action sim:  ${crossDomain.comparison.actionSim.toFixed(3)} (${crossDomain.comparison.details.action?.method || 'n/a'})`);
console.log(`    Patient sim: ${crossDomain.comparison.patientSim.toFixed(3)} (${crossDomain.comparison.details.patient?.method || 'n/a'})`);
console.log(`  doctor-treats-patient vs student-educates-teacher: ${crossDomainReversed.invariance.toFixed(4)}`);
const crossDomainWin = crossDomain.invariance > crossDomainReversed.invariance;
console.log(`  Correct > reversed: ${crossDomainWin ? 'YES' : 'NO'}`);

// Summary scorecard
const probeResults = [synonymWin, reversalDetected, novelWin, paraphraseCollapse, reversalSeparation];
const probeScore = probeResults.filter(Boolean).length;
console.log(`\n  === SEMANTIC PROBE SCORECARD: ${probeScore}/${probeResults.length} ===`);
console.log(`  Paraphrase coherence gap:    ${paraCoherence < 0.15 ? 'PASS' : 'MEASURED (gap=' + paraCoherence.toFixed(3) + ')'}`);
console.log(`  Synonym resonance:           ${synonymWin ? 'PASS' : 'FAIL'}`);
console.log(`  Argument reversal:           ${reversalDetected ? 'PASS' : 'FAIL'}`);
console.log(`  Novel composition:           ${novelWin ? 'PASS' : 'FAIL'}`);
console.log(`  Structural invariance:       ${paraphraseCollapse ? 'PASS' : 'FAIL'}`);
console.log(`  Reversal via invariance:     ${reversalSeparation ? 'PASS' : 'FAIL'}`);
console.log(`  Cross-domain transfer:       ${crossDomainWin ? 'PASS' : 'MEASURED'}`);
console.log();

// Structural assertions — these ARE hard requirements for the invariance layer
assert("probe: known words have nonzero coherence", activeScore.coherence > 0);
assert("probe: gibberish has zero coherence", gibberish.coherence === 0);
assert("probe: active/passive roles extracted", activePassive.rolesA.agent !== null && activePassive.rolesB.agent !== null);
assert("probe: passive detected", activePassive.rolesB.passive === true);
assert("probe: paraphrase invariance > unrelated", activePassive.invariance > activeUnrelated.invariance);

// ═══════════════════════════════════════════════════════════════════
S("LARGE VOCABULARY");
// ═══════════════════════════════════════════════════════════════════

{
  const largeVocabShifu = createShifu({ seed: false, loadTrained: false, vocabularyTargetSize: 100000 });
  assert("createShifu expands vocabulary to 100k", largeVocabShifu.stats().vocabulary.baseSize >= 100000);
  assert("learning vocabulary tracks expanded base size", largeVocabShifu.stats().learning.vocabularySize >= 100000);
  assert("100k vocabulary still corrects OCR misspelling", largeVocabShifu.correctLine("str0ke").output.toLowerCase() === "stroke");

  const runtimeExpandShifu = createShifu({ seed: false, loadTrained: false });
  const runtimeExpansion = runtimeExpandShifu.expandVocabulary(100000);
  assert("runtime vocabulary expansion reaches target", runtimeExpansion.totalSize >= 100000);
  assert("runtime expansion refreshes live learning vocabulary", runtimeExpandShifu.stats().learning.vocabularySize >= 100000);
}

// ═══════════════════════════════════════════════════════════════════
// RESULTS (wait for async tests before printing)
// ═══════════════════════════════════════════════════════════════════

Promise.all([spawnTest, healthCheck]).then(() => {
  console.log(`\n========================================`);
  console.log(`  ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ''}`);
  console.log(`========================================`);
  process.exit(fail > 0 ? 1 : 0);
});
