// Shifu OCR — Local Interactive Server
// Run: node server.js
// Open: http://localhost:3737

const http = require('http');
const url = require('url');
const { createShifu, ShifuPipeline, FeedbackLoop, DocumentIngestor, MetricsTracker } = require('./index');

const PORT = 3737;
const VOCABULARY_TARGET_SIZE = Number(process.env.SHIFU_VOCAB_TARGET || 100000);

// Boot the system
console.log('Booting Shifu OCR...');
const shifu = createShifu({
  seed: true,
  loadTrained: true,
  autoSave: true,
  stateDir: '.state',
  vocabularyTargetSize: VOCABULARY_TARGET_SIZE,
});
const pipeline = shifu.createPipeline();
const feedback = new FeedbackLoop(shifu, { stateDir: '.state' });
const metrics = new MetricsTracker({ stateDir: '.state' });
const ingestor = new DocumentIngestor(pipeline, { outputDir: '.ingested' });
console.log('Shifu OCR ready.');

function jsonResponse(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': 'http://localhost:3737' });
  res.end(JSON.stringify(data, null, 2));
}

function parseBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { resolve({}); }
    });
  });
}

const server = http.createServer(async (req, res) => {
  const parsed = url.parse(req.url, true);
  const path = parsed.pathname;

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(200, { 'Access-Control-Allow-Origin': 'http://localhost:3737', 'Access-Control-Allow-Methods': 'GET,POST', 'Access-Control-Allow-Headers': 'Content-Type' });
    res.end();
    return;
  }

  // ── Serve the UI ──────────────────────────────────────────────
  if (path === '/' || path === '/index.html') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(HTML);
    return;
  }

  // ── API Routes ────────────────────────────────────────────────

  // Correct text (handles multi-line: splits on newlines, corrects each)
  if (path === '/api/correct' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    const lines = text.split('\n').filter(l => l.trim());
    if (lines.length === 1) {
      const result = shifu.correctLine(lines[0]);
      const decision = shifu.assessConfidence(result);
      const coherence = shifu.scoreSentence(result.output);
      return jsonResponse(res, { ...result, decision, coherence: coherence.coherence });
    }
    // Multi-line: correct each line independently
    const results = lines.map(line => {
      const r = shifu.correctLine(line);
      return { ...r, decision: shifu.assessConfidence(r) };
    });
    const allWords = results.flatMap(r => r.words);
    const allFlags = results.flatMap(r => r.safetyFlags);
    const output = results.map(r => r.output).join('\n');
    const avgConf = allWords.length > 0 ? allWords.reduce((s, w) => s + w.confidence, 0) / allWords.length : 0;
    const hasDangers = results.some(r => r.hasDangers);
    const hasWarnings = results.some(r => r.hasWarnings);
    const decision = hasDangers ? 'reject' : hasWarnings ? 'verify' : results.some(r => r.decision !== 'accept') ? 'verify' : 'accept';
    const coherence = shifu.scoreSentence(output);
    return jsonResponse(res, { input: text, output, words: allWords, safetyFlags: allFlags, avgConfidence: Math.round(avgConf * 100) / 100, hasDangers, hasWarnings, decision, coherence: coherence.coherence });
  }

  // Correct a table row (ward census)
  if (path === '/api/correct-row' && req.method === 'POST') {
    const { row } = await parseBody(req);
    if (!row) return jsonResponse(res, { error: 'row required' }, 400);
    const result = pipeline.processTableRow(row);
    return jsonResponse(res, result);
  }

  // Propose + evaluate (feedback loop)
  if (path === '/api/propose' && req.method === 'POST') {
    const { text, row } = await parseBody(req);
    if (text) return jsonResponse(res, feedback.propose(text));
    if (row) return jsonResponse(res, feedback.proposeRow(row));
    return jsonResponse(res, { error: 'text or row required' }, 400);
  }

  if (path === '/api/evaluate' && req.method === 'POST') {
    const { proposal, confirmed } = await parseBody(req);
    if (!proposal || confirmed === undefined) return jsonResponse(res, { error: 'proposal and confirmed required' }, 400);
    const result = feedback.evaluate(proposal, confirmed);
    return jsonResponse(res, result);
  }

  // Learn from a correction
  if (path === '/api/learn' && req.method === 'POST') {
    const { ocrRow, confirmedRow } = await parseBody(req);
    if (!ocrRow || !confirmedRow) return jsonResponse(res, { error: 'ocrRow and confirmedRow required' }, 400);
    const result = shifu.learn(ocrRow, confirmedRow);
    return jsonResponse(res, result);
  }

  // Undo last learn
  if (path === '/api/undo' && req.method === 'POST') {
    const result = shifu.undo();
    return jsonResponse(res, result || { message: 'nothing to undo' });
  }

  // Structural invariance comparison
  if (path === '/api/compare-structure' && req.method === 'POST') {
    const { sentA, sentB } = await parseBody(req);
    if (!sentA || !sentB) return jsonResponse(res, { error: 'sentA and sentB required' }, 400);
    return jsonResponse(res, shifu.compareStructure(sentA, sentB));
  }

  // Score a sentence
  if (path === '/api/score' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    return jsonResponse(res, shifu.scoreSentence(text));
  }

  // Extract roles
  if (path === '/api/roles' && req.method === 'POST') {
    const { text } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    return jsonResponse(res, shifu.extractRoles(text));
  }

  // System stats
  if (path === '/api/stats') {
    return jsonResponse(res, {
      ...shifu.stats(),
      feedback: feedback.getMetrics(),
      metrics: metrics.summary(),
    });
  }

  // History — return metadata only, strip raw patient data
  if (path === '/api/history') {
    const history = shifu.getHistory(20).map(h => ({
      id: h.id,
      timestamp: h.timestamp,
      columns: h.columns || [],
      rejectedCount: h.rejectedCount || 0,
    }));
    return jsonResponse(res, history);
  }

  // Ingest raw text
  if (path === '/api/ingest' && req.method === 'POST') {
    const { text, format, columns, delimiter } = await parseBody(req);
    if (!text) return jsonResponse(res, { error: 'text required' }, 400);
    const result = ingestor.ingestRawText(text, { format, columns, delimiter });
    return jsonResponse(res, result);
  }

  // Upload a file (PDF, CSV, TXT, images) — multipart form or raw body
  if (path === '/api/upload' && req.method === 'POST') {
    const fs = require('fs');
    const os = require('os');
    const tmpDir = os.tmpdir();

    // Read raw body
    const chunks = [];
    await new Promise(resolve => {
      req.on('data', chunk => chunks.push(chunk));
      req.on('end', resolve);
    });
    const body = Buffer.concat(chunks);

    // Extract filename from Content-Disposition or query param
    const qs = url.parse(req.url, true).query;
    const rawFilename = qs.filename || 'upload' + Date.now();
    const correctNativeText = qs.correctNativeText === '1' || qs.correctNativeText === 'true';
    // Sanitize filename: strip path separators and traversal sequences, then verify
    const filename = require('path').basename(rawFilename).replace(/\.\./g, '_').replace(/[/\\]/g, '_');
    const tmpPath = require('path').join(tmpDir, 'shifu_' + filename);
    if (!tmpPath.startsWith(tmpDir)) return jsonResponse(res, { error: 'invalid filename' }, 400);

    // Check for multipart form data
    const contentType = req.headers['content-type'] || '';
    if (contentType.includes('multipart/form-data')) {
      let boundary = contentType.split('boundary=')[1];
      if (boundary) {
        // Strip quotes from boundary (RFC 2046 allows boundary="token")
        boundary = boundary.replace(/^["']|["']$/g, '').split(';')[0].trim();
        // Per RFC 2046, boundary delimiters appear as \r\n--boundary at line starts.
        // The first delimiter may omit the leading \r\n. We search for \r\n--boundary
        // to avoid false matches inside file content.
        const lineDelim = Buffer.from('\r\n--' + boundary);
        const firstDelim = Buffer.from('--' + boundary);
        const hdrSep = Buffer.from('\r\n\r\n');

        // Find the first boundary (may not have leading \r\n)
        let firstStart = body.indexOf(firstDelim);
        if (firstStart === -1) {
          return jsonResponse(res, { error: 'no boundary found in upload' }, 400);
        }

        // Find all parts by searching for \r\n--boundary (line-start only)
        const parts = [];
        let partStart = firstStart + firstDelim.length;
        // Skip past the \r\n after the first boundary line
        if (body[partStart] === 0x0D && body[partStart + 1] === 0x0A) partStart += 2;

        while (partStart < body.length) {
          const nextDelim = body.indexOf(lineDelim, partStart);
          const partEnd = nextDelim === -1 ? body.length : nextDelim;
          parts.push(body.slice(partStart, partEnd));
          if (nextDelim === -1) break;
          partStart = nextDelim + lineDelim.length;
          // Skip \r\n or -- after boundary
          if (body[partStart] === 0x2D && body[partStart + 1] === 0x2D) break; // closing --
          if (body[partStart] === 0x0D && body[partStart + 1] === 0x0A) partStart += 2;
        }

        for (const partBuf of parts) {
          const hdrEnd = partBuf.indexOf(hdrSep);
          if (hdrEnd === -1) continue;
          const headers = partBuf.slice(0, hdrEnd).toString('utf8');
          const fileData = partBuf.slice(hdrEnd + hdrSep.length);
          const fnMatch = headers.match(/filename="([^"]+)"/);
          if (fnMatch) {
            const realName = require('path').basename(fnMatch[1]).replace(/\.\./g, '_').replace(/[/\\]/g, '_');
            const realPath = require('path').join(tmpDir, 'shifu_' + realName);
            if (!realPath.startsWith(tmpDir)) continue;
            fs.writeFileSync(realPath, fileData);
            try {
              const doc = await ingestor.ingestFile(realPath, { correctNativeText });
              return jsonResponse(res, doc);
            } catch (err) {
              return jsonResponse(res, { error: err.message }, 500);
            } finally {
              try { fs.unlinkSync(realPath); } catch {}
            }
          }
        }
      }
      return jsonResponse(res, { error: 'no file found in upload' }, 400);
    }

    // Raw body upload
    fs.writeFileSync(tmpPath, body);
    try {
      const doc = await ingestor.ingestFile(tmpPath, { correctNativeText });
      return jsonResponse(res, doc);
    } catch (err) {
      return jsonResponse(res, { error: err.message }, 500);
    } finally {
      try { fs.unlinkSync(tmpPath); } catch {}
    }
  }

  jsonResponse(res, { error: 'not found' }, 404);
});

// ── HTML UI ───────────────────────────────────────────────────────
const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shifu OCR v2.0.0</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e17; color: #e0e6f0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a1f35, #0d1220); padding: 20px 30px; border-bottom: 1px solid #2a3050; }
  .header h1 { font-size: 1.5rem; color: #7eb8ff; }
  .header p { font-size: 0.85rem; color: #667; margin-top: 4px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab { padding: 10px 20px; background: #151a2e; border: 1px solid #2a3050; border-radius: 8px 8px 0 0; cursor: pointer; color: #889; font-size: 0.9rem; }
  .tab.active { background: #1e2540; color: #7eb8ff; border-bottom-color: #1e2540; }
  .panel { display: none; background: #1e2540; border: 1px solid #2a3050; border-radius: 0 8px 8px 8px; padding: 24px; }
  .panel.active { display: block; }
  label { display: block; font-size: 0.8rem; color: #778; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  input, textarea { width: 100%; padding: 10px 14px; background: #0d1220; border: 1px solid #2a3050; border-radius: 6px; color: #e0e6f0; font-size: 0.95rem; font-family: 'Consolas', monospace; margin-bottom: 12px; }
  textarea { min-height: 80px; resize: vertical; }
  button { padding: 10px 24px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; }
  button:hover { background: #1d4ed8; }
  button.secondary { background: #374151; }
  button.secondary:hover { background: #4b5563; }
  button.danger { background: #dc2626; }
  .result { background: #0d1220; border: 1px solid #2a3050; border-radius: 8px; padding: 16px; margin-top: 16px; font-family: 'Consolas', monospace; font-size: 0.85rem; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }
  .word { display: inline-block; padding: 2px 6px; margin: 2px; border-radius: 4px; }
  .word.exact { background: #064e3b; color: #6ee7b7; }
  .word.high_confidence { background: #164e63; color: #67e8f9; }
  .word.corrected_verify, .word.verify { background: #713f12; color: #fde68a; }
  .word.low_confidence, .word.unknown { background: #7f1d1d; color: #fca5a5; }
  .word.number, .word.title, .word.dosage, .word.room_code, .word.short, .word.punctuation, .word.passthrough { background: #1e293b; color: #94a3b8; }
  .word.digraph_corrected { background: #312e81; color: #c4b5fd; }
  .word.DANGER_medication_ambiguity { background: #dc2626; color: white; font-weight: bold; }
  .flag { display: inline-block; padding: 3px 8px; margin: 2px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
  .flag.danger { background: #dc2626; color: white; }
  .flag.warning { background: #d97706; color: white; }
  .flag.error { background: #ea580c; color: white; }
  .decision { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; font-weight: 700; margin-top: 8px; }
  .decision.accept { background: #064e3b; color: #6ee7b7; }
  .decision.verify { background: #713f12; color: #fde68a; }
  .decision.reject { background: #7f1d1d; color: #fca5a5; }
  .row-inputs { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { display: inline-block; background: #0d1220; padding: 8px 16px; border-radius: 6px; margin: 4px; }
  .stat .val { font-size: 1.3rem; font-weight: 700; color: #7eb8ff; }
  .stat .lbl { font-size: 0.7rem; color: #667; text-transform: uppercase; }
  .actions { display: flex; gap: 8px; margin-top: 12px; }
</style>
</head>
<body>
<div class="header">
  <h1>Shifu OCR v2.0.0</h1>
  <p>Medical OCR post-processing with adaptive learning, safety flags, and structural invariance</p>
</div>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('correct')">Correct Text</div>
    <div class="tab" onclick="switchTab('ward')">Ward Census</div>
    <div class="tab" onclick="switchTab('learn')">Learn</div>
    <div class="tab" onclick="switchTab('upload')">Upload File</div>
    <div class="tab" onclick="switchTab('structure')">Structure</div>
    <div class="tab" onclick="switchTab('stats')">Stats</div>
  </div>

  <!-- Tab 1: Correct Text -->
  <div id="correct" class="panel active">
    <label>OCR Text (paste raw OCR output)</label>
    <textarea id="ocrInput" placeholder="e.g. Levetiracetarn 5OOmg for seizure prophylaxis">Levetiracetarn 5OOmg for seizure prophylaxis</textarea>
    <button onclick="correctText()">Correct</button>
    <div id="correctResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 2: Ward Census Row -->
  <div id="ward" class="panel">
    <label>Ward Census Row (fill in columns)</label>
    <div class="row-inputs">
      <div><label>Patient</label><input id="wPatient" placeholder="e.g. Bader A1athoub"></div>
      <div><label>Diagnosis</label><input id="wDiagnosis" placeholder="e.g. lschemic str0ke"></div>
      <div><label>Doctor</label><input id="wDoctor" placeholder="e.g. Dr. Hisharn"></div>
      <div><label>Medication</label><input id="wMedication" placeholder="e.g. Levetiracetarn"></div>
      <div><label>Room</label><input id="wRoom" placeholder="e.g. 3O2A"></div>
      <div><label>Status</label><input id="wStatus" placeholder="e.g. Red"></div>
    </div>
    <div class="actions">
      <button onclick="correctRow()">Correct Row</button>
    </div>
    <div id="wardResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 3: Learn -->
  <div id="learn" class="panel">
    <label>Teach the system (OCR vs Confirmed)</label>
    <div class="row-inputs">
      <div>
        <label>OCR Column</label><input id="learnCol" value="Diagnosis">
        <label>OCR Text</label><input id="learnOcr" placeholder="e.g. str0ke">
      </div>
      <div>
        <label>&nbsp;</label><input disabled style="visibility:hidden">
        <label>Confirmed Text</label><input id="learnConfirmed" placeholder="e.g. stroke">
      </div>
    </div>
    <div class="actions">
      <button onclick="learnCorrection()">Learn</button>
      <button class="danger" onclick="undoLearn()">Undo Last</button>
    </div>
    <div id="learnResult" class="result" style="display:none"></div>
    <div style="margin-top:16px"><label>Correction History</label></div>
    <div id="historyResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab: Upload File -->
  <div id="upload" class="panel">
    <label>Upload a document (PDF, CSV, TSV, TXT, or image)</label>
    <div style="border: 2px dashed #2a3050; border-radius: 8px; padding: 40px; text-align: center; margin-bottom: 16px; cursor: pointer" id="dropZone" onclick="document.getElementById('fileInput').click()" ondrop="handleDrop(event)" ondragover="event.preventDefault(); this.style.borderColor='#7eb8ff'" ondragleave="this.style.borderColor='#2a3050'">
      <p style="color: #667; font-size: 1rem;">Drop a file here or click to browse</p>
      <p style="color: #445; font-size: 0.8rem; margin-top: 8px;">Supports: PDF, CSV, TSV, TXT, PNG, JPG, TIFF, BMP</p>
      <input type="file" id="fileInput" style="display:none" accept=".pdf,.csv,.tsv,.txt,.png,.jpg,.jpeg,.tiff,.tif,.bmp" onchange="uploadFile(this.files[0])">
    </div>
    <div id="uploadStatus" style="display:none; color: #7eb8ff; margin-bottom: 12px;"></div>
    <div id="uploadResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 4: Structure -->
  <div id="structure" class="panel">
    <label>Compare Structural Invariance</label>
    <input id="structA" placeholder="e.g. doctor treats patient with medication">
    <input id="structB" placeholder="e.g. patient was treated by the doctor with medication">
    <div class="actions">
      <button onclick="compareStructure()">Compare</button>
      <button class="secondary" onclick="extractRoles()">Extract Roles (sentence A)</button>
    </div>
    <div id="structResult" class="result" style="display:none"></div>
  </div>

  <!-- Tab 5: Stats -->
  <div id="stats" class="panel">
    <button onclick="loadStats()">Refresh Stats</button>
    <div id="statsResult" class="result" style="display:none"></div>
  </div>
</div>

<script>
const API = '';

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  event.target.classList.add('active');
}

async function api(path, body) {
  const res = await fetch(API + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function renderWords(words) {
  if (!words || !words.length) return '';
  return words.map(w => {
    const cls = esc(w.flag || 'unknown');
    const orig = esc(w.original);
    const corr = esc(w.corrected || w.original);
    const tip = w.original !== w.corrected ? orig + ' -> ' + corr : orig;
    return '<span class="word ' + cls + '" title="' + tip + ' [' + cls + '] conf:' + (w.confidence||0).toFixed(2) + '">' + corr + '</span>';
  }).join(' ');
}

function renderFlags(flags) {
  if (!flags || !flags.length) return '';
  return flags.map(f => '<span class="flag ' + esc(f.severity||'warning') + '">' + esc(f.message||f.status) + '</span>').join(' ');
}

function show(id, html) {
  const el = document.getElementById(id);
  el.style.display = 'block';
  el.innerHTML = html;
}

async function correctText() {
  const text = document.getElementById('ocrInput').value;
  if (!text) return;
  const r = await api('/api/correct', { text });
  let html = '<b>Output:</b> ' + renderWords(r.words) + '\\n\\n';
  html += '<b>Text:</b> ' + esc(r.output||'') + '\\n';
  html += '<b>Decision:</b> <span class="decision ' + (r.decision||'verify') + '">' + (r.decision||'?') + '</span>\\n';
  html += '<b>Coherence:</b> ' + (r.coherence||0).toFixed(4) + '\\n';
  html += '<b>Avg Confidence:</b> ' + (r.avgConfidence||0) + '\\n';
  if (r.safetyFlags && r.safetyFlags.length) html += '\\n<b>Safety Flags:</b> ' + renderFlags(r.safetyFlags);
  show('correctResult', html);
}

async function correctRow() {
  const row = {};
  for (const col of ['Patient','Diagnosis','Doctor','Medication','Room','Status']) {
    const v = document.getElementById('w' + col).value;
    if (v) row[col] = v;
  }
  if (!Object.keys(row).length) return;
  const r = await api('/api/correct-row', { row });
  let html = '<b>Decision:</b> <span class="decision ' + (r.decision||'verify') + '">' + (r.decision||'?') + '</span>\\n\\n';
  for (const [col, data] of Object.entries(r.corrected || {})) {
    html += '<b>' + esc(col) + ':</b> ' + esc(data.input||'') + ' -> ' + renderWords(data.words) + '\\n';
  }
  if (r.safetyFlags && r.safetyFlags.length) html += '\\n<b>Safety Flags:</b>\\n' + renderFlags(r.safetyFlags);
  show('wardResult', html);
}

async function learnCorrection() {
  const col = document.getElementById('learnCol').value || 'Diagnosis';
  const ocr = document.getElementById('learnOcr').value;
  const confirmed = document.getElementById('learnConfirmed').value;
  if (!ocr || !confirmed) return;
  const ocrRow = {}; ocrRow[col] = ocr;
  const confirmedRow = {}; confirmedRow[col] = confirmed;
  const r = await api('/api/learn', { ocrRow, confirmedRow });
  let html = '<b>Accepted:</b> ' + r.accepted + '\\n';
  if (r.rejected && r.rejected.length) html += '<b>Rejected:</b> ' + esc(JSON.stringify(r.rejected)) + '\\n';
  if (r.reason) html += '<b>Reason:</b> ' + esc(r.reason) + '\\n';
  show('learnResult', html);
  loadHistory();
}

async function undoLearn() {
  const r = await api('/api/undo', {});
  show('learnResult', '<b>Undo:</b> ' + esc(JSON.stringify(r, null, 2)));
  loadHistory();
}

async function loadHistory() {
  const r = await api('/api/history');
  if (!r || !r.length) { show('historyResult', 'No corrections yet.'); return; }
  let html = r.map((h,i) => '#' + esc(h.id) + ' [' + esc(h.timestamp) + '] columns: ' + esc((h.columns||[]).join(', '))).join('\\n');
  show('historyResult', html);
}

async function compareStructure() {
  const a = document.getElementById('structA').value;
  const b = document.getElementById('structB').value;
  if (!a || !b) return;
  const r = await api('/api/compare-structure', { sentA: a, sentB: b });
  let html = '<b>Invariance Score:</b> ' + (r.invariance||0).toFixed(4) + '\\n\\n';
  html += '<b>Sentence A roles:</b>\\n';
  html += '  Agent: ' + esc(r.rolesA?.agent||'-') + ', Action: ' + esc(r.rolesA?.action||'-') + ', Patient: ' + esc(r.rolesA?.patient||'-') + ' (passive: ' + (r.rolesA?.passive||false) + ')\\n';
  html += '<b>Sentence B roles:</b>\\n';
  html += '  Agent: ' + esc(r.rolesB?.agent||'-') + ', Action: ' + esc(r.rolesB?.action||'-') + ', Patient: ' + esc(r.rolesB?.patient||'-') + ' (passive: ' + (r.rolesB?.passive||false) + ')\\n\\n';
  const c = r.comparison || {};
  html += '<b>Agent similarity:</b> ' + (c.agentSim||0).toFixed(3) + '\\n';
  html += '<b>Action similarity:</b> ' + (c.actionSim||0).toFixed(3) + '\\n';
  html += '<b>Patient similarity:</b> ' + (c.patientSim||0).toFixed(3) + '\\n';
  if (c.swapDetected) html += '\\n<span class="flag danger">ROLE SWAP DETECTED</span>';
  show('structResult', html);
}

async function extractRoles() {
  const a = document.getElementById('structA').value;
  if (!a) return;
  const r = await api('/api/roles', { text: a });
  show('structResult', esc(JSON.stringify(r, null, 2)));
}

async function loadStats() {
  const r = await api('/api/stats');
  let html = '<b>Core Engine</b>\\n';
  html += '  Vocabulary: ' + (r.core?.vocabulary||0) + ' words\\n';
  html += '  Sentences fed: ' + (r.core?.sentences||0) + '\\n';
  html += '  Resonance pairs: ' + (r.core?.resonancePairs||0) + '\\n\\n';
  html += '<b>Learning Engine</b>\\n';
  html += '  Total corrections: ' + (r.learning?.totalCorrections||0) + '\\n';
  html += '  Confusion pairs: ' + (r.learning?.confusionPairs||0) + '\\n';
  html += '  Vocabulary size: ' + (r.learning?.vocabularySize||0) + '\\n';
  html += '  Learned words: ' + (r.learning?.learnedWords||0) + '\\n';
  html += '  Context chains: ' + (r.learning?.contextChains||0) + '\\n\\n';
  if (r.learning?.topConfusions?.length) {
    html += '<b>Top Confusions:</b>\\n';
    for (const c of r.learning.topConfusions) html += '  ' + c.pair + ' (' + c.count + 'x, cost: ' + c.cost.toFixed(3) + ')\\n';
  }
  show('statsResult', html);
}

function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.borderColor = '#2a3050';
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  if (!file) return;
  document.getElementById('uploadStatus').style.display = 'block';
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  document.getElementById('uploadStatus').innerHTML = 'Processing <b>' + esc(file.name) + '</b> (' + sizeMB + ' MB)...<br><span style="color:#94a3b8;font-size:0.85rem">Large PDFs (1000+ pages) may take a few minutes for text extraction.</span>';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(API + '/api/upload?filename=' + encodeURIComponent(file.name), {
      method: 'POST',
      body: formData,
    });
    const r = await res.json();
    document.getElementById('uploadStatus').textContent = 'Done: ' + file.name;

    let html = '<b>Type:</b> ' + esc(r.type||'unknown') + '\\n';
    if (r.extractionMethod) {
      html += '<b>Extraction:</b> ' + esc(r.extractionMethod) + '\\n';
      if (r.extractionDetail) html += '<span style="color:#94a3b8;font-size:0.85rem">' + esc(r.extractionDetail) + '</span>\\n';
      if (r.pageCount) html += '<b>Pages:</b> ' + r.pageCount + '\\n';
      if (r.charCount) html += '<b>Characters:</b> ' + r.charCount.toLocaleString() + '\\n';
    }
    if (r.sourceMode) html += '<b>Mode:</b> ' + esc(r.sourceMode) + '\\n';
    if (r.error) html += '<b style="color:#fca5a5">Error:</b> ' + esc(r.error) + '\\n';
    if (r.overallDecision) html += '<b>Decision:</b> <span class="decision ' + r.overallDecision + '">' + r.overallDecision + '</span>\\n';

    // Raw extracted text (what the PDF/file reader saw)
    if (r.rawText) {
      html += '\\n<b style="color:#94a3b8">--- Raw Extracted Text ---</b>\\n';
      html += '<span style="color:#667">' + r.rawText.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
    }
    if (r.rawLines && r.rawLines.length && !r.rawText) {
      html += '\\n<b style="color:#94a3b8">--- Raw Extracted Lines ---</b>\\n';
      for (const rl of r.rawLines) {
        html += '<span style="color:#667">' + rl.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
      }
    }

    // Corrected lines (text/PDF)
    if (r.lines && r.lines.length) {
      html += '\\n<b style="color:#7eb8ff">--- Corrected Output ---</b>\\n';
      for (let li = 0; li < r.lines.length; li++) {
        const line = r.lines[li];
        const rawLine = (r.rawLines && r.rawLines[li]) || '';
        if (rawLine) html += '<span style="color:#445;font-size:0.75rem">RAW: ' + rawLine.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>\\n';
        if (line && line.words) {
          html += renderWords(line.words) + '\\n';
          if (line.safetyFlags && line.safetyFlags.length) html += renderFlags(line.safetyFlags) + '\\n';
        } else if (line && line.corrected) {
          html += esc(line.corrected) + '\\n';
        }
        html += '\\n';
      }
    }

    // Rows (CSV)
    if (r.rows && r.rows.length) {
      html += '\\n<b>Rows (' + r.rows.length + '):</b>\\n';
      for (let i = 0; i < r.rows.length; i++) {
        const row = r.rows[i];
        html += '\\n<b>Row ' + (i+1) + '</b> <span class="decision ' + (row.decision||'verify') + '">' + (row.decision||'?') + '</span>\\n';
        if (row.corrected) {
          for (const [col, data] of Object.entries(row.corrected)) {
            html += '  <b>' + esc(col) + ':</b> ' + esc(data.input||'') + ' -> ';
            if (data.words) html += renderWords(data.words);
            else html += esc(data.output||data);
            html += '\\n';
          }
        }
        if (row.safetyFlags && row.safetyFlags.length) {
          html += '  ' + renderFlags(row.safetyFlags) + '\\n';
        }
      }
    }

    // Summary
    if (r.summary) {
      html += '\\n<b>Summary:</b> ' + r.summary.total + ' rows, ' + r.summary.accepted + ' accepted, ' + r.summary.needsVerification + ' verify, ' + r.summary.rejected + ' rejected';
    }

    show('uploadResult', html);
  } catch (err) {
    document.getElementById('uploadStatus').textContent = 'Error: ' + err.message;
    show('uploadResult', '<b>Error:</b> ' + err.message);
  }
}
</script>
</body>
</html>`;

server.listen(PORT, () => {
  console.log(`\n  Shifu OCR running at http://localhost:${PORT}\n`);
  console.log('  Tabs:');
  console.log('    Correct Text  — paste OCR output, get corrected text + safety flags');
  console.log('    Ward Census   — correct a full ward census row');
  console.log('    Learn         — teach the system from nurse corrections');
  console.log('    Structure     — compare sentences for structural invariance');
  console.log('    Stats         — view system stats, confusion pairs, metrics\n');
});
